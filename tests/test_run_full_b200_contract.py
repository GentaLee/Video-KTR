"""Static contract checks for the manual B200 full launcher."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (ROOT / "run_full_b200.sh").read_text(encoding="utf-8")


class B200FullLauncherContractTests(unittest.TestCase):
    def test_locks_upstream_shape_and_current_selector_definition(self) -> None:
        for required in (
            "nproc_per_node=8",
            "max_prompt_length=16384",
            "max_completion_length=768",
            "num_generations=8",
            "max_pixels=401408",
            "nframes=8",
            "temporal_permutations=1",
            "temporal_include_reverse=false",
            "--selection_scope per_completion",
            "--temporal_include_reverse false",
            "--video_ktr \"${video_ktr_flag}\"",
        ):
            self.assertIn(required, LAUNCHER)

    def test_uses_mixed_preflight_and_explicit_data_downgrade_gates(self) -> None:
        self.assertIn("--data-type all", LAUNCHER)
        self.assertIn("grpo_verify_media_decode.py", LAUNCHER)
        self.assertIn("grpo_validate_b200_dataset.py", LAUNCHER)
        self.assertIn("B200_ALLOW_REDUCED_HOLMES", LAUNCHER)
        self.assertIn("B200_ALLOW_DECODER_FILTERED", LAUNCHER)
        self.assertIn("FORCE_QWENVL_VIDEO_READER=torchvision", LAUNCHER)

    def test_keeps_protection_running_until_gpu_phase_and_restores_by_controller(self) -> None:
        self.assertLess(
            LAUNCHER.index("phase 6/8: CPU-only mixed image/video"),
            LAUNCHER.index("phase 7/8: pausing keep-alive only for B200 GPU work"),
        )
        self.assertIn('bash "${keepalive_launcher}" stop', LAUNCHER)
        self.assertIn('bash "${keepalive_launcher}" start', LAUNCHER)
        self.assertIn("wait_for_gpu_quiescence \"before keep-alive restore\"", LAUNCHER)
        self.assertNotIn("pkill", LAUNCHER)

    def test_requires_complete_eight_gpu_resource_evidence(self) -> None:
        self.assertIn("validate_resource_summary", LAUNCHER)
        self.assertIn("eight GPUs sampled with no monitor errors", LAUNCHER)
        self.assertIn("GPU resource monitor exited before training started", LAUNCHER)

    def test_full_requires_fa2_and_distinct_qwen_media_ids(self) -> None:
        self.assertIn("VIDEO_KTR_REQUIRE_ATTN_IMPLEMENTATION=flash_attention_2", LAUNCHER)
        self.assertIn("VIDEO_KTR_QWEN_FA2_ROTARY_DTYPE_COMPAT=1", LAUNCHER)
        self.assertIn("image_id == video_id", LAUNCHER)
        self.assertIn('b"sm_100"', LAUNCHER)


if __name__ == "__main__":
    unittest.main()
