"""Static contract checks for the manual B200 launchers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (ROOT / "scripts/run_full_b200.sh").read_text(encoding="utf-8")
SMOKE_LAUNCHER = (ROOT / "scripts/somke-b200.sh").read_text(encoding="utf-8")


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

    def test_command_line_decode_timeout_survives_private_env_file(self) -> None:
        """The operator's 600-second override must win over a stale local default."""
        marker = 'timestamp="$(date -u +%Y%m%dT%H%M%SZ)"'
        self.assertIn(marker, LAUNCHER)
        prefix = LAUNCHER.split(marker, 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / "b200.env"
            env_file.write_text(
                "B200_MEDIA_DECODE_TIMEOUT_SECONDS=120\n"
                "B200_MEDIA_DECODE_WORKERS=4\n"
                "VARIANT=baseline\n"
                "MAX_STEPS=1\n",
                encoding="utf-8",
            )
            probe = root / "probe.sh"
            probe.write_text(
                prefix
                + '\nprintf "%s,%s,%s,%s\\n" "$B200_MEDIA_DECODE_TIMEOUT_SECONDS" "$B200_MEDIA_DECODE_WORKERS" "$VARIANT" "$MAX_STEPS"\n',
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                B200_ROOT=str(root),
                B200_ENV_FILE=str(env_file),
                B200_MEDIA_DECODE_TIMEOUT_SECONDS="600",
                B200_MEDIA_DECODE_WORKERS="8",
                VARIANT="ktr",
                MAX_STEPS="-1",
            )
            result = subprocess.run(
                ["bash", str(probe)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.stdout.strip(), "600,8,ktr,-1")
        self.assertIn("media_decode_timeout_seconds=%s", LAUNCHER)
        self.assertIn("media_decode_workers=%s", LAUNCHER)

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

    def test_training_entrypoint_imports_on_eight_cpu_only_ranks_before_data_work(self) -> None:
        self.assertIn("eight-rank CPU-only training-entrypoint import gate", LAUNCHER)
        self.assertIn('env CUDA_VISIBLE_DEVICES=""', LAUNCHER)
        self.assertIn("grpo_validate_b200_training_import.py", LAUNCHER)
        self.assertIn('export PYTHON_EXEC="${python_bin}"', LAUNCHER)
        self.assertIn('--expected-python "${python_bin}"', LAUNCHER)
        self.assertLess(
            LAUNCHER.index("eight-rank CPU-only training-entrypoint import gate"),
            LAUNCHER.index("phase 5/8: path-verifying"),
        )
        self.assertIn("first worker traceback", LAUNCHER)

    def test_smoke_confirms_pause_before_marking_keepalive_paused(self) -> None:
        """A failed controller stop must be reconciled before cleanup restores."""
        stop_request = SMOKE_LAUNCHER.index("keepalive_stop_requested=1")
        controller_stop = SMOKE_LAUNCHER.index(
            'bash "${keepalive_launcher}" stop', stop_request
        )
        pause_confirmation = SMOKE_LAUNCHER.index(
            'wait_for_keepalive_count 0 "pause"', controller_stop
        )
        confirmed_pause = SMOKE_LAUNCHER.index(
            "keepalive_paused=1", pause_confirmation
        )
        self.assertLess(stop_request, controller_stop)
        self.assertLess(controller_stop, pause_confirmation)
        self.assertLess(pause_confirmation, confirmed_pause)
        self.assertIn(
            "controller stop request left no verified keep-alive process",
            SMOKE_LAUNCHER,
        )


if __name__ == "__main__":
    unittest.main()
