"""Contract tests for the B200 smoke provenance gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TARGET = SRC / "grpo_validate_b200_smoke.py"
spec = importlib.util.spec_from_file_location("b200_smoke_gate_under_test", TARGET)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class B200SmokeGateContractTests(unittest.TestCase):
    def test_profile_locks_requested_alignment(self) -> None:
        self.assertEqual(module.REQUIRED_PROFILE["nframes"], "8")
        self.assertEqual(module.REQUIRED_PROFILE["max_pixels"], "401408")
        self.assertEqual(module.REQUIRED_PROFILE["selection_scope"], "per_completion")
        self.assertEqual(module.REQUIRED_PROFILE["delta"], "absolute")
        self.assertEqual(module.REQUIRED_PROFILE["temporal_permutations"], "1")
        self.assertEqual(module.REQUIRED_PROFILE["temporal_include_reverse"], "false")

    def test_critical_sources_cover_qwen_and_token_selection_paths(self) -> None:
        required = set(module.CRITICAL_SOURCES)
        self.assertIn("src/r1-v/src/open_r1/trainer/qwen25vl_fa2_compat.py", required)
        self.assertIn("src/r1-v/src/open_r1/trainer/video_ktr_grpo_trainer.py", required)
        self.assertIn("src/r1-v/src/open_r1/trainer/ktr_token_utils.py", required)

    def test_gate_source_requires_real_attention_log_evidence(self) -> None:
        source = TARGET.read_text(encoding="utf-8")
        self.assertIn("requested_attention=flash_attention_2", source)
        self.assertIn("resolved_attention=flash_attention_2", source)

    def test_key_value_parser_ignores_non_assignment_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory) / "values.txt"
            temporary.write_text("plain\na=b=c\nempty=\n", encoding="utf-8")
            self.assertEqual(module.parse_key_value(temporary), {"a": "b=c", "empty": ""})


if __name__ == "__main__":
    unittest.main()
