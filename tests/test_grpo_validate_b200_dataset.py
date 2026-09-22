"""Unit tests for fixed B200 Holmes-16k contract constants and counts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TARGET = SRC / "grpo_validate_b200_dataset.py"
spec = importlib.util.spec_from_file_location("b200_dataset_gate_under_test", TARGET)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class B200DatasetGateTests(unittest.TestCase):
    def test_expected_source_contract_is_holmes_16k(self) -> None:
        self.assertEqual(module.EXPECTED_SOURCE_RECORDS, 16916)
        self.assertEqual(module.EXPECTED_SOURCE_COUNTS, {"image": 8765, "video": 8151})

    def test_known_reduced_contract_is_explicitly_video_only_missing(self) -> None:
        self.assertEqual(module.EXPECTED_REDUCED_RECORDS, 15365)
        self.assertEqual(module.EXPECTED_REDUCED_COUNTS, {"image": 8765, "video": 6600})
        self.assertEqual(
            module.EXPECTED_SOURCE_RECORDS - module.EXPECTED_REDUCED_RECORDS,
            1551,
        )

    def test_modality_counts_preserve_unsupported_visibility(self) -> None:
        self.assertEqual(
            module.modality_counts(
                [{"data_type": "image"}, {"data_type": "video"}, {"data_type": "audio"}, {}]
            ),
            {"image": 1, "video": 1, "unsupported": 2},
        )

    def test_known_reduced_manifest_requires_explicit_opt_in(self) -> None:
        source_records = [
            {"problem_id": index, "data_type": "image", "path": f"./image/{index}.jpg"}
            for index in range(module.EXPECTED_SOURCE_COUNTS["image"])
        ] + [
            {"problem_id": 10_000 + index, "data_type": "video", "path": f"./video/{index}.mp4"}
            for index in range(module.EXPECTED_SOURCE_COUNTS["video"])
        ]
        selected_records = source_records[: module.EXPECTED_SOURCE_COUNTS["image"]] + source_records[
            module.EXPECTED_SOURCE_COUNTS["image"] : module.EXPECTED_SOURCE_COUNTS["image"]
            + module.EXPECTED_REDUCED_COUNTS["video"]
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "source.json"
            selected = directory / "selected.json"
            manifest = directory / "selected.json.manifest.json"
            source.write_text(json.dumps(source_records), encoding="utf-8")
            selected.write_text(json.dumps(selected_records), encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "data_type": "all",
                        "source_records": len(source_records),
                        "selected_records": len(selected_records),
                        "source_sha256": module.sha256_file(source),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "B200 Holmes gate"):
                module.path_contract(source, selected, manifest, allow_reduced=False)
            result = module.path_contract(source, selected, manifest, allow_reduced=True)
        self.assertFalse(result["strict_holmes_16k"])
        self.assertEqual(result["reproduction_class"], "reduced-media-subset")
        self.assertEqual(result["missing_by_modality"], {"image": 0, "video": 1551})


if __name__ == "__main__":
    unittest.main()
