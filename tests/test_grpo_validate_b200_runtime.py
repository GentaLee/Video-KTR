"""Focused contract tests for the B200 checkpoint/runtime identity gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

WRITER_PATH = SRC / "grpo_write_model_sha256_manifest.py"
writer_spec = importlib.util.spec_from_file_location("b200_model_writer_under_test", WRITER_PATH)
assert writer_spec is not None and writer_spec.loader is not None
writer = importlib.util.module_from_spec(writer_spec)
sys.modules[writer_spec.name] = writer
writer_spec.loader.exec_module(writer)

VALIDATOR_PATH = SRC / "grpo_validate_b200_runtime.py"
validator_spec = importlib.util.spec_from_file_location("b200_runtime_gate_under_test", VALIDATOR_PATH)
assert validator_spec is not None and validator_spec.loader is not None
validator = importlib.util.module_from_spec(validator_spec)
sys.modules[validator_spec.name] = validator
validator_spec.loader.exec_module(validator)

PINNED_MANIFEST = (
    ROOT
    / "manifests"
    / "video-r1-qwen25vl-7b-cot-sft-f71f0f1e22c015007fccd080eef87824fe292a10.json"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def make_manifest(files: dict[str, bytes]) -> dict[str, object]:
    return {
        "schema": writer.MODEL_MANIFEST_SCHEMA,
        "source": {
            "repository": "fixture/repository",
            "revision": "fixture-revision",
            "tree_url": "https://example.invalid/tree",
        },
        "files": {
            name: {"sha256": sha256(payload), "size_bytes": len(payload), "role": "fixture"}
            for name, payload in files.items()
        },
    }


class B200RuntimeGateTests(unittest.TestCase):
    def test_runtime_gate_rejects_platform_venv_even_if_python_binary_resolves_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            project_venv = temporary / "project-venv"
            platform_venv = temporary / "platform-venv"
            for venv in (project_venv, platform_venv):
                (venv / "bin").mkdir(parents=True)
                (venv / "bin" / "python").symlink_to(sys.executable)
            self.assertEqual(
                (project_venv / "bin" / "python").resolve(),
                (platform_venv / "bin" / "python").resolve(),
            )
            bridge_torch = temporary / "bridge" / "torch.py"
            bridge_vision = temporary / "bridge" / "torchvision.py"
            bridge_torch.parent.mkdir()
            bridge_torch.touch()
            bridge_vision.touch()
            manifest = {
                "schema": validator.EXPECTED_ENVIRONMENT_SCHEMA,
                "repo_root": str(temporary),
                "python_executable": str(project_venv / "bin" / "python"),
                "venv": str(project_venv),
                "versions": {"av": validator.EXPECTED_RUNTIME["av"], "deepspeed": validator.EXPECTED_RUNTIME["deepspeed"]},
                "torch": {"version": validator.EXPECTED_RUNTIME["torch"], "cuda": "13.1", "file": str(bridge_torch)},
                "torchvision": {"version": validator.EXPECTED_RUNTIME["torchvision"], "file": str(bridge_vision)},
            }
            manifest_path = temporary / "environment.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            locations = {
                "python_executable": str(platform_venv / "bin" / "python"),
                "venv": str(platform_venv),
                "torch_file": str(bridge_torch),
                "torchvision_file": str(bridge_vision),
                "torch_cuda": "13.1",
            }
            with mock.patch.object(validator, "observed_runtime", return_value=(validator.EXPECTED_RUNTIME, locations)):
                with self.assertRaisesRegex(RuntimeError, "environment manifest venv mismatch"):
                    validator.validate_environment_identity(manifest_path, temporary)

    def test_pinned_manifest_is_path_free_and_has_expected_shape(self) -> None:
        manifest = writer.load_model_manifest(PINNED_MANIFEST)
        self.assertEqual(manifest["schema"], writer.MODEL_MANIFEST_SCHEMA)
        self.assertEqual(manifest["source"], validator.EXPECTED_MODEL_SOURCE)
        self.assertEqual(len(manifest["files"]), 19)
        self.assertEqual(
            sum(record["role"] == "weight-shard" for record in manifest["files"].values()), 4
        )
        for name, record in manifest["files"].items():
            self.assertEqual(name, Path(name).name)
            self.assertNotIn("/", name)
            self.assertEqual(len(record["sha256"]), 64)

    def test_manifest_parser_rejects_path_escape_and_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            payload = make_manifest({"safe.txt": b"safe"})
            payload["files"]["../escape"] = payload["files"].pop("safe.txt")
            path = temporary / "escape.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "top-level model-relative"):
                writer.load_model_manifest(path)

            payload = make_manifest({"safe.txt": b"safe"})
            payload["files"]["safe.txt"]["unexpected"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly sha256, size_bytes, role"):
                writer.load_model_manifest(path)

            valid = temporary / "valid.json"
            valid.write_text(json.dumps(make_manifest({"safe.txt": b"safe"})), encoding="utf-8")
            link = temporary / "manifest-link.json"
            link.symlink_to(valid.name)
            with self.assertRaisesRegex(RuntimeError, "refusing symlinked trusted model manifest"):
                writer.load_model_manifest(link)

    def test_file_validation_rejects_extra_missing_and_symlinked_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            model = temporary / "model"
            model.mkdir()
            contents = {"config.json": b"config", "tokenizer.json": b"tokenizer"}
            for name, payload in contents.items():
                (model / name).write_bytes(payload)
            manifest_path = temporary / "trusted.json"
            manifest_path.write_text(json.dumps(make_manifest(contents)), encoding="utf-8")
            manifest = writer.load_model_manifest(manifest_path)
            self.assertEqual(len(writer.validate_model_files(model, manifest)), 2)

            (model / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not cover exactly"):
                writer.validate_model_files(model, manifest)
            (model / "unexpected.txt").unlink()

            (model / "tokenizer.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "does not cover exactly"):
                writer.validate_model_files(model, manifest)
            (model / "tokenizer.json").symlink_to("config.json")
            with self.assertRaisesRegex(RuntimeError, "refusing symlinked"):
                writer.validate_model_files(model, manifest)

    def test_writer_only_materializes_a_reference_that_matches_every_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            model = temporary / "model"
            model.mkdir()
            contents = {"config.json": b"config", "tokenizer.json": b"tokenizer"}
            for name, payload in contents.items():
                (model / name).write_bytes(payload)
            reference = temporary / "reference.json"
            reference.write_text(json.dumps(make_manifest(contents)), encoding="utf-8")
            output = temporary / "verified-copy.json"
            result = writer.write_manifest(model, reference, output)
            self.assertEqual(result["file_count"], 2)
            self.assertEqual(writer.load_model_manifest(output), writer.load_model_manifest(reference))

            (model / "config.json").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "model size mismatch|model SHA-256 mismatch"):
                writer.write_manifest(model, reference, output)

    def test_full_gate_locks_exact_b200_runtime_and_rechecks_before_gpu_pause(self) -> None:
        self.assertEqual(validator.EXPECTED_RUNTIME["python"], "3.12.3")
        self.assertEqual(validator.EXPECTED_RUNTIME["deepspeed"], "0.15.4")
        launcher = (ROOT / "run_full_b200.sh").read_text(encoding="utf-8")
        self.assertIn(PINNED_MANIFEST.name, launcher)
        self.assertIn("B200 smoke provenance recheck before GPU pause", launcher)
        self.assertIn("B200 model/environment recheck before GPU pause", launcher)
        self.assertLess(
            launcher.index("B200 smoke provenance recheck before GPU pause"),
            launcher.index("phase 7/8: pausing keep-alive only for B200 GPU work"),
        )


if __name__ == "__main__":
    unittest.main()
