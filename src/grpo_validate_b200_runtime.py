#!/usr/bin/env python3
"""Fail-closed B200 checkpoint and isolated-runtime identity gate.

This is CPU-only: it hashes the exact committed model manifest and imports the
same Python packages torchrun will use, but never calls ``torch.cuda`` or loads
weights.  The full launcher runs it while the GPU keep-alive is still active.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from grpo_write_model_sha256_manifest import (
    is_relative_to,
    load_model_manifest,
    top_level_regular_files,
    validate_model_files,
)


EXPECTED_RUNTIME = {
    "python": "3.12.3",
    "torch": "2.11.0a0+eb65b36914.nv26.02",
    "torchvision": "0.25.0a0+1e53952f.nv26.02.44259020",
    "av": "14.2.0",
    "deepspeed": "0.15.4",
}
EXPECTED_ENVIRONMENT_SCHEMA = "video-ktr-b200-environment/v1"
EXPECTED_MODEL_SOURCE = {
    "repository": "Video-R1/Qwen2.5-VL-7B-COT-SFT",
    "revision": "f71f0f1e22c015007fccd080eef87824fe292a10",
    "tree_url": "https://huggingface.co/api/models/Video-R1/Qwen2.5-VL-7B-COT-SFT/tree/f71f0f1e22c015007fccd080eef87824fe292a10?recursive=true&expand=true",
}
EXPECTED_MODEL_FILE_COUNT = 19
EXPECTED_WEIGHT_SHARD_COUNT = 4


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is unavailable: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def require_equal(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} mismatch: observed={observed!r} expected={expected!r}")


def validate_model_identity(model_path: Path, manifest_path: Path) -> dict[str, Any]:
    model_root = model_path.resolve(strict=True)
    if not model_root.is_dir():
        raise RuntimeError(f"model path is not a directory: {model_root}")
    if manifest_path.is_symlink():
        raise RuntimeError(f"refusing symlinked model integrity manifest: {manifest_path}")
    manifest_file = manifest_path.resolve(strict=True)
    # A committed, path-free manifest normally lives in the source checkout.
    # Refusing a manifest inside the model directory eliminates self-hashing
    # and makes any explicit override equally constrained.
    if is_relative_to(manifest_file, model_root):
        raise RuntimeError(
            "model integrity manifest must live outside the model directory so it cannot validate itself: "
            f"{manifest_file}"
        )
    manifest = load_model_manifest(manifest_file)
    require_equal("trusted model source", manifest["source"], EXPECTED_MODEL_SOURCE)
    files = manifest["files"]
    require_equal("trusted model file count", len(files), EXPECTED_MODEL_FILE_COUNT)
    role_counts: dict[str, int] = {}
    for record in files.values():
        role = str(record["role"])
        role_counts[role] = role_counts.get(role, 0) + 1
    require_equal("trusted model weight-shard count", role_counts.get("weight-shard", 0), EXPECTED_WEIGHT_SHARD_COUNT)
    require_equal("trusted model weight-index count", role_counts.get("weight-index", 0), 1)
    require_equal("trusted model model-config count", role_counts.get("model-config", 0), 1)
    if role_counts.get("tokenizer", 0) < 1:
        raise RuntimeError("trusted model manifest has no tokenizer file")
    if "config.json" not in files or "model.safetensors.index.json" not in files:
        raise RuntimeError("trusted model manifest must include config.json and model.safetensors.index.json")

    # Validate the index structure before hashing the large shards.  This
    # rejects a local model whose four filenames happen to look plausible but
    # are not the checkpoint referenced by its own safetensors index.
    index = load_json_object(model_root / "model.safetensors.index.json", "model safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model safetensors index has no non-empty weight_map")
    if not all(isinstance(value, str) for value in weight_map.values()):
        raise RuntimeError("model safetensors index has non-string shard names")
    indexed_shards = set(weight_map.values())
    if len(indexed_shards) != EXPECTED_WEIGHT_SHARD_COUNT:
        raise RuntimeError(
            f"model safetensors index must reference {EXPECTED_WEIGHT_SHARD_COUNT} shards, "
            f"found {len(indexed_shards)}"
        )
    manifest_shards = {name for name, record in files.items() if record["role"] == "weight-shard"}
    actual_shards = {entry.name for entry in top_level_regular_files(model_root) if entry.name.endswith(".safetensors")}
    if indexed_shards != manifest_shards or indexed_shards != actual_shards:
        raise RuntimeError(
            "model safetensors index/shard set mismatch: "
            + json.dumps(
                {
                    "index_only": sorted(indexed_shards - manifest_shards),
                    "manifest_only": sorted(manifest_shards - indexed_shards),
                    "directory_only": sorted(actual_shards - indexed_shards),
                },
                sort_keys=True,
            )
        )
    records = validate_model_files(model_root, manifest)
    return {
        "model_path": str(model_root),
        "model_integrity_manifest": str(manifest_file),
        "model_integrity_manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "source": manifest["source"],
        "top_level_file_count": len(records),
        "indexed_safetensors_shards": sorted(indexed_shards),
        "role_counts": role_counts,
        "files": records,
    }


def observed_runtime() -> tuple[dict[str, str], dict[str, str]]:
    # These imports check the actual isolated runtime path inherited by
    # torchrun.  This function intentionally has no torch.cuda call.
    import av
    import deepspeed
    import torch
    import torchvision

    values = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "av": av.__version__,
        "deepspeed": deepspeed.__version__,
    }
    locations = {
        "python_executable": str(Path(sys.executable).resolve()),
        "venv": str(Path(sys.prefix).resolve()),
        "torch_file": str(Path(torch.__file__).resolve()),
        "torchvision_file": str(Path(torchvision.__file__).resolve()),
        "torch_cuda": str(torch.version.cuda),
    }
    return values, locations


def validate_environment_identity(environment_manifest_path: Path, project_root: Path) -> dict[str, Any]:
    manifest_path = environment_manifest_path.resolve(strict=True)
    manifest = load_json_object(manifest_path, "B200 environment manifest")
    require_equal("environment manifest schema", manifest.get("schema"), EXPECTED_ENVIRONMENT_SCHEMA)
    observed, locations = observed_runtime()
    for name, expected in EXPECTED_RUNTIME.items():
        require_equal(f"known B200 runtime {name}", observed[name], expected)

    versions = manifest.get("versions")
    if not isinstance(versions, dict):
        raise RuntimeError("B200 environment manifest has no versions object")
    require_equal("environment manifest av", versions.get("av"), observed["av"])
    require_equal("environment manifest deepspeed", versions.get("deepspeed"), observed["deepspeed"])
    torch_payload = manifest.get("torch")
    torchvision_payload = manifest.get("torchvision")
    if not isinstance(torch_payload, dict) or not isinstance(torchvision_payload, dict):
        raise RuntimeError("B200 environment manifest is missing torch/torchvision objects")
    require_equal("environment manifest torch", torch_payload.get("version"), observed["torch"])
    require_equal(
        "environment manifest torchvision", torchvision_payload.get("version"), observed["torchvision"]
    )
    require_equal("environment manifest torch CUDA", torch_payload.get("cuda"), locations["torch_cuda"])

    manifest_repo = manifest.get("repo_root")
    if not isinstance(manifest_repo, str):
        raise RuntimeError("B200 environment manifest has no repo_root")
    require_equal(
        "environment manifest repo_root",
        str(Path(manifest_repo).resolve()),
        str(project_root.resolve()),
    )
    manifest_python = manifest.get("python")
    if manifest_python is not None:
        if not isinstance(manifest_python, dict):
            raise RuntimeError("B200 environment manifest python field must be an object")
        require_equal("environment manifest Python", manifest_python.get("version"), observed["python"])
        require_equal(
            "environment manifest Python executable",
            str(Path(str(manifest_python.get("executable", ""))).resolve()),
            locations["python_executable"],
        )
    else:
        # Existing B200 manifests predate the explicit Python object; accept
        # them only with the fixed runtime version and exact executable check.
        manifest_executable = manifest.get("python_executable")
        if not isinstance(manifest_executable, str):
            raise RuntimeError("legacy B200 environment manifest has no python_executable")
        require_equal(
            "legacy environment manifest Python executable",
            str(Path(manifest_executable).resolve()),
            locations["python_executable"],
        )
    return {
        "environment_manifest": str(manifest_path),
        "environment_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "observed": observed,
        "locations": locations,
        "legacy_manifest_without_python_object": manifest_python is None,
    }


def validate(
    *,
    model_path: Path,
    model_integrity_manifest: Path,
    environment_manifest: Path,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "environment": validate_environment_identity(environment_manifest, project_root),
        "model": validate_model_identity(model_path, model_integrity_manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-integrity-manifest", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        model_path=args.model_path,
        model_integrity_manifest=args.model_integrity_manifest,
        environment_manifest=args.environment_manifest,
        project_root=args.project_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "[validate-b200-runtime] PASSED "
        f"model_files={result['model']['top_level_file_count']} "
        f"model_shards={len(result['model']['indexed_safetensors_shards'])} "
        f"python={result['environment']['observed']['python']} "
        f"torch={result['environment']['observed']['torch']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
