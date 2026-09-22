#!/usr/bin/env python3
"""Atomically materialize a verified local copy of the trusted model manifest.

``manifests/video-r1-*.json`` is the path-free trust anchor committed with the
reproduction source. This helper is deliberately *not* an alternate source of
truth: it first proves that every top-level regular model file matches that
trusted manifest, then writes the same schema to a durable local destination.
It is useful after checkpoint transfer or environment provisioning, and has no
torch, model-loading, GPU, or network dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any


MODEL_MANIFEST_SCHEMA = "video-ktr/model-integrity-manifest/v1"
EXPECTED_SOURCE_KEYS = {"repository", "revision", "tree_url"}
EXPECTED_FILE_KEYS = {"sha256", "size_bytes", "role"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_relative_filename(raw: str) -> str:
    """Allow exactly one portable file name, never a path or traversal."""

    if "\\" in raw:
        raise RuntimeError(f"model manifest uses a non-portable path separator: {raw!r}")
    relative = PurePosixPath(raw)
    if (
        not raw
        or relative.is_absolute()
        or len(relative.parts) != 1
        or relative.parts[0] in {".", ".."}
    ):
        raise RuntimeError(
            "model manifest entries must be top-level model-relative filenames; "
            f"got {raw!r}"
        )
    return relative.name


def sha256_file_stable(path: Path) -> tuple[str, int]:
    """Hash one regular file and reject an in-flight replacement or rewrite."""

    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"model entry is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat(follow_symlinks=False)
    fingerprint_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    fingerprint_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if fingerprint_before != fingerprint_after:
        raise RuntimeError(f"model file changed while hashing; retry after transfer is idle: {path}")
    return digest.hexdigest(), before.st_size


def top_level_regular_files(model_root: Path) -> list[Path]:
    if not model_root.is_dir():
        raise RuntimeError(f"model directory is unavailable: {model_root}")
    files: list[Path] = []
    for entry in sorted(model_root.iterdir(), key=lambda item: item.name):
        if entry.is_symlink():
            raise RuntimeError(f"refusing symlinked model entry: {entry}")
        if entry.is_file():
            mode = entry.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"model entry is not a regular file: {entry}")
            files.append(entry)
        elif not entry.is_dir():
            raise RuntimeError(f"refusing non-regular, non-directory model entry: {entry}")
    if not files:
        raise RuntimeError(f"model directory has no top-level regular files: {model_root}")
    return files


def load_model_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"trusted model manifest is unavailable: {path}")
    if path.is_symlink():
        raise RuntimeError(f"refusing symlinked trusted model manifest: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"trusted model manifest is invalid JSON: {path}: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"schema", "source", "files"}:
        raise RuntimeError("trusted model manifest must contain exactly schema, source, and files")
    if payload["schema"] != MODEL_MANIFEST_SCHEMA:
        raise RuntimeError(
            f"trusted model manifest schema mismatch: {payload['schema']!r} != {MODEL_MANIFEST_SCHEMA!r}"
        )
    source = payload["source"]
    if not isinstance(source, dict) or set(source) != EXPECTED_SOURCE_KEYS:
        raise RuntimeError("trusted model manifest source must contain exactly repository, revision, tree_url")
    if not all(isinstance(source[key], str) and source[key] for key in EXPECTED_SOURCE_KEYS):
        raise RuntimeError("trusted model manifest source values must be non-empty strings")
    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise RuntimeError("trusted model manifest files must be a non-empty object")
    normalized_files: dict[str, dict[str, Any]] = {}
    for raw_name, record in files.items():
        if not isinstance(raw_name, str):
            raise RuntimeError("trusted model manifest file names must be strings")
        name = validate_relative_filename(raw_name)
        if name in normalized_files:
            raise RuntimeError(f"trusted model manifest has duplicate normalized file name: {name}")
        if not isinstance(record, dict) or set(record) != EXPECTED_FILE_KEYS:
            raise RuntimeError(
                f"trusted model manifest entry {name!r} must contain exactly sha256, size_bytes, role"
            )
        digest = record["sha256"]
        size = record["size_bytes"]
        role = record["role"]
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise RuntimeError(f"trusted model manifest entry {name!r} has invalid lowercase SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeError(f"trusted model manifest entry {name!r} has invalid size_bytes")
        if not isinstance(role, str) or not role:
            raise RuntimeError(f"trusted model manifest entry {name!r} has invalid role")
        normalized_files[name] = {"sha256": digest, "size_bytes": size, "role": role}
    return {"schema": payload["schema"], "source": dict(source), "files": normalized_files}


def ensure_regular_model_file(model_root: Path, filename: str) -> Path:
    candidate = model_root / filename
    if candidate.is_symlink():
        raise RuntimeError(f"refusing symlinked model file named by manifest: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"model manifest file is missing: {candidate}") from error
    if not is_relative_to(resolved, model_root):
        raise RuntimeError(f"model manifest entry escapes model root: {filename!r}")
    if not stat.S_ISREG(candidate.stat(follow_symlinks=False).st_mode):
        raise RuntimeError(f"model manifest entry is not a regular file: {candidate}")
    return candidate


def validate_model_files(model_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    model_root = model_path.resolve(strict=True)
    if not model_root.is_dir():
        raise RuntimeError(f"model path is not a directory: {model_root}")
    expected_files = manifest["files"]
    actual_files = {entry.name for entry in top_level_regular_files(model_root)}
    expected_names = set(expected_files)
    if actual_files != expected_names:
        raise RuntimeError(
            "trusted model manifest does not cover exactly the current top-level regular files: "
            + json.dumps(
                {
                    "unlisted_model_files": sorted(actual_files - expected_names),
                    "missing_model_files": sorted(expected_names - actual_files),
                },
                sort_keys=True,
            )
        )
    records: list[dict[str, Any]] = []
    for index, filename in enumerate(sorted(expected_files), start=1):
        expected = expected_files[filename]
        candidate = ensure_regular_model_file(model_root, filename)
        print(
            f"[model-sha256] START {index}/{len(expected_files)} {filename} "
            f"expected_bytes={expected['size_bytes']}",
            flush=True,
        )
        observed_digest, observed_size = sha256_file_stable(candidate)
        print(
            f"[model-sha256] PASS {index}/{len(expected_files)} {filename} bytes={observed_size}",
            flush=True,
        )
        if observed_size != expected["size_bytes"]:
            raise RuntimeError(
                f"model size mismatch for {filename}: observed={observed_size} "
                f"expected={expected['size_bytes']}"
            )
        if observed_digest != expected["sha256"]:
            raise RuntimeError(
                f"model SHA-256 mismatch for {filename}: observed={observed_digest} "
                f"expected={expected['sha256']}"
            )
        records.append(
            {
                "name": filename,
                "role": expected["role"],
                "bytes": observed_size,
                "sha256": observed_digest,
            }
        )
    return records


def write_manifest(model_path: Path, reference_manifest_path: Path, output_path: Path) -> dict[str, object]:
    model_root = model_path.resolve(strict=True)
    if not model_root.is_dir():
        raise RuntimeError(f"model path is not a directory: {model_root}")
    if reference_manifest_path.is_symlink():
        raise RuntimeError(f"refusing symlinked trusted model manifest: {reference_manifest_path}")
    reference = reference_manifest_path.resolve(strict=True)
    output = output_path.expanduser().resolve(strict=False)
    if is_relative_to(output, model_root):
        raise RuntimeError(
            "model checksum manifest must live outside the model directory so it cannot "
            f"include itself: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.parent.is_dir():
        raise RuntimeError(f"manifest parent is unavailable: {output.parent}")
    manifest = load_model_manifest(reference)
    records = validate_model_files(model_root, manifest)
    contents = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "model_root": str(model_root),
        "reference_manifest": str(reference),
        "manifest": str(output),
        "file_count": len(records),
        "manifest_sha256": hashlib.sha256(contents.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_manifest(args.model_path, args.reference_manifest, args.output)
    print(
        "[model-sha256] PASSED "
        f"files={result['file_count']} manifest={result['manifest']} "
        f"manifest_sha256={result['manifest_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
