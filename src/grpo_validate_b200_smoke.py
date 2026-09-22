#!/usr/bin/env python3
"""Validate that a passed B200 smoke still covers the current GRPO sources.

The full B200 launcher deliberately treats a smoke result as an input with a
contract, rather than merely checking that an artifact directory exists.  A
new launcher or documentation commit may legitimately follow the smoke, so a
literal Git-HEAD equality check would be too strict.  Instead this utility
requires the smoke commit to be an ancestor of the current clean checkout and
compares the hashes of the code that determines model loading, attention,
media handling, and E/V/T selection.

It is intentionally offline and does not import torch or load a model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


REQUIRED_PROFILE = {
    "profile": "8xB200-upstream-aligned",
    "holmes_requested": "Holmes-16k",
    "max_prompt_length": "16384",
    "max_completion_length": "768",
    "num_generations": "8",
    "max_pixels": "401408",
    "nframes": "8",
    "attn_implementation": "flash_attention_2",
    "qwen_fa2_rotary_dtype_compat": "true",
    "selection_scope": "per_completion",
    "selection_ratio": "0.2",
    "delta": "absolute",
    "temporal_permutations": "1",
    "temporal_include_reverse": "false",
    "image_video_token_ids": "distinct",
}

CRITICAL_SOURCES = (
    "src/r1-v/src/open_r1/grpo.py",
    "src/r1-v/src/open_r1/trainer/grpo_trainer.py",
    "src/r1-v/src/open_r1/trainer/qwen25vl_fa2_compat.py",
    "src/r1-v/src/open_r1/trainer/video_ktr_grpo_trainer.py",
    "src/r1-v/src/open_r1/trainer/ktr_token_utils.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_key_value(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def command(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}"
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def validate(smoke_root: Path, project_root: Path) -> dict[str, Any]:
    smoke_root = smoke_root.resolve()
    project_root = project_root.resolve()
    profile_path = smoke_root / "b200_profile.txt"
    terminal_path = smoke_root / "terminal.log"
    provenance_path = smoke_root / "source_provenance.txt"
    for required in (profile_path, terminal_path, provenance_path):
        if not required.is_file():
            raise RuntimeError(f"missing required B200 smoke evidence: {required}")

    profile = parse_key_value(profile_path)
    profile_mismatches = {
        key: {"expected": expected, "observed": profile.get(key)}
        for key, expected in REQUIRED_PROFILE.items()
        if profile.get(key) != expected
    }
    if profile_mismatches:
        raise RuntimeError("B200 smoke profile mismatch: " + json.dumps(profile_mismatches, sort_keys=True))

    terminal = terminal_path.read_text(encoding="utf-8", errors="replace")
    if "B200 smoke wrapper completed successfully" not in terminal:
        raise RuntimeError("smoke terminal log has no successful wrapper completion marker")

    variants: dict[str, dict[str, Any]] = {}
    for variant in ("baseline", "ktr"):
        completion_path = smoke_root / variant / "training" / "training_complete.json"
        training_log_path = smoke_root / variant / "training.log"
        if not completion_path.is_file():
            raise RuntimeError(f"missing {variant} smoke completion record: {completion_path}")
        if not training_log_path.is_file():
            raise RuntimeError(f"missing {variant} smoke training log: {training_log_path}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not isinstance(completion, dict) or int(completion.get("global_step", 0)) < 1:
            raise RuntimeError(f"{variant} smoke did not complete an optimizer step")
        if variant == "ktr" and completion.get("video_ktr") is not True:
            raise RuntimeError("KTR smoke completion is not marked video_ktr=true")
        if variant == "baseline" and completion.get("video_ktr") is not False:
            raise RuntimeError("baseline smoke completion is not marked video_ktr=false")
        training_log = training_log_path.read_text(encoding="utf-8", errors="replace")
        if (
            "requested_attention=flash_attention_2" not in training_log
            or "resolved_attention=flash_attention_2" not in training_log
        ):
            raise RuntimeError(
                f"{variant} smoke log does not prove FlashAttention-2 resolved at model load"
            )
        variants[variant] = {
            "global_step": int(completion["global_step"]),
            "video_ktr": bool(completion["video_ktr"]),
            "resolved_attention": "flash_attention_2",
        }

    provenance = parse_key_value(provenance_path)
    smoke_commit = provenance.get("git_head")
    if not smoke_commit:
        raise RuntimeError("smoke source provenance has no git_head")
    if command(project_root, "status", "--porcelain"):
        raise RuntimeError("current checkout is dirty; commit/deploy it before a full B200 run")
    current_commit = command(project_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(project_root), "merge-base", "--is-ancestor", smoke_commit, current_commit],
        check=False,
    )
    if ancestor.returncode:
        raise RuntimeError(
            "the smoke commit is not an ancestor of the current checkout; "
            "rerun the B200 smoke on this source lineage"
        )

    source_hashes: dict[str, dict[str, str]] = {}
    for relative in CRITICAL_SOURCES:
        recorded = provenance.get(f"source_sha256[{relative}]")
        current_path = project_root / relative
        if recorded is None:
            raise RuntimeError(f"smoke provenance does not fingerprint critical source: {relative}")
        if not current_path.is_file():
            raise RuntimeError(f"current critical source is missing: {current_path}")
        current = sha256_file(current_path)
        if current != recorded:
            raise RuntimeError(
                f"critical source changed since smoke: {relative}; rerun B200 smoke before full"
            )
        source_hashes[relative] = {"smoke": recorded, "current": current}

    return {
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "smoke_root": str(smoke_root),
        "smoke_git_head": smoke_commit,
        "current_git_head": current_commit,
        "profile": {key: profile[key] for key in REQUIRED_PROFILE},
        "variants": variants,
        "critical_source_hashes": source_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = validate(args.smoke_root, args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "[validate-b200-smoke] "
        f"PASSED smoke={args.smoke_root} smoke_commit={result['smoke_git_head']} "
        f"current_commit={result['current_git_head']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
