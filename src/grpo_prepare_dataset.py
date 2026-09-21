#!/usr/bin/env python3
"""Create a reproducible, path-verified JSON subset for Video-KTR GRPO.

The upstream 16k JSON contains paths for a media subset that is not mounted
locally.  Training must never silently substitute a different image/video for
a missing record, so launchers call this utility before creating a trainer.
This preflight verifies only a regular, non-empty file; actual video decoding
remains a trainer-time check and is intentionally reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_problem_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid problem id {item!r}; expected an integer"
            ) from exc
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("--problem-ids must not contain duplicates")
    return result


def resolve_media_path(video_root: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    raw = raw_path.strip()
    candidate = Path(raw)
    # Check the original path before stripping a cosmetic leading "./".
    # ``str.lstrip('./')`` would turn "../x" into "x", silently remapping an
    # unsafe record under DATA_ROOT instead of rejecting it.
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        return None
    relative = Path(raw[2:] if raw.startswith("./") else raw)
    resolved_root = video_root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-type", choices=("video", "image", "all"), default="video"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="maximum selected records; zero means no limit",
    )
    parser.add_argument(
        "--problem-ids",
        type=parse_problem_ids,
        default=[],
        help="comma-separated, ordered exact IDs for a deterministic smoke subset",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="print a flushed progress line after this many inspected candidates; zero disables it",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if not args.source.is_file():
        parser.error(f"source JSON is unavailable: {args.source}")
    if not args.video_root.is_dir():
        parser.error(f"video root is unavailable: {args.video_root}")

    loaded = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("source JSON must be a list of records")
    records = [record for record in loaded if isinstance(record, dict)]
    if len(records) != len(loaded):
        raise ValueError("source JSON contains a non-object record")

    desired_ids = list(args.problem_ids)
    by_id = {record.get("problem_id"): record for record in records}
    if desired_ids:
        missing_ids = [problem_id for problem_id in desired_ids if problem_id not in by_id]
        if missing_ids:
            raise ValueError(f"requested problem IDs are absent: {missing_ids}")
        candidates = [by_id[problem_id] for problem_id in desired_ids]
    else:
        candidates = records

    selected: list[dict[str, Any]] = []
    skipped_type = 0
    skipped_media = 0
    for inspected, record in enumerate(candidates, start=1):
        data_type = record.get("data_type")
        if args.data_type != "all" and data_type != args.data_type:
            skipped_type += 1
        else:
            media_path = resolve_media_path(args.video_root, record.get("path"))
            if media_path is None or not media_path.is_file() or media_path.stat().st_size == 0:
                skipped_media += 1
            else:
                selected.append(record)
        if args.progress_every and inspected % args.progress_every == 0:
            print(
                "[prepare-grpo-dataset] "
                f"inspected={inspected}/{len(candidates)} selected={len(selected)} "
                f"skipped_missing_or_empty_media={skipped_media}",
                flush=True,
            )
        if args.limit and len(selected) >= args.limit:
            break

    if desired_ids and len(selected) != len(desired_ids):
        selected_ids = {record.get("problem_id") for record in selected}
        unavailable_ids = [problem_id for problem_id in desired_ids if problem_id not in selected_ids]
        raise RuntimeError(
            "requested smoke media are unavailable or have the wrong type: "
            f"{unavailable_ids}"
        )
    if not selected:
        raise RuntimeError("no path-verified records matched the requested selection")

    atomic_json_write(args.output, selected)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "video_root": str(args.video_root.resolve()),
        "output": str(args.output.resolve()),
        "data_type": args.data_type,
        "requested_problem_ids": desired_ids,
        "limit": args.limit,
        "source_records": len(records),
        "selected_records": len(selected),
        "skipped_wrong_type": skipped_type,
        "skipped_missing_or_empty_media": skipped_media,
        "selected_problem_ids": [record.get("problem_id") for record in selected],
    }
    atomic_json_write(manifest_path, manifest)
    print(
        "[prepare-grpo-dataset] "
        f"selected={len(selected)} source_records={len(records)} "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
