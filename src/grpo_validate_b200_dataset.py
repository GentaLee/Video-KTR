#!/usr/bin/env python3
"""Validate the fixed Holmes-16k data contract for the B200 full profile.

This gate separates an honest strict Holmes-16k run from the currently
available reduced-media subset.  It intentionally rejects a changed or
partially filtered manifest unless the operator makes the applicable downgrade
explicit on the command line.  The launcher invokes it once after path
verification (before the potentially long CPU decode) and once after decoder
verification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_COUNTS = {"image": 8765, "video": 8151}
EXPECTED_SOURCE_RECORDS = sum(EXPECTED_SOURCE_COUNTS.values())
EXPECTED_REDUCED_COUNTS = {"image": 8765, "video": 6600}
EXPECTED_REDUCED_RECORDS = sum(EXPECTED_REDUCED_COUNTS.values())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"{label} is unavailable: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(record, dict) for record in data):
        raise RuntimeError(f"{label} must be a JSON list of object records: {path}")
    return data


def load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is unavailable: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return data


def modality_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"image": 0, "video": 0, "unsupported": 0}
    for record in records:
        value = record.get("data_type")
        counts[value if value in {"image", "video"} else "unsupported"] += 1
    return counts


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def path_contract(
    source_path: Path,
    path_dataset_path: Path,
    path_manifest_path: Path,
    *,
    allow_reduced: bool,
) -> dict[str, Any]:
    source = load_records(source_path, "Holmes source JSON")
    selected = load_records(path_dataset_path, "path-verified JSON")
    manifest = load_object(path_manifest_path, "path verification manifest")
    source_counts = modality_counts(source)
    selected_counts = modality_counts(selected)
    require(
        len(source) == EXPECTED_SOURCE_RECORDS,
        f"Holmes source count changed: expected {EXPECTED_SOURCE_RECORDS}, got {len(source)}",
    )
    require(
        source_counts == {**EXPECTED_SOURCE_COUNTS, "unsupported": 0},
        f"Holmes source modality contract changed: expected {EXPECTED_SOURCE_COUNTS}, got {source_counts}",
    )
    require(manifest.get("data_type") == "all", "path verification must use --data-type all")
    require(int(manifest.get("source_records", -1)) == len(source), "path manifest source_records mismatch")
    require(int(manifest.get("selected_records", -1)) == len(selected), "path manifest selected_records mismatch")
    require(manifest.get("source_sha256") == sha256_file(source_path), "path manifest source SHA mismatch")
    require(selected_counts["unsupported"] == 0, "path-verified dataset contains unsupported data_type")
    missing_by_modality = {
        modality: source_counts[modality] - selected_counts[modality]
        for modality in ("image", "video")
    }
    if any(value < 0 for value in missing_by_modality.values()):
        raise RuntimeError("path-verified modality count exceeds source count")

    if len(selected) == EXPECTED_SOURCE_RECORDS and selected_counts == {**EXPECTED_SOURCE_COUNTS, "unsupported": 0}:
        reproduction_class = "strict-holmes-16k"
        strict = True
    elif (
        allow_reduced
        and len(selected) == EXPECTED_REDUCED_RECORDS
        and selected_counts == {**EXPECTED_REDUCED_COUNTS, "unsupported": 0}
        and missing_by_modality == {"image": 0, "video": 1551}
    ):
        reproduction_class = "reduced-media-subset"
        strict = False
    else:
        detail = {
            "expected_source_records": EXPECTED_SOURCE_RECORDS,
            "available_records": len(selected),
            "missing_records": len(source) - len(selected),
            "missing_image": missing_by_modality["image"],
            "missing_video": missing_by_modality["video"],
            "result": "refuse-not-full-holmes",
        }
        if len(selected) == EXPECTED_REDUCED_RECORDS and not allow_reduced:
            detail["required_override"] = "B200_ALLOW_REDUCED_HOLMES=1"
        raise RuntimeError("B200 Holmes gate: " + json.dumps(detail, sort_keys=True))

    return {
        "source_records": len(source),
        "source_modality_counts": source_counts,
        "path_verified_records": len(selected),
        "path_verified_modality_counts": selected_counts,
        "missing_records": len(source) - len(selected),
        "missing_by_modality": missing_by_modality,
        "path_verified_sha256": sha256_file(path_dataset_path),
        "strict_holmes_16k": strict,
        "reproduction_class": reproduction_class,
    }


def decoder_contract(
    path_dataset_path: Path,
    decoder_dataset_path: Path,
    decoder_manifest_path: Path,
    *,
    nframes: int,
    max_pixels: int,
    allow_decoder_filtered: bool,
) -> dict[str, Any]:
    source = load_records(path_dataset_path, "path-verified JSON")
    verified = load_records(decoder_dataset_path, "decoder-verified JSON")
    manifest = load_object(decoder_manifest_path, "decoder verification manifest")
    source_counts = modality_counts(source)
    verified_counts = modality_counts(verified)
    semantics = manifest.get("semantics")
    decoder = manifest.get("decoder")
    require(manifest.get("status") == "passed", "decoder manifest is not passed")
    require(manifest.get("verification_complete") is True, "decoder verification was incomplete")
    require(Path(str(manifest.get("source", ""))).resolve() == path_dataset_path.resolve(), "decoder source path mismatch")
    require(manifest.get("source_sha256") == sha256_file(path_dataset_path), "decoder source SHA mismatch")
    require(Path(str(manifest.get("output", ""))).resolve() == decoder_dataset_path.resolve(), "decoder output path mismatch")
    require(manifest.get("output_sha256") == sha256_file(decoder_dataset_path), "decoder output SHA mismatch")
    require(isinstance(decoder, dict) and decoder.get("reader_backend") == "torchvision", "decoder backend mismatch")
    require(decoder.get("cuda_visible_devices") == "", "decoder was not CUDA-hidden")
    require(isinstance(semantics, dict), "decoder semantics are missing")
    require(int(semantics.get("video_nframes", -1)) == nframes, "decoder nframes mismatch")
    require(int(semantics.get("video_max_pixels", -1)) == max_pixels, "decoder max_pixels mismatch")
    require(semantics.get("image_max_pixels") is None, "decoder image cap diverges from current trainer")
    require(
        semantics.get("image_default_matches_current_direct_trainer") is True,
        "decoder image preprocessing does not match the direct trainer",
    )
    require(verified_counts["unsupported"] == 0, "decoder-verified dataset contains unsupported data_type")
    rejected = int(manifest.get("rejected_records", -1))
    require(rejected >= 0, "decoder rejected_records is invalid")
    require(int(manifest.get("source_records", -1)) == len(source), "decoder source_records mismatch")
    require(int(manifest.get("verified_records", -1)) == len(verified), "decoder verified_records mismatch")
    require(
        len(verified) + rejected == len(source),
        "decoder verified/rejected counts do not balance against source",
    )
    filtered = len(verified) != len(source) or rejected != 0
    if filtered and not allow_decoder_filtered:
        raise RuntimeError(
            "decoder preflight filtered media; refuse a silent dataset change. "
            "Inspect its rejection manifest or explicitly set B200_ALLOW_DECODER_FILTERED=1."
        )
    return {
        "decoder_verified_records": len(verified),
        "decoder_verified_modality_counts": verified_counts,
        "decoder_rejected_records": rejected,
        "decoder_timed_out_records": int(manifest.get("timed_out_records", -1)),
        "decoder_verified_sha256": sha256_file(decoder_dataset_path),
        "decoder_filtered": filtered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--path-dataset", type=Path, required=True)
    parser.add_argument("--path-manifest", type=Path, required=True)
    parser.add_argument("--decoder-dataset", type=Path)
    parser.add_argument("--decoder-manifest", type=Path)
    parser.add_argument("--nframes", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--allow-reduced-holmes", action="store_true")
    parser.add_argument("--allow-decoder-filtered", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.nframes < 2 or args.nframes % 2:
        parser.error("--nframes must be an even integer of at least 2")
    if args.max_pixels < 1:
        parser.error("--max-pixels must be positive")
    if (args.decoder_dataset is None) != (args.decoder_manifest is None):
        parser.error("--decoder-dataset and --decoder-manifest must be supplied together")

    result: dict[str, Any] = {
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": str(args.source.resolve()),
        "path_dataset": str(args.path_dataset.resolve()),
        "path_manifest": str(args.path_manifest.resolve()),
        "path_contract": path_contract(
            args.source,
            args.path_dataset,
            args.path_manifest,
            allow_reduced=args.allow_reduced_holmes,
        ),
    }
    result["reproduction_class"] = result["path_contract"]["reproduction_class"]
    result["strict_holmes_16k"] = result["path_contract"]["strict_holmes_16k"]
    if args.decoder_dataset is not None and args.decoder_manifest is not None:
        result["decoder_dataset"] = str(args.decoder_dataset.resolve())
        result["decoder_manifest"] = str(args.decoder_manifest.resolve())
        result["decoder_contract"] = decoder_contract(
            args.path_dataset,
            args.decoder_dataset,
            args.decoder_manifest,
            nframes=args.nframes,
            max_pixels=args.max_pixels,
            allow_decoder_filtered=args.allow_decoder_filtered,
        )
        if result["decoder_contract"]["decoder_filtered"]:
            # Decoder filtering is a second data alteration even if all paths
            # had existed.  Never leave a filtered output labeled strict.
            result["strict_holmes_16k"] = False
            result["reproduction_class"] = (
                f"{result['path_contract']['reproduction_class']}+decoder-filtered"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path_gate = result["path_contract"]
    print(
        "[validate-b200-dataset] "
        f"PASSED class={result['reproduction_class']} "
        f"available={path_gate['path_verified_records']}/{path_gate['source_records']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
