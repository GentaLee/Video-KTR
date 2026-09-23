#!/usr/bin/env python3
"""CPU-only exact-cache parity probe against the real Qwen decoders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-name", action="append", default=[])
    parser.add_argument("--nframes", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=401408)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from qwen_vl_utils.vision_process import fetch_image, fetch_video
    from grpo_media_cache import load_cached_media, runtime_fingerprint
    from grpo_prepare_dataset import atomic_json_write

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records = json.loads(args.source.read_text())
    selected = [next(record for record in records if record.get("data_type") == "image")]
    names = args.video_name or [
        Path(record["path"]).name for record in records if record.get("data_type") == "video"
    ][:2]
    for name in names:
        selected.append(next(record for record in records
                             if record.get("data_type") == "video"
                             and Path(record["path"]).name == name))
    report = {"status": "running", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "torch_threads": torch.get_num_threads(), "runtime": runtime_fingerprint(),
              "samples": []}
    begin = time.perf_counter()
    for _ in range(50):
        runtime_fingerprint()
    report["fingerprint_mean_seconds"] = (time.perf_counter() - begin) / 50
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(args.output, report)
    try:
        for record in selected:
            media_type = record["data_type"]
            path = (args.media_root / record["path"]).resolve(strict=True)
            if not path.is_relative_to(args.media_root.resolve()):
                raise RuntimeError(f"media path escapes root: {path}")
            element = {"type": media_type, media_type: str(path)}
            if media_type == "video":
                element.update(nframes=args.nframes, max_pixels=args.max_pixels)
            print(f"[media-cache-parity] raw decode starting: {path.name}", flush=True)
            begin = time.perf_counter()
            raw, raw_fps = ((fetch_image(element), None) if media_type == "image"
                            else fetch_video(element, return_video_sample_fps=True))
            sample = {"path": str(path), "media_type": media_type,
                      "raw_decode_seconds": time.perf_counter() - begin}
            for label, create in (("create_or_hit", True), ("hit", False)):
                begin = time.perf_counter()
                cached, fps = load_cached_media(element, cache_root=args.cache_root, allow_create=create)
                sample[f"{label}_seconds"] = time.perf_counter() - begin
                equal = (raw.size == cached.size and raw.mode == cached.mode
                         and raw.tobytes() == cached.tobytes()) if media_type == "image" else torch.equal(raw, cached)
                assert equal, f"pixel/tensor mismatch ({label}): {path}"
                assert fps == raw_fps, f"sampled FPS mismatch ({label}): {path}"
                sample[f"{label}_exact_equal"] = equal
            sample["fps"] = raw_fps
            sample["shape"] = list(raw.size if media_type == "image" else raw.shape)
            report["samples"].append(sample)
            atomic_json_write(args.output, report)
            print(f"[media-cache-parity] PASSED {json.dumps(sample, sort_keys=True)}", flush=True)
        report["status"] = "passed"
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_json_write(args.output, report)


if __name__ == "__main__":
    raise SystemExit(main())
