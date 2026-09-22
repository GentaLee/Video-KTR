#!/usr/bin/env python3
"""Build a CPU-decoder-verified mixed-media manifest for Video-KTR GRPO.

The input must already be a path-verified JSON list.  This second preflight
checks the operation that precedes model processing: local image decoding and
resize through :mod:`qwen_vl_utils`, or local video decoding, frame sampling,
and resize through the same package.  It deliberately does *not* tokenize a
prompt, load a model, or use CUDA.  A passing record is therefore evidence
that its media preprocessing completed on CPU under the selected qwen-vl-utils
reader; it is not evidence that a subsequent model/optimizer step will fit or
complete.

Each potentially blocking decoder call runs in a short-lived spawned child.
The parent enforces a per-record wall-clock timeout, kills a child which
overruns it, records a rejection, and replaces the child before proceeding.
Accepted records retain their original JSON object and source order.  Raw
relative paths are retained in the output so the GRPO trainer resolves them
against the same data root during training.

The current direct Video-KTR trainer attaches ``nframes`` and ``max_pixels``
only to video content.  By default this tool mirrors that exact behavior:
images use qwen-vl-utils' image defaults.  ``--image-max-pixels`` is an
explicit opt-in for a future trainer profile which also sets an image cap; it
must not be enabled unless the training path uses the same setting.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
import time
from typing import Any, Literal

from grpo_prepare_dataset import atomic_json_write, resolve_media_path, sha256_file


MediaType = Literal["image", "video"]
_SUPPORTED_MEDIA_TYPES: tuple[MediaType, ...] = ("image", "video")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _even_nframes(value: str) -> int:
    """Match qwen-vl-utils' frame-factor contract instead of rounding input."""

    parsed = _positive_int(value)
    if parsed < 2 or parsed % 2:
        raise argparse.ArgumentTypeError("must be an even integer of at least 2")
    return parsed


def _media_type(record: dict[str, Any]) -> MediaType | None:
    candidate = record.get("data_type")
    return candidate if candidate in _SUPPORTED_MEDIA_TYPES else None


def _new_modality_counts() -> dict[str, int]:
    return {"image": 0, "video": 0, "unsupported": 0}


def _count_records(records: list[dict[str, Any]], indices: set[int] | None = None) -> dict[str, int]:
    """Return modality counts with unsupported/malformed types kept visible."""

    counts = _new_modality_counts()
    selected_indices = range(len(records)) if indices is None else sorted(indices)
    for index in selected_indices:
        media_type = _media_type(records[index])
        counts[media_type if media_type is not None else "unsupported"] += 1
    return counts


def _safe_rejection(
    source_index: int, record: dict[str, Any], reason: str, detail: str
) -> dict[str, Any]:
    """Make a bounded, serializable rejection without resolving unsafe input."""

    return {
        "source_index": source_index,
        "problem_id": record.get("problem_id"),
        "data_type": record.get("data_type"),
        "path": record.get("path"),
        "reason": reason,
        "detail": detail[:1000],
    }


def _safe_regular_nonempty_path(root: Path, raw_path: object) -> Path | None:
    """Resolve a record path without turning filesystem errors into a run crash."""

    try:
        path = resolve_media_path(root, raw_path)
        if path is None or not path.is_file() or path.stat().st_size == 0:
            return None
    except OSError:
        return None
    return path


def _build_media_element(
    media_type: MediaType,
    resolved_path: Path,
    *,
    nframes: int,
    max_pixels: int,
    image_max_pixels: int | None,
) -> dict[str, Any]:
    """Reconstruct the media content item received by qwen-vl-utils.

    ``VideoKTRGRPOTrainer._prepare_prompt`` injects an absolute, root-checked
    path into the first prompt content item.  It adds ``nframes`` and
    ``max_pixels`` for video only.  Keeping that distinction makes this
    preflight an honest decoder check for the current training implementation
    rather than a stronger-but-different image preprocessing profile.
    """

    element: dict[str, Any] = {"type": media_type, media_type: str(resolved_path)}
    if media_type == "video":
        element["nframes"] = nframes
        element["max_pixels"] = max_pixels
    elif image_max_pixels is not None:
        element["max_pixels"] = image_max_pixels
    return element


def _decode_one(
    media_type: MediaType, media_element: dict[str, Any], required_frames: int
) -> dict[str, Any]:
    """Run the qwen-vl-utils per-media preprocessing reached by training.

    Imports occur inside the spawned worker so FFmpeg/Pillow/torchvision
    failures remain killable by the supervising process.  We pass only an
    already-resolved local path, never a URL or an arbitrary ``file://`` value.
    """

    from qwen_vl_utils.vision_process import fetch_image, fetch_video

    if media_type == "image":
        image = fetch_image(media_element)
        size = getattr(image, "size", None)
        if not isinstance(size, tuple) or len(size) != 2:
            raise RuntimeError(f"unexpected decoded image type: {type(image).__name__}")
        width, height = (int(size[0]), int(size[1]))
        if width < 1 or height < 1:
            raise RuntimeError(f"decoded image has non-positive dimensions: {size!r}")
        return {
            "media_type": "image",
            "shape": [height, width],
            "mode": str(getattr(image, "mode", "unknown")),
        }

    video = fetch_video(media_element)
    if not hasattr(video, "ndim") or not hasattr(video, "shape"):
        raise RuntimeError(f"unexpected decoded video type: {type(video).__name__}")
    if int(video.ndim) != 4 or int(video.shape[1]) != 3:
        raise RuntimeError(f"unexpected decoded video shape: {tuple(video.shape)}")
    frame_count = int(video.shape[0])
    if frame_count < required_frames:
        raise RuntimeError(
            f"decoded only {frame_count} frame(s), need at least {required_frames}"
        )
    return {
        "media_type": "video",
        "shape": [int(dimension) for dimension in video.shape],
        "frame_count": frame_count,
    }


def _decoder_worker(connection: Connection) -> None:
    """Decode one assigned record at a time in a CUDA-hidden child process."""

    # Set before importing qwen-vl-utils/torch.  The verifier has no torch.cuda
    # calls and deliberately does not inherit visible training GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    while True:
        job = connection.recv()
        if job is None:
            return
        index, media_type, media_element, required_frames = job
        connection.send(("started", index))
        try:
            metadata = _decode_one(media_type, media_element, required_frames)
        except BaseException as exc:
            connection.send(("result", index, False, type(exc).__name__, str(exc)[:1000]))
        else:
            connection.send(("result", index, True, metadata, None))


class DecoderWorkerCrash(RuntimeError):
    """A child death is a verifier failure, not a media rejection to ignore."""

    def __init__(
        self,
        *,
        worker_id: int,
        rejection: dict[str, Any],
        accepted: set[int],
        rejections: list[dict[str, Any]],
        timeout_count: int,
        completed: int,
    ) -> None:
        super().__init__(
            f"decoder worker {worker_id} exited unexpectedly while processing "
            f"source_index={rejection['source_index']}; inspect retained diagnostics"
        )
        self.rejection = rejection
        self.accepted = set(accepted)
        self.rejections = list(rejections)
        self.timeout_count = timeout_count
        self.completed = completed


class MediaDecodeSupervisor:
    """Bounded, killable CPU workers with one active media item per worker."""

    def __init__(self, worker_count: int, timeout_seconds: float) -> None:
        self._context = mp.get_context("spawn")
        self._timeout_seconds = timeout_seconds
        self._slots = [self._spawn(worker_id) for worker_id in range(worker_count)]

    def _spawn(self, worker_id: int) -> dict[str, Any]:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_decoder_worker,
            args=(child,),
            name=f"media-decode-worker-{worker_id}",
            daemon=True,
        )
        process.start()
        child.close()
        return {
            "worker_id": worker_id,
            "connection": parent,
            "process": process,
            "active": None,
            "assigned_at": 0.0,
        }

    @staticmethod
    def _close_slot(slot: dict[str, Any], *, terminate: bool) -> None:
        """Stop a worker, escalating to SIGKILL rather than leaking decoders."""

        process = slot["process"]
        connection = slot["connection"]
        try:
            if not terminate and process.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            if process.is_alive():
                raise RuntimeError(
                    f"decoder worker {slot['worker_id']} survived SIGKILL; "
                    "stop and inspect the host before continuing"
                )
        finally:
            connection.close()

    def _replace(self, position: int) -> dict[str, Any]:
        old = self._slots[position]
        worker_id = int(old["worker_id"])
        self._close_slot(old, terminate=True)
        replacement = self._spawn(worker_id)
        self._slots[position] = replacement
        return replacement

    def close(self) -> None:
        for slot in self._slots:
            self._close_slot(slot, terminate=False)

    def run(
        self,
        records: list[dict[str, Any]],
        jobs: dict[int, tuple[MediaType, dict[str, Any]]],
        *,
        required_frames: int,
        progress_every: int,
        initial_rejection_count: int,
        status_seconds: float,
    ) -> tuple[set[int], list[dict[str, Any]], int]:
        pending = iter(sorted(jobs))
        accepted: set[int] = set()
        rejected: list[dict[str, Any]] = []
        timeout_count = 0
        completed = len(records) - len(jobs)
        started_at = time.monotonic()
        last_status_at = started_at
        last_reported_completed = -1

        def fail_worker(slot: dict[str, Any], index: int, detail: str) -> None:
            rejection = _safe_rejection(index, records[index], "worker_crash", detail)
            rejected.append(rejection)
            raise DecoderWorkerCrash(
                worker_id=int(slot["worker_id"]),
                rejection=rejection,
                accepted=accepted,
                rejections=rejected,
                timeout_count=timeout_count,
                completed=completed,
            )

        def report(*, periodic: bool = False) -> None:
            nonlocal last_status_at, last_reported_completed
            now = time.monotonic()
            milestone = (
                progress_every
                and completed
                and completed != last_reported_completed
                and completed % progress_every == 0
            )
            periodic_due = periodic and now - last_status_at >= status_seconds
            if not milestone and not periodic_due:
                return
            elapsed = now - started_at
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            remaining = len(records) - completed
            eta = f"{remaining / rate:.0f}s" if rate > 0.0 else "unknown"
            active = sum(slot["active"] is not None for slot in self._slots)
            print(
                "[verify-media-decode] "
                f"completed={completed}/{len(records)} active={active} accepted={len(accepted)} "
                f"rejected={initial_rejection_count + len(rejected)} timed_out={timeout_count} "
                f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta}",
                flush=True,
            )
            last_status_at = now
            last_reported_completed = completed

        def assign(slot: dict[str, Any]) -> bool:
            try:
                index = next(pending)
            except StopIteration:
                return False
            media_type, media_element = jobs[index]
            slot["connection"].send((index, media_type, media_element, required_frames))
            slot["active"] = index
            slot["assigned_at"] = time.monotonic()
            return True

        for slot in self._slots:
            assign(slot)

        try:
            while completed < len(records):
                active_slots = [slot for slot in self._slots if slot["active"] is not None]
                if not active_slots:
                    break
                connections = [slot["connection"] for slot in active_slots]
                positions = {
                    slot["connection"]: position for position, slot in enumerate(self._slots)
                }
                for connection in wait(connections, timeout=0.2):
                    position = positions[connection]
                    slot = self._slots[position]
                    try:
                        event = connection.recv()
                    except (EOFError, OSError) as exc:
                        index = int(slot["active"])
                        fail_worker(
                            slot,
                            index,
                            "decoder worker pipe closed unexpectedly; "
                            f"exitcode={slot['process'].exitcode}; error={type(exc).__name__}: {exc}",
                        )
                    kind = event[0]
                    if kind == "started":
                        continue
                    index = int(event[1])
                    if index != slot["active"]:
                        raise RuntimeError(
                            f"decoder worker {slot['worker_id']} returned unexpected record {index}"
                        )
                    if kind != "result":
                        raise RuntimeError(f"unexpected decoder worker event: {event!r}")
                    if bool(event[2]):
                        accepted.add(index)
                    else:
                        rejected.append(
                            _safe_rejection(index, records[index], str(event[3]), str(event[4]))
                        )
                    slot["active"] = None
                    completed += 1
                    report()
                    assign(slot)

                now = time.monotonic()
                for position, slot in enumerate(list(self._slots)):
                    index = slot["active"]
                    if index is None:
                        continue
                    if not slot["process"].is_alive():
                        fail_worker(
                            slot,
                            int(index),
                            f"decoder child exitcode={slot['process'].exitcode}",
                        )
                    if now - float(slot["assigned_at"]) > self._timeout_seconds:
                        timeout_count += 1
                        rejected.append(
                            _safe_rejection(
                                int(index),
                                records[int(index)],
                                "timeout",
                                "qwen media preprocessing exceeded "
                                f"{self._timeout_seconds:.1f}s",
                            )
                        )
                        slot["active"] = None
                        completed += 1
                        report()
                        assign(self._replace(position))
                report(periodic=True)
        finally:
            self.close()

        return accepted, rejected, timeout_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="path-verified mixed JSON list")
    parser.add_argument(
        "--media-root",
        "--video-root",
        dest="media_root",
        type=Path,
        required=True,
        help="root used to resolve the raw relative image/video paths",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nframes", type=_even_nframes, default=8)
    parser.add_argument(
        "--max-pixels",
        type=_positive_int,
        default=100352,
        help="video cap passed to qwen-vl-utils, matching the current direct trainer",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=_positive_int,
        default=None,
        help="optional image cap; omit to mirror the current trainer's image defaults",
    )
    parser.add_argument("--workers", type=_positive_int, default=4)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--status-seconds",
        type=_positive_float,
        default=30.0,
        help="emit a flushed terminal heartbeat even when no record has completed",
    )
    parser.add_argument("--rejection-record-limit", type=int, default=200)
    parser.add_argument(
        "--reader-backend",
        choices=("torchvision",),
        default="torchvision",
        help="force the qwen-vl-utils local video reader used by verification/training",
    )
    args = parser.parse_args()

    if args.progress_every < 0 or args.rejection_record_limit < 0:
        parser.error("progress and rejection limits must be non-negative")
    if not args.source.is_file():
        parser.error(f"source JSON is unavailable: {args.source}")
    if not args.media_root.is_dir():
        parser.error(f"media root is unavailable: {args.media_root}")
    if args.output.resolve() == args.source.resolve():
        parser.error("--output must differ from --source")

    loaded = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not all(isinstance(record, dict) for record in loaded):
        raise ValueError("source JSON must be a list of object records")
    records: list[dict[str, Any]] = loaded
    root = args.media_root.resolve()
    jobs: dict[int, tuple[MediaType, dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        media_type = _media_type(record)
        if media_type is None:
            rejected.append(
                _safe_rejection(
                    index,
                    record,
                    "unsupported_data_type",
                    "record data_type must be exactly image or video",
                )
            )
            continue
        path = _safe_regular_nonempty_path(root, record.get("path"))
        if path is None:
            rejected.append(
                _safe_rejection(
                    index,
                    record,
                    "missing_or_empty",
                    "unsafe, missing, empty, or unreadable media",
                )
            )
            continue
        jobs[index] = (
            media_type,
            _build_media_element(
                media_type,
                path,
                nframes=args.nframes,
                max_pixels=args.max_pixels,
                image_max_pixels=args.image_max_pixels,
            ),
        )

    # Apply before workers spawn and before their lazy qwen/torch imports.  No
    # decoder process should see a training GPU as an available device.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FORCE_QWENVL_VIDEO_READER"] = args.reader_backend

    source_counts = _count_records(records)
    candidate_counts = _count_records(records, set(jobs))
    print(
        "[verify-media-decode] "
        f"source_records={len(records)} candidates={len(jobs)} "
        f"images={candidate_counts['image']} videos={candidate_counts['video']} "
        f"workers={args.workers} timeout_seconds={args.timeout_seconds:g} "
        f"backend={args.reader_backend} cuda_visible_devices=hidden",
        flush=True,
    )

    verification_complete = True
    verification_error: str | None = None
    timeout_count = 0
    accepted: set[int] = set()
    worker_rejections: list[dict[str, Any]] = []
    completed_records = len(records) - len(jobs)
    if jobs:
        supervisor = MediaDecodeSupervisor(args.workers, args.timeout_seconds)
        try:
            accepted, worker_rejections, timeout_count = supervisor.run(
                records,
                jobs,
                required_frames=args.nframes,
                progress_every=args.progress_every,
                initial_rejection_count=len(rejected),
                status_seconds=args.status_seconds,
            )
            completed_records = len(records)
        except DecoderWorkerCrash as exc:
            accepted = exc.accepted
            worker_rejections = exc.rejections
            timeout_count = exc.timeout_count
            completed_records = exc.completed
            verification_complete = False
            verification_error = str(exc)

    rejected.extend(worker_rejections)
    rejected.sort(key=lambda item: int(item["source_index"]))
    rejected_indices = {int(item["source_index"]) for item in rejected}
    verified = [record for index, record in enumerate(records) if index in accepted]
    verified_indices = set(accepted)
    rejection_path = args.output.with_suffix(args.output.suffix + ".rejections.json")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    atomic_json_write(rejection_path, rejected)

    output_published = False
    output_sha256: str | None = None
    if verification_complete:
        # Publish an empty JSON list after a complete no-media run too: it is
        # unambiguous evidence of what was verified, while status still fails
        # so a launcher cannot accidentally train an empty dataset.
        atomic_json_write(args.output, verified)
        output_published = True
        output_sha256 = sha256_file(args.output)

    if not verification_complete:
        status = "failed_worker_crash"
    elif not verified:
        status = "failed_no_verified_records"
    else:
        status = "passed"

    timed_out_indices = {
        int(item["source_index"]) for item in rejected if item.get("reason") == "timeout"
    }
    manifest = {
        "status": status,
        "verification_complete": verification_complete,
        "verification_error": verification_error,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "media_root": str(root),
        "output": str(args.output.resolve()),
        "output_published": output_published,
        "output_sha256": output_sha256,
        "rejection_file": str(rejection_path.resolve()),
        "rejection_file_sha256": sha256_file(rejection_path),
        "decoder": {
            "image": "qwen_vl_utils.vision_process.fetch_image",
            "video": "qwen_vl_utils.vision_process.fetch_video",
            "reader_backend": args.reader_backend,
            "cuda_visible_devices": "",
        },
        "semantics": {
            "source_order_preserved": True,
            "raw_relative_paths_preserved": True,
            "path_resolution": "grpo_prepare_dataset.resolve_media_path",
            "video_nframes": args.nframes,
            "video_max_pixels": args.max_pixels,
            "image_max_pixels": args.image_max_pixels,
            "image_default_matches_current_direct_trainer": args.image_max_pixels is None,
            "prompt_tokenization_or_model_execution": False,
            "per_record_hard_timeout_seconds": args.timeout_seconds,
        },
        "workers": args.workers,
        "status_seconds": args.status_seconds,
        "source_records": len(records),
        "candidate_records": len(jobs),
        "verified_records": len(verified),
        "rejected_records": len(rejected),
        "timed_out_records": timeout_count,
        "completed_records": completed_records,
        "unprocessed_records": max(0, len(records) - completed_records),
        "modality_counts": {
            "source": source_counts,
            "candidate": candidate_counts,
            "verified": _count_records(records, verified_indices),
            "rejected": _count_records(records, rejected_indices),
            "timed_out": _count_records(records, timed_out_indices),
        },
        "rejection_examples": rejected[: args.rejection_record_limit],
    }
    atomic_json_write(manifest_path, manifest)

    summary = (
        "[verify-media-decode] "
        f"{status.upper()} verified={len(verified)} rejected={len(rejected)} "
        f"timed_out={timeout_count} manifest={manifest_path} rejections={rejection_path}"
    )
    print(summary, flush=True)
    if status != "passed":
        for example in rejected[:3]:
            print(
                "[verify-media-decode] rejection_example="
                + json.dumps(example, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        raise RuntimeError("mixed-media decoder verification failed; inspect manifest and rejections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
