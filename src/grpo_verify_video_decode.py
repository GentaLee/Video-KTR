#!/usr/bin/env python3
"""Build a decoder-verified Video-R1 manifest with hard per-video timeouts.

The training path uses qwen-vl-utils' video preprocessing, which on the
validated profile is explicitly pinned to ``torchvision.io.read_video``.  It
reads a whole clip before selecting ``nframes``.  A damaged or pathological
clip can therefore stall a single distributed rank before its next ZeRO-3
collective, leaving the other ranks to time out.  File existence alone is not
enough evidence that a record is safe for this reader.

This tool keeps the potentially stuck decoder inside a short-lived worker
process.  The supervisor can terminate only that child on timeout, record the
rejection, and continue.  The output JSON preserves input order for accepted
records and is accompanied by a manifest of all rejections.  It is intended
to run on the GPU node with CUDA hidden from its CPU-only decoder workers.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
import time
from typing import Any

from grpo_prepare_dataset import atomic_json_write, resolve_media_path, sha256_file


def _decode_one(media_element: dict[str, Any], required_frames: int) -> tuple[int, list[int]]:
    """Run the same qwen-vl-utils video decode, sampling, and resize path.

    The training trainer mutates its first video content item with a resolved
    path, ``nframes``, and ``max_pixels`` before calling
    :func:`process_vision_info`.  ``fetch_video`` is the exact per-video
    operation reached from there, so executing it here detects not only
    unreadable media but also sampling and resize failures.  Keep the import
    inside the spawned CPU-only worker: FFmpeg errors then remain killable.
    """

    from qwen_vl_utils.vision_process import fetch_video

    video = fetch_video(media_element)
    if not hasattr(video, "ndim"):
        raise RuntimeError(f"unexpected decoded video type: {type(video).__name__}")
    if video.ndim != 4 or video.shape[1] != 3:
        raise RuntimeError(f"unexpected decoded tensor shape: {tuple(video.shape)}")
    frame_count = int(video.shape[0])
    if frame_count < required_frames:
        raise RuntimeError(
            f"decoded only {frame_count} frame(s), need at least {required_frames}"
        )
    return frame_count, [int(dimension) for dimension in video.shape]


def _decoder_worker(connection: Connection) -> None:
    """Decode one assigned record at a time and report lifecycle events."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    while True:
        job = connection.recv()
        if job is None:
            return
        index, media_element, required_frames = job
        connection.send(("started", index))
        try:
            frame_count, shape = _decode_one(media_element, required_frames)
        except BaseException as exc:
            connection.send(("result", index, False, type(exc).__name__, str(exc)[:1000]))
        else:
            connection.send(("result", index, True, frame_count, shape))


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
    """Match Qwen's frame-factor requirement instead of silently rounding."""

    parsed = _positive_int(value)
    if parsed < 2 or parsed % 2:
        raise argparse.ArgumentTypeError("must be an even integer of at least 2")
    return parsed


def _safe_rejection(
    source_index: int, record: dict[str, Any], reason: str, detail: str
) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "problem_id": record.get("problem_id"),
        "path": record.get("path"),
        "reason": reason,
        "detail": detail[:1000],
    }


class DecoderWorkerCrash(RuntimeError):
    """An unexpected worker death is a verifier failure, not a media filter."""

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


class DecodeSupervisor:
    """A bounded group of killable decoder workers, one active job each."""

    def __init__(self, worker_count: int, timeout_seconds: float) -> None:
        self._context = mp.get_context("spawn")
        self._timeout_seconds = timeout_seconds
        self._slots: list[dict[str, Any]] = []
        for worker_id in range(worker_count):
            self._slots.append(self._spawn(worker_id))

    def _spawn(self, worker_id: int) -> dict[str, Any]:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_decoder_worker,
            args=(child,),
            name=f"video-decode-worker-{worker_id}",
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
        """Close a worker without allowing an unkillable decoder to accumulate.

        ``Process.terminate`` is normally enough for a stalled FFmpeg call.
        If it is not, a replacement worker while the original survives would
        silently turn one bad clip into an unbounded process leak.  Escalate
        once to SIGKILL and fail the verifier if the kernel still cannot reap
        the child; the operator can then inspect the host instead of training
        against an incomplete safety check.
        """

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

    def replace(self, position: int, *, terminate: bool) -> dict[str, Any]:
        old = self._slots[position]
        worker_id = int(old["worker_id"])
        self._close_slot(old, terminate=terminate)
        replacement = self._spawn(worker_id)
        self._slots[position] = replacement
        return replacement

    def close(self) -> None:
        for slot in self._slots:
            self._close_slot(slot, terminate=False)

    def run(
        self,
        records: list[dict[str, Any]],
        media_elements: dict[int, dict[str, Any]],
        required_frames: int,
        progress_every: int,
        initial_rejection_count: int,
        status_seconds: float,
    ) -> tuple[set[int], list[dict[str, Any]], int]:
        pending = iter(sorted(media_elements))
        accepted: set[int] = set()
        rejected: list[dict[str, Any]] = []
        timeout_count = 0
        completed = len(records) - len(media_elements)
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
            completed_milestone = (
                progress_every
                and completed
                and completed != last_reported_completed
                and completed % progress_every == 0
            )
            elapsed = now - started_at
            periodic_due = periodic and now - last_status_at >= status_seconds
            if not completed_milestone and not periodic_due:
                return
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            remaining = len(records) - completed
            eta_seconds = remaining / rate if rate > 0.0 else None
            eta_text = f"{eta_seconds:.0f}s" if eta_seconds is not None else "unknown"
            active = sum(slot["active"] is not None for slot in self._slots)
            print(
                "[verify-video-decode] "
                f"completed={completed}/{len(records)} active={active} accepted={len(accepted)} "
                f"rejected={initial_rejection_count + len(rejected)} timed_out={timeout_count} "
                f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta_text}",
                flush=True,
            )
            last_status_at = now
            last_reported_completed = completed

        def assign(slot: dict[str, Any]) -> bool:
            try:
                index = next(pending)
            except StopIteration:
                return False
            slot["connection"].send((index, media_elements[index], required_frames))
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
                by_connection = {slot["connection"]: position for position, slot in enumerate(self._slots)}
                for connection in wait(connections, timeout=0.2):
                    position = by_connection[connection]
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
                    if kind == "result":
                        if bool(event[2]):
                            accepted.add(index)
                        else:
                            rejected.append(
                                _safe_rejection(
                                    index, records[index], str(event[3]), str(event[4])
                                )
                            )
                        slot["active"] = None
                        completed += 1
                        report()
                        assign(slot)
                    else:
                        raise RuntimeError(f"unexpected decoder worker event: {event!r}")

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
                    elif now - float(slot["assigned_at"]) > self._timeout_seconds:
                        timeout_count += 1
                        rejected.append(
                            _safe_rejection(
                                index,
                                records[index],
                                "timeout",
                                f"qwen video preprocessing exceeded {self._timeout_seconds:.1f}s",
                            )
                        )
                        slot["active"] = None
                        completed += 1
                        report()
                        replacement = self.replace(position, terminate=True)
                        assign(replacement)
                report(periodic=True)
        finally:
            self.close()

        return accepted, rejected, timeout_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="path-verified input JSON list")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nframes", type=_even_nframes, default=8)
    parser.add_argument("--workers", type=_positive_int, default=4)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--status-seconds",
        type=_positive_float,
        default=30.0,
        help="emit a terminal heartbeat even when no record has completed",
    )
    parser.add_argument("--rejection-record-limit", type=int, default=200)
    parser.add_argument(
        "--max-pixels",
        type=_positive_int,
        default=100352,
        help="same qwen-vl-utils per-frame pixel limit used by training",
    )
    parser.add_argument(
        "--reader-backend",
        choices=("torchvision",),
        default="torchvision",
        help="force the qwen-vl-utils reader used by both verification and training",
    )
    args = parser.parse_args()

    if args.progress_every < 0 or args.rejection_record_limit < 0:
        parser.error("progress and rejection limits must be non-negative")
    if not args.source.is_file():
        parser.error(f"source JSON is unavailable: {args.source}")
    if not args.video_root.is_dir():
        parser.error(f"video root is unavailable: {args.video_root}")
    if args.output.resolve() == args.source.resolve():
        parser.error("--output must differ from --source")

    loaded = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not all(isinstance(record, dict) for record in loaded):
        raise ValueError("source JSON must be a list of object records")
    records: list[dict[str, Any]] = loaded
    media_elements: dict[int, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    root = args.video_root.resolve()
    for index, record in enumerate(records):
        if record.get("data_type") != "video":
            rejected.append(
                _safe_rejection(index, record, "wrong_type", "record is not data_type=video")
            )
            continue
        path = resolve_media_path(root, record.get("path"))
        if path is None or not path.is_file() or path.stat().st_size == 0:
            rejected.append(
                _safe_rejection(index, record, "missing_or_empty", "unsafe, missing, or empty media")
            )
            continue
        # ``open_r1/grpo.py`` maps every raw record to a fresh prompt whose
        # first content item is exactly ``{"type": data_type}``; raw manifests
        # deliberately do not carry a prompt field.  Reconstruct that item
        # rather than treating the raw record as malformed, then apply the
        # same mutations as ``VideoKTRGRPOTrainer._prepare_prompt``.
        media_element: dict[str, Any] = {"type": "video"}
        media_element["video"] = str(path)
        media_element["nframes"] = args.nframes
        media_element["max_pixels"] = args.max_pixels
        media_elements[index] = media_element

    # qwen-vl-utils reads this at module import time in every spawned worker.
    os.environ["FORCE_QWENVL_VIDEO_READER"] = args.reader_backend

    print(
        "[verify-video-decode] "
        f"source_records={len(records)} candidate_videos={len(media_elements)} "
        f"workers={args.workers} timeout_seconds={args.timeout_seconds:g} "
        f"backend={args.reader_backend} qwen_preprocess=enabled",
        flush=True,
    )
    supervisor = DecodeSupervisor(args.workers, args.timeout_seconds)
    verification_complete = True
    verification_error: str | None = None
    completed_records: int
    try:
        accepted, worker_rejections, timeout_count = supervisor.run(
            records,
            media_elements,
            args.nframes,
            args.progress_every,
            len(rejected),
            args.status_seconds,
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
    verified = [record for index, record in enumerate(records) if index in accepted]
    rejection_path = args.output.with_suffix(args.output.suffix + ".rejections.json")
    atomic_json_write(rejection_path, rejected)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if verification_complete and verified:
        atomic_json_write(args.output, verified)
        status = "passed"
        output_sha256: str | None = sha256_file(args.output)
    elif not verification_complete:
        status = "failed_worker_crash"
        output_sha256 = None
    else:
        # Retain the rejection evidence even for an environment-wide decoder
        # failure. Without this, a generic exception would hide whether the
        # cause was a missing import, a codec failure, or hard timeouts.
        status = "failed_no_verified_records"
        output_sha256 = None
    manifest = {
        "status": status,
        "verification_complete": verification_complete,
        "verification_error": verification_error,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "video_root": str(root),
        "output": str(args.output.resolve()),
        "decoder": "qwen_vl_utils.vision_process.fetch_video",
        "reader_backend": args.reader_backend,
        "qwen_preprocess": True,
        "nframes": args.nframes,
        "max_pixels": args.max_pixels,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "status_seconds": args.status_seconds,
        "source_records": len(records),
        "candidate_videos": len(media_elements),
        "verified_records": len(verified),
        "rejected_records": len(rejected),
        "timed_out_records": timeout_count,
        "completed_records": completed_records,
        "unprocessed_records": max(0, len(records) - completed_records),
        "output_sha256": output_sha256,
        "rejection_file": str(rejection_path.resolve()),
        "rejection_file_sha256": sha256_file(rejection_path),
        "rejection_examples": rejected[: args.rejection_record_limit],
    }
    atomic_json_write(manifest_path, manifest)
    if status != "passed":
        print(
            "[verify-video-decode] "
            f"FAIL status={status} verified={len(verified)} rejected={len(rejected)} "
            f"timed_out={timeout_count} "
            f"manifest={manifest_path} rejections={rejection_path}",
            flush=True,
        )
        for example in rejected[:3]:
            print(
                "[verify-video-decode] rejection_example="
                + json.dumps(example, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        raise RuntimeError(
            "decoder verification failed; inspect the retained manifest and rejections"
        )
    print(
        "[verify-video-decode] "
        f"PASS verified={len(verified)} rejected={len(rejected)} timed_out={timeout_count} "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
