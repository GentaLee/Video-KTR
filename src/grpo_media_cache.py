"""Exact CPU media preprocessing cache shared by preflight and GRPO workers.

This caches the *output* of Qwen's fetch_image/fetch_video, not a recompressed
source video. Video tensors retain their float32 values and sampled FPS;
images retain RGB pixels in lossless PNG. A cache-only training read never
falls back to decoding. Source stat identity is checked on every access, but
is not protection against adversarial changes which forge filesystem times.
The strong source content hash is recorded at creation, not recomputed each
training step. Payload SHA256 is verified on every read.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any


CACHE_SCHEMA = "video-ktr-exact-media-v1"


class MediaCacheError(RuntimeError):
    """A missing, stale, or corrupt exact preprocessing cache entry."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def runtime_fingerprint() -> dict[str, Any]:
    """Identify actual installed preprocessing code and its environment.

    The package metadata lookup deliberately avoids importing CUDA/torch in
    the supervising preflight process. Qwen's source hash and effective
    reader environment are part of the key; changing either invalidates it.
    """
    versions: dict[str, str | None] = {}
    for package in ("torch", "torchvision", "av", "Pillow", "decord", "qwen-vl-utils"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    spec = importlib.util.find_spec("qwen_vl_utils")
    locations = [] if spec is None else list(spec.submodule_search_locations or [])
    candidates = [Path(location) / "vision_process.py" for location in locations]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise MediaCacheError("cannot fingerprint installed qwen_vl_utils/vision_process.py")
    return {
        "versions": versions,
        "qwen_vision_process_sha256": _sha256(source),
        "environment": {
            name: os.environ.get(name)
            for name in ("VIDEO_MAX_PIXELS", "FORCE_QWENVL_VIDEO_READER")
        },
    }


def _source_identity(path: Path) -> dict[str, Any]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise MediaCacheError(f"media must be a nonempty regular file: {path}")
    return {
        "path": str(path), "size": info.st_size,
        "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
        "device": info.st_dev, "inode": info.st_ino,
    }


def _contract(element: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
    media_type = element.get("type")
    if media_type not in ("image", "video"):
        raise MediaCacheError("exact media cache requires type=image or type=video")
    raw_path = element.get(media_type)
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise MediaCacheError("exact media cache requires an absolute local media path")
    path = Path(raw_path).resolve(strict=True)
    normalized = dict(element)
    normalized[media_type] = str(path)
    contract = {
        "schema": CACHE_SCHEMA, "element": normalized,
        "source": _source_identity(path), "runtime": runtime_fingerprint(),
    }
    try:
        key = hashlib.sha256(_json_bytes(contract)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise MediaCacheError("media preprocessing parameters must be finite JSON values") from exc
    return contract, path, key


def _decode(element: dict[str, Any]) -> tuple[Any, float | None]:
    from qwen_vl_utils.vision_process import fetch_image, fetch_video
    if element["type"] == "image":
        return fetch_image(element), None
    return fetch_video(element, return_video_sample_fps=True)


def _assert_source_unchanged(path: Path, contract: dict[str, Any]) -> None:
    if _source_identity(path) != contract["source"]:
        raise MediaCacheError(f"source media changed during cache access: {path}")


def _describe_payload(payload: Any, media_type: str, fps: Any) -> dict[str, Any]:
    if media_type == "image":
        from PIL import Image
        if not isinstance(payload, Image.Image) or payload.mode != "RGB":
            raise MediaCacheError("Qwen image output must be an RGB PIL image")
        if min(payload.size) < 1 or fps is not None:
            raise MediaCacheError("invalid cached image dimensions or FPS")
        return {"kind": "image", "mode": "RGB", "size": list(payload.size), "fps": None}
    import torch
    if not isinstance(payload, torch.Tensor):
        raise MediaCacheError("Qwen video output must be a torch tensor")
    if payload.device.type != "cpu" or payload.dtype != torch.float32:
        raise MediaCacheError("Qwen video output must be CPU float32; refusing a lossy conversion")
    if payload.ndim != 4 or payload.shape[1] != 3 or min(payload.shape) < 1:
        raise MediaCacheError(f"invalid Qwen video shape: {tuple(payload.shape)}")
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise MediaCacheError(f"invalid sampled video FPS: {fps!r}")
    return {"kind": "video", "dtype": "float32", "shape": list(payload.shape), "fps": float(fps)}


def _read_entry(entry: Path, contract: dict[str, Any], key: str) -> tuple[Any, float | None]:
    try:
        metadata = json.loads((entry / "metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("status") != "complete" or metadata.get("key") != key
                or metadata.get("contract") != contract):
            raise MediaCacheError(f"cache certificate does not match runtime/source: {entry}")
        source_hash = metadata.get("source_sha256", "")
        if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
            raise MediaCacheError(f"cache source hash certificate is missing: {entry}")
        media_type = contract["element"]["type"]
        filename = "image.png" if media_type == "image" else "video.pt"
        payload_path = entry / filename
        if _sha256(payload_path) != metadata["payload_sha256"]:
            raise MediaCacheError(f"cache payload SHA256 mismatch: {entry}")
        if media_type == "image":
            from PIL import Image
            with Image.open(payload_path) as image:
                payload = image.copy()
        else:
            import torch
            payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        fps = metadata["payload"]["fps"]
        if _describe_payload(payload, media_type, fps) != metadata["payload"]:
            raise MediaCacheError(f"cache payload metadata mismatch: {entry}")
        return payload, fps
    except MediaCacheError:
        raise
    except Exception as exc:
        raise MediaCacheError(f"cannot read exact media cache entry {entry}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(entry: Path, contract: dict[str, Any], source: Path, key: str) -> None:
    """Caller holds the per-key lock; publish only a fully complete entry."""
    pending = Path(tempfile.mkdtemp(prefix=f".{key}.pending-", dir=entry.parent))
    try:
        source_hash = _sha256(source)
        _assert_source_unchanged(source, contract)
        payload, fps = _decode(contract["element"])
        description = _describe_payload(payload, contract["element"]["type"], fps)
        if description["kind"] == "image":
            payload_path = pending / "image.png"
            payload.save(payload_path, format="PNG")
        else:
            import torch
            payload_path = pending / "video.pt"
            # No uint8 conversion/re-encode: resized fractional pixel values
            # are part of the original model input and must remain exact.
            torch.save(payload.detach().contiguous(), payload_path)
        with payload_path.open("rb") as stream:
            os.fsync(stream.fileno())
        metadata = {
            "status": "complete", "key": key, "contract": contract,
            "source_sha256": source_hash,
            "payload_sha256": _sha256(payload_path), "payload": description,
        }
        with (pending / "metadata.json").open("wb") as stream:
            stream.write(_json_bytes(metadata))
            stream.flush()
            os.fsync(stream.fileno())
        _assert_source_unchanged(source, contract)
        _fsync_directory(pending)
        os.replace(pending, entry)
        _fsync_directory(entry.parent)
    finally:
        if pending.exists():
            # Only our explicit mkdtemp child, never a caller-supplied root.
            shutil.rmtree(pending)


def load_cached_media(
    element: dict[str, Any], *, cache_root: Path | str, allow_create: bool = False,
) -> tuple[Any, float | None]:
    """Return exact Qwen CPU preprocessing output and video sampled FPS.

    ``allow_create=True`` is for supervised preflight: first miss decodes
    once under a per-key process lock. ``False`` is for training: missing or
    corrupt entries fail clearly, with no potentially unbounded fallback.
    A corrupt published entry always fails (even for creators), preserving
    evidence; it is not silently trusted or overwritten. No source changes
    are ever made. A returned PIL image/tensor belongs to this caller.
    """
    try:
        contract, source, key = _contract(element)
        entry = Path(cache_root).expanduser().resolve() / CACHE_SCHEMA / key[:2] / key
        if not entry.exists():
            if not allow_create:
                raise MediaCacheError(
                    f"exact media cache missing or invalidated for {source}; "
                    f"run supervised media cache preflight first (entry={entry})"
                )
            entry.parent.mkdir(parents=True, exist_ok=True)
            with (entry.parent / f".{key}.lock").open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if not entry.exists():
                    _assert_source_unchanged(source, contract)
                    _publish(entry, contract, source, key)
        result = _read_entry(entry, contract, key)
        _assert_source_unchanged(source, contract)
        return result
    except MediaCacheError:
        raise
    except Exception as exc:
        raise MediaCacheError(f"exact media cache access failed: {exc}") from exc
