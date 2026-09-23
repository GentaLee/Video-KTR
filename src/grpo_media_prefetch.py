"""CPU-only, ordered loading of already verified Qwen media cache payloads.

The collator is deliberately independent of the trainer and model so DataLoader
workers can use ``spawn`` after a training rank has initialized CUDA.  It never
decodes on a cache miss, changes the examples, or chooses substitute examples.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

from grpo_prepare_dataset import resolve_media_path


MEDIA_PAYLOAD_KEY = "_video_ktr_preprocessed_media"


@dataclass
class PreparedMedia:
    """Opaque CPU payload: Accelerate must not move raw frames to CUDA.

    In particular, do not add a ``to`` method or inherit from dict/list.  The
    normal processor and H2D path in the training rank remains responsible for
    producing model inputs.  The tensor/PIL fields are safe to pickle.
    """

    media_type: str
    media_path: str
    nframes: int
    max_pixels: int
    image_inputs: list[Any] | None
    video_inputs: list[Any] | None
    video_kwargs: dict[str, Any]


@dataclass
class CachedMediaCollator:
    data_root: str
    cache_root: str
    nframes: int
    max_pixels: int
    _worker_pid: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import torch
        from PIL import Image
        from grpo_media_cache import load_cached_media

        # With num_workers=0 this runs in the training rank: leave its thread
        # settings alone.  A spawned worker only performs bounded CPU I/O.
        if torch.utils.data.get_worker_info() is not None and self._worker_pid != os.getpid():
            torch.set_num_threads(1)
            self._worker_pid = os.getpid()

        root = Path(self.data_root).resolve()
        prepared: list[dict[str, Any]] = []
        for feature in features:
            if MEDIA_PAYLOAD_KEY in feature:
                raise ValueError("dataset uses the reserved preprocessed-media field")
            media_type = feature.get("data_type")
            if media_type not in {"image", "video"}:
                raise ValueError(f"unsupported cached media type: {media_type!r}")
            media = resolve_media_path(root, feature.get("path"))
            if media is None or not media.is_file() or media.stat().st_size == 0:
                raise ValueError(f"unsafe, missing, or empty media path: {feature.get('path')!r}")
            element: dict[str, Any] = {"type": media_type, media_type: str(media)}
            if media_type == "video":
                element.update(nframes=self.nframes, max_pixels=self.max_pixels)
            value, fps = load_cached_media(
                element, cache_root=self.cache_root, allow_create=False
            )
            if media_type == "video":
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                    raise ValueError("cached video must be a CPU tensor")
                if value.ndim != 4 or value.shape[0] != self.nframes or value.shape[1] != 3:
                    raise ValueError(f"cached video shape does not match training: {tuple(value.shape)}")
                if value.dtype != torch.float32 or fps is None or not math.isfinite(float(fps)) or float(fps) <= 0:
                    raise ValueError("cached video requires exact float32 frames and a positive sample fps")
                image_inputs, video_inputs = None, [value]
                video_kwargs = {"fps": [float(fps)]}
            else:
                if not isinstance(value, Image.Image):
                    raise ValueError("cached image must be a PIL image")
                image_inputs, video_inputs = [value], None
                # process_vision_info returns an empty fps list for images.
                video_kwargs = {"fps": []}
            example = dict(feature)
            example[MEDIA_PAYLOAD_KEY] = PreparedMedia(
                media_type=media_type,
                media_path=str(media),
                nframes=self.nframes,
                max_pixels=self.max_pixels,
                image_inputs=image_inputs,
                video_inputs=video_inputs,
                video_kwargs=video_kwargs,
            )
            prepared.append(example)
        return prepared


def use_spawn_for_cpu_prefetch(dataloader: Any) -> Any:
    """Configure the existing loader without replacing its sharded sampler.

    Pinned Accelerate exposes its torch loader through ``base_dataloader``;
    older versions directly inherit DataLoader.  Both permit setting the
    multiprocessing context before the first iterator starts workers.
    """

    base = getattr(dataloader, "base_dataloader", dataloader)
    if int(base.num_workers) > 0:
        base.multiprocessing_context = "spawn"
        if base.multiprocessing_context.get_start_method() != "spawn":
            raise RuntimeError("media cache prefetch workers must use spawn")
    return dataloader
