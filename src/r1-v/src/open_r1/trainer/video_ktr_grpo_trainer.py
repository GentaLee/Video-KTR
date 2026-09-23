"""Correct, observable direct GRPO path for Video-KTR.

This module deliberately does not use the checked-in vLLM trainer: that path
does not apply the E/V/T selection mask to the GRPO or KL terms.  It retains
the upstream trainer's model/DeepSpeed setup, but replaces its fragile
``compute_loss`` implementation with a single-sample-per-rank path that:

* resolves media under an explicit data root and never substitutes fallback
  content;
* shares original Qwen mRoPE positions across original, visual-masked and
  temporal counterfactual forwards;
* applies the selected ``E union V union T`` mask to *both* policy and KL;
* writes bounded, rank-safe selected-token evidence for comparison runs.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
from torch.utils.data import Sampler

from .ktr_token_utils import (
    completion_valid_mask,
    compute_logp_delta,
    compute_token_entropy,
    generate_with_synced_gpus,
    nonidentity_frame_permutations,
    requires_synced_generation,
    select_top_ratio,
    target_token_logps,
    union_masks,
    values_match,
)
from qwen_vl_utils import process_vision_info
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation

from .grpo_trainer import Qwen2VLGRPOTrainer


class ModalityBlockSampler(Sampler[int]):
    """Yield a reproducible order whose global blocks have one media modality.

    ``accelerate`` shards a normal per-device batch sampler by handing rank
    ``r`` the ``r``-th microbatch in each global group.  With a batch size of
    one, a plain random order can therefore hand an image to one ZeRO-3 rank
    and a video to another at the same optimizer step.  Video-KTR has a real
    temporal counterfactual forward only for videos, so that mixed group would
    enter a different sequence of model/collective calls.

    This sampler preserves a random epoch order at *block* granularity while
    making every global block ``world_size * per_device_batch_size`` records of
    exactly one modality.  A non-divisible modality tail is padded only with
    records from that same modality.  The caller records the exact plan so the
    few deliberate repeats are visible in run provenance.
    """

    _MODALITIES = ("image", "video")

    def __init__(
        self,
        data_types: Sequence[object],
        *,
        world_size: int,
        per_device_batch_size: int,
        seed: int,
        plan_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if world_size < 1:
            raise ValueError("world_size must be at least one")
        if per_device_batch_size < 1:
            raise ValueError("per_device_batch_size must be at least one")
        self.world_size = int(world_size)
        self.per_device_batch_size = int(per_device_batch_size)
        self.global_block_size = self.world_size * self.per_device_batch_size
        self.seed = int(seed)
        self._plan_callback = plan_callback
        self._epoch = 0
        self._cached_epoch: int | None = None
        self._cached_indices: list[int] = []
        self._cached_summary: dict[str, Any] = {}
        self._indices_by_modality: dict[str, list[int]] = {
            modality: [] for modality in self._MODALITIES
        }
        self._modality_by_index: dict[int, str] = {}
        invalid: list[tuple[int, object]] = []
        for index, value in enumerate(data_types):
            modality = str(value)
            if modality not in self._indices_by_modality:
                invalid.append((index, value))
            else:
                self._indices_by_modality[modality].append(index)
                self._modality_by_index[index] = modality
        if invalid:
            examples = ", ".join(
                f"index={index}, data_type={value!r}" for index, value in invalid[:4]
            )
            raise ValueError(
                "ModalityBlockSampler supports only data_type=image/video; "
                f"invalid examples: {examples}"
            )
        if not any(self._indices_by_modality.values()):
            raise ValueError("ModalityBlockSampler requires at least one record")

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self._epoch:
            self._epoch = epoch
            self._cached_epoch = None

    def _build_plan(self) -> None:
        if self._cached_epoch == self._epoch:
            return

        # Keep the plan fully independent of global PyTorch RNG state: every
        # distributed rank constructs the same ordered sampler locally.
        generator = random.Random(self.seed + self._epoch)
        blocks: list[tuple[str, list[int]]] = []
        padding_by_modality: dict[str, list[int]] = {
            modality: [] for modality in self._MODALITIES
        }
        records_by_modality = {
            modality: len(indices)
            for modality, indices in self._indices_by_modality.items()
        }

        for modality in self._MODALITIES:
            indices = list(self._indices_by_modality[modality])
            if not indices:
                continue
            generator.shuffle(indices)
            padding_count = (-len(indices)) % self.global_block_size
            if padding_count:
                # Recycle from this shuffled modality only.  This is
                # intentional and is persisted in the plan summary rather
                # than allowing accelerate's generic end padding to make a
                # mixed final global block.
                padding = [indices[offset % len(indices)] for offset in range(padding_count)]
                indices.extend(padding)
                padding_by_modality[modality] = padding
            blocks.extend(
                (modality, indices[start : start + self.global_block_size])
                for start in range(0, len(indices), self.global_block_size)
            )

        generator.shuffle(blocks)
        flattened = [index for _modality, block in blocks for index in block]
        if len(flattened) % self.global_block_size:
            raise RuntimeError("internal error: modality blocks are not globally aligned")
        for start in range(0, len(flattened), self.global_block_size):
            block_modalities = {
                self._modality_for_index(index)
                for index in flattened[start : start + self.global_block_size]
            }
            if len(block_modalities) != 1:
                raise RuntimeError("internal error: generated a mixed-modality global block")

        summary = {
            "epoch": self._epoch,
            "seed": self.seed,
            "world_size": self.world_size,
            "per_device_batch_size": self.per_device_batch_size,
            "global_block_size": self.global_block_size,
            "input_records": sum(records_by_modality.values()),
            "records_by_modality": records_by_modality,
            "global_blocks": len(blocks),
            "emitted_records": len(flattened),
            "tail_padding_count": sum(len(values) for values in padding_by_modality.values()),
            "tail_padding_indices_by_modality": padding_by_modality,
        }
        self._cached_epoch = self._epoch
        self._cached_indices = flattened
        self._cached_summary = summary
        if self._plan_callback is not None:
            self._plan_callback(dict(summary))

    def _modality_for_index(self, index: int) -> str:
        try:
            return self._modality_by_index[index]
        except KeyError as exc:
            raise RuntimeError(f"internal error: sampler index {index} is unknown") from exc

    @property
    def plan_summary(self) -> dict[str, Any]:
        self._build_plan()
        return json.loads(json.dumps(self._cached_summary, sort_keys=True))

    def __iter__(self) -> Iterator[int]:
        self._build_plan()
        return iter(self._cached_indices)

    def __len__(self) -> int:
        self._build_plan()
        return len(self._cached_indices)


class VideoKTRGRPOTrainer(Qwen2VLGRPOTrainer):
    """A memory-conscious direct GRPO trainer with optional KTR masking.

    The inherited trainer assumes one media sample per rank.  We retain that
    deliberate constraint rather than silently mixing unrelated visual inputs
    in a batch.  Launchers enforce ``per_device_train_batch_size=1``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        script_args = kwargs.get("script_args")
        if script_args is None:
            raise ValueError("VideoKTRGRPOTrainer requires script_args")

        root_text = script_args.data_root or os.environ.get("VIDEO_ROOT")
        if not root_text:
            raise ValueError("set --data_root (or VIDEO_ROOT) to the mounted Video-R1 data root")
        self.data_root = Path(root_text).expanduser().resolve()
        if not self.data_root.is_dir():
            raise ValueError(f"data root is unavailable: {self.data_root}")

        self.nframes = int(script_args.nframes)
        self.selection_mode = str(script_args.selection_mode or "paper").lower()
        if self.selection_mode not in {"paper", "repo"}:
            raise ValueError("--selection_mode must be paper or repo")
        requested_scope = script_args.selection_scope
        self.selection_scope = (
            str(requested_scope).lower()
            if requested_scope
            else ("per_completion" if self.selection_mode == "paper" else "batch")
        )
        if self.selection_scope not in {"per_completion", "batch"}:
            raise ValueError("--selection_scope must be per_completion or batch")
        self.temporal_permutations = int(script_args.temporal_permutations)
        self.temporal_include_reverse = bool(script_args.temporal_include_reverse)
        self.temporal_seed = int(script_args.temporal_seed)
        self.token_record_limit = int(script_args.token_record_limit)
        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._token_record_path = Path(self.args.output_dir) / (
            f"selected_tokens_rank{self._local_rank}.jsonl"
        )
        self._rank_trace_enabled = os.environ.get("VIDEO_KTR_RANK_TRACE", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self._rank_trace_path = Path(self.args.output_dir) / (
            f"rank_trace_rank{self._local_rank}.jsonl"
        )
        self._generation_sync_reported = False
        self._modality_sampler: ModalityBlockSampler | None = None
        self._modality_sampler_plan_path = (
            Path(self.args.output_dir) / "modality_sampler_plan.jsonl"
        )
        self._media_cache_root = os.environ.get("VIDEO_KTR_MEDIA_CACHE_DIR", "").strip()
        if self._media_cache_root:
            from grpo_media_prefetch import CachedMediaCollator

            self.data_collator = CachedMediaCollator(
                data_root=str(self.data_root),
                cache_root=self._media_cache_root,
                nframes=self.nframes,
                max_pixels=self.max_pixels,
            )

    def get_train_dataloader(self) -> Any:
        dataloader = super().get_train_dataloader()
        if getattr(self, "_media_cache_root", ""):
            from grpo_media_prefetch import use_spawn_for_cpu_prefetch

            # Preserve HF/Accelerate sampling and sharding; only select how
            # its existing CPU workers are started before first iteration.
            use_spawn_for_cpu_prefetch(dataloader)
        return dataloader

    def _rank_trace(self, phase: str, **fields: Any) -> None:
        """Append a flushed per-rank phase marker when a diagnostic run opts in.

        A C-level decoder stall cannot reliably raise Python in the affected
        rank.  The last durable ``decode_start`` event then identifies the
        record that prevented that rank from reaching the next ZeRO
        collective, without interleaving four ranks' stdout.
        """

        if not self._rank_trace_enabled:
            return
        payload = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rank": self._local_rank,
            "global_step": int(self.state.global_step),
            "phase": phase,
            **fields,
        }
        self._rank_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._rank_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    def _synced_generation_enabled(self) -> bool:
        distributed = torch.distributed
        available = distributed.is_available()
        initialized = available and distributed.is_initialized()
        world_size = distributed.get_world_size() if initialized else 1
        return requires_synced_generation(available, initialized, world_size)

    def _record_modality_sampler_plan(self, summary: dict[str, Any]) -> None:
        """Persist the rank-0 sampler plan, including intentional tail repeats."""

        if not self.accelerator.is_main_process:
            return
        payload = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **summary,
        }
        self._modality_sampler_plan_path.parent.mkdir(parents=True, exist_ok=True)
        with self._modality_sampler_plan_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        padding = payload["tail_padding_indices_by_modality"]
        print(
            "[ktr] modality sampler "
            f"epoch={payload['epoch']} block={payload['global_block_size']} "
            f"blocks={payload['global_blocks']} "
            f"tail_padding={payload['tail_padding_count']} "
            f"(image={len(padding['image'])}, video={len(padding['video'])})",
            flush=True,
        )

    def _get_train_sampler(self) -> Sampler[int] | None:
        """Use global homogeneous blocks only when both media types exist.

        A homogeneous dataset is already globally modality-safe.  Delegating
        it to the parent sampler deliberately preserves the historical
        ``RandomSampler`` behavior used by the original all-video full run,
        so a checkpoint diagnostic can retain its prior data-order contract.
        Mixed image/video data instead needs deterministic global blocks,
        because one rank taking an image branch while another takes a video
        branch is unsafe with ZeRO-3.
        """

        if self.train_dataset is None:
            return None
        if not hasattr(self.train_dataset, "__len__"):
            raise ValueError(
                "VideoKTRGRPOTrainer requires a sized dataset for synchronized modality blocks"
            )
        try:
            data_types = list(self.train_dataset["data_type"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "VideoKTRGRPOTrainer dataset must retain a data_type column containing image/video"
            ) from exc
        if len(data_types) != len(self.train_dataset):
            raise RuntimeError(
                "dataset data_type column length differs from dataset length; cannot build modality blocks"
            )

        modalities = set(data_types)
        if not modalities:
            raise ValueError("VideoKTRGRPOTrainer requires at least one image/video record")
        if not modalities.issubset({"image", "video"}):
            invalid = sorted(repr(value) for value in modalities - {"image", "video"})
            raise ValueError(
                "VideoKTRGRPOTrainer data_type must be image/video; "
                f"got {', '.join(invalid)}"
            )
        if len(modalities) == 1:
            # Do not perturb the original sampler on a single-modality
            # dataset.  This branch is particularly important for resuming
            # the historical all-video checkpoint as a focused diagnostic.
            if self.accelerator.is_main_process:
                print(
                    "[ktr] homogeneous dataset uses the parent RandomSampler "
                    "to preserve the historical sampler contract",
                    flush=True,
                )
            return super()._get_train_sampler()

        world_size = int(getattr(self.accelerator, "num_processes", 1))
        per_device_batch_size = int(self.args.per_device_train_batch_size)
        configured_data_seed = getattr(self.args, "data_seed", None)
        seed = int(
            configured_data_seed
            if configured_data_seed is not None
            else getattr(self.args, "seed", 42)
        )
        if self._modality_sampler is None:
            self._modality_sampler = ModalityBlockSampler(
                data_types,
                world_size=world_size,
                per_device_batch_size=per_device_batch_size,
                seed=seed,
                plan_callback=self._record_modality_sampler_plan,
            )
        return self._modality_sampler

    @staticmethod
    def _local_modality_code(inputs: object) -> int:
        """Return an all-gatherable modality code without raising on one rank."""

        if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
            return -1
        example = inputs[0]
        if not isinstance(example, dict):
            return -1
        data_type = example.get("data_type")
        if data_type == "image":
            return 0
        if data_type == "video":
            return 1
        return -1

    def _collective_homogeneous_data_type(self, inputs: object) -> str:
        """Reject a mixed global microbatch before any decode or ZeRO forward.

        The check itself is one fixed-shape collective on every rank.  It
        catches an accidental sampler regression before image and video ranks
        can take different temporal-forward or metric branches.
        """

        local_code = self._local_modality_code(inputs)
        distributed = torch.distributed
        initialized = distributed.is_available() and distributed.is_initialized()
        world_size = distributed.get_world_size() if initialized else 1
        if world_size > 1:
            local = torch.tensor(
                [local_code], dtype=torch.int64, device=self.accelerator.device
            )
            gathered = [torch.empty_like(local) for _ in range(world_size)]
            distributed.all_gather(gathered, local)
            codes = [int(value.item()) for value in gathered]
        else:
            codes = [local_code]

        code_names = {-1: "invalid", 0: "image", 1: "video"}
        if any(code not in {0, 1} for code in codes):
            rendered = [code_names.get(code, f"unknown({code})") for code in codes]
            raise ValueError(
                "every direct Video-KTR rank requires one image/video record; "
                f"global modality codes={rendered}"
            )
        if len(set(codes)) != 1:
            rendered = [code_names[code] for code in codes]
            raise RuntimeError(
                "mixed image/video global microbatch is unsafe with ZeRO-3; "
                f"global modality order={rendered}. Use ModalityBlockSampler."
            )
        return code_names[codes[0]]

    @staticmethod
    def _remove_none_values(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict):
                    for key in [key for key, value in item.items() if value is None]:
                        del item[key]
        return messages

    def _resolve_media(self, raw_path: object) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("dataset record has no usable media path")
        raw = raw_path.strip()
        raw_candidate = Path(raw)
        # Validate the original path before removing only a cosmetic leading
        # "./".  Broad lstrip('./') would convert '../x' into 'x' and silently
        # substitute a different file below data_root.
        if raw_candidate.is_absolute() or not raw_candidate.parts or ".." in raw_candidate.parts:
            raise ValueError(f"unsafe dataset media path: {raw_path!r}")
        relative = Path(raw[2:] if raw.startswith("./") else raw)
        candidate = (self.data_root / relative).resolve()
        try:
            candidate.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError(f"dataset media path escapes data root: {raw_path!r}") from exc
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise FileNotFoundError(f"dataset media is unavailable or empty: {candidate}")
        return candidate

    @staticmethod
    def _to_device(value: Any, device: torch.device) -> Any:
        return value.to(device) if isinstance(value, torch.Tensor) else value

    def _prepare_prompt(
        self, inputs: list[dict[str, Any]]
    ) -> tuple[
        list[str],
        list[Any] | None,
        list[torch.Tensor] | None,
        dict[str, Any],
        dict[str, Any],
        str,
    ]:
        if len(inputs) != 1:
            raise ValueError(
                "VideoKTRGRPOTrainer supports per_device_train_batch_size=1 only; "
                f"received {len(inputs)} records"
            )
        example = inputs[0]
        data_type = str(example.get("data_type", ""))
        if data_type not in {"image", "video"}:
            raise ValueError(f"unsupported data_type for Video-KTR GRPO: {data_type!r}")

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
        ]
        messages = self._remove_none_values(copy.deepcopy(example["prompt"]))
        media = self._resolve_media(example.get("path"))
        content = messages[0].get("content")
        if not isinstance(content, list) or not content or not isinstance(content[0], dict):
            raise ValueError("unexpected prompt layout: first content item must be media")
        content[0][data_type] = str(media)
        if data_type == "video":
            content[0]["nframes"] = self.nframes
            content[0]["max_pixels"] = self.max_pixels

        self._rank_trace(
            "decode_start",
            problem_id=example.get("problem_id"),
            media_path=str(media),
            data_type=data_type,
        )
        try:
            if getattr(self, "_media_cache_root", ""):
                from grpo_media_prefetch import MEDIA_PAYLOAD_KEY, PreparedMedia

                prepared = example.get(MEDIA_PAYLOAD_KEY)
                if not isinstance(prepared, PreparedMedia):
                    raise RuntimeError("media cache mode requires the CPU cache collator payload")
                if (
                    prepared.media_type != data_type
                    or prepared.media_path != str(media)
                    or prepared.nframes != self.nframes
                    or prepared.max_pixels != self.max_pixels
                ):
                    raise RuntimeError("prefetched media identity or preprocessing settings changed")
                image_inputs = prepared.image_inputs
                video_inputs = prepared.video_inputs
                video_kwargs = prepared.video_kwargs
            else:
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
        except BaseException as exc:
            self._rank_trace(
                "decode_error",
                problem_id=example.get("problem_id"),
                media_path=str(media),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        if data_type == "video" and not video_inputs:
            self._rank_trace(
                "decode_error",
                problem_id=example.get("problem_id"),
                media_path=str(media),
                error_type="EmptyVideoDecode",
                error="video decoder returned no frame tensor",
            )
            raise RuntimeError(f"video decoder returned no frame tensor for {media}")
        self._rank_trace(
            "decode_done",
            problem_id=example.get("problem_id"),
            media_path=str(media),
            video_shape=(list(video_inputs[0].shape) if video_inputs else None),
        )
        encoded = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **video_kwargs,
        )
        prompt_inputs = {
            name: self._to_device(value, self.accelerator.device)
            for name, value in dict(encoded).items()
        }
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        if self.max_prompt_length is not None and prompt_length > self.max_prompt_length:
            raise RuntimeError(
                f"prompt has {prompt_length} tokens, beyond max_prompt_length={self.max_prompt_length}; "
                "do not truncate multimodal prompts because that breaks grid alignment"
            )
        return prompts_text, image_inputs, video_inputs, video_kwargs, prompt_inputs, data_type

    @staticmethod
    def _repeat_tensor(value: torch.Tensor, repeats: int) -> torch.Tensor:
        return value.repeat((repeats,) + (1,) * (value.ndim - 1))

    def _teacher_inputs(
        self,
        prompt_inputs: dict[str, Any],
        prompt_completion_ids: torch.Tensor,
        prompt_length: int,
        completion_attention: torch.Tensor,
    ) -> dict[str, Any]:
        """Append sampled completions and repeat multimodal tensors per sample."""

        repetitions = int(prompt_completion_ids.shape[0])
        result: dict[str, Any] = {}
        for name, value in prompt_inputs.items():
            if name == "input_ids":
                result[name] = prompt_completion_ids
            elif name == "attention_mask":
                prefix = value[:, :prompt_length].repeat(repetitions, 1).to(torch.long)
                result[name] = torch.cat((prefix, completion_attention.to(torch.long)), dim=1)
            elif name in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{name} must be a tensor")
                result[name] = self._repeat_tensor(value, repetitions)
            elif name == "second_per_grid_ts":
                if isinstance(value, torch.Tensor):
                    result[name] = self._repeat_tensor(value, repetitions)
                elif isinstance(value, list):
                    result[name] = value * repetitions
                else:
                    result[name] = value
            else:
                result[name] = value
        if "attention_mask" not in result:
            result["attention_mask"] = torch.cat(
                (
                    torch.ones_like(prompt_completion_ids[:, :prompt_length], dtype=torch.long),
                    completion_attention.to(torch.long),
                ),
                dim=1,
            )
        return result

    @staticmethod
    def _unwrap_rope_model(model: Any) -> Any:
        """Find the Qwen module that exposes ``get_rope_index`` under DDP/DS."""

        current = model
        seen: set[int] = set()
        while id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, "get_rope_index"):
                return current
            nested = getattr(current, "module", None)
            if nested is not None and nested is not current:
                current = nested
                continue
            get_base_model = getattr(current, "get_base_model", None)
            if callable(get_base_model):
                nested = get_base_model()
                if nested is not current:
                    current = nested
                    continue
            break
        raise RuntimeError("could not find Qwen get_rope_index on the training model")

    def _original_position_ids(self, model: Any, teacher_inputs: dict[str, Any]) -> torch.Tensor:
        rope_model = self._unwrap_rope_model(model)
        with torch.no_grad():
            position_ids, _ = rope_model.get_rope_index(
                input_ids=teacher_inputs["input_ids"],
                image_grid_thw=teacher_inputs.get("image_grid_thw"),
                video_grid_thw=teacher_inputs.get("video_grid_thw"),
                second_per_grid_ts=teacher_inputs.get("second_per_grid_ts"),
                attention_mask=teacher_inputs["attention_mask"],
            )
        expected = teacher_inputs["input_ids"].shape
        if position_ids.ndim != 3 or tuple(position_ids.shape[1:]) != tuple(expected):
            raise RuntimeError(
                "Qwen returned unexpected mRoPE positions: "
                f"{tuple(position_ids.shape)} for input IDs {tuple(expected)}"
            )
        return position_ids

    def _forward_completion(
        self,
        model: Any,
        teacher_inputs: dict[str, Any],
        position_ids: torch.Tensor,
        prompt_length: int,
        completion_length: int,
        *,
        with_grad: bool,
        need_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        model_inputs = {
            name: value
            for name, value in teacher_inputs.items()
            # Positions have already been derived.  Keeping a per-video timing
            # vector here is unnecessary and may have a batch-shape ambiguity.
            if name != "second_per_grid_ts"
        }
        model_inputs["position_ids"] = position_ids
        target_ids = teacher_inputs["input_ids"][:, prompt_length : prompt_length + completion_length]

        def evaluate() -> tuple[torch.Tensor, torch.Tensor | None]:
            output = model(**model_inputs, use_cache=False)
            logits = output.logits[
                :, prompt_length - 1 : prompt_length - 1 + completion_length, :
            ]
            if logits.shape[:2] != target_ids.shape:
                raise RuntimeError(
                    "teacher-forced completion logits do not align with sampled token IDs: "
                    f"logits={tuple(logits.shape)}, target={tuple(target_ids.shape)}"
                )
            logps = target_token_logps(logits, target_ids)
            entropy = None
            if need_entropy:
                with torch.no_grad():
                    entropy = compute_token_entropy(logits.detach().float())
            return logps, entropy

        if with_grad:
            return evaluate()
        with torch.no_grad():
            return evaluate()

    @staticmethod
    def _assert_same_layout(
        original: dict[str, Any], candidate: dict[str, Any], label: str
    ) -> None:
        for name in (
            "input_ids",
            "attention_mask",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        ):
            left = original.get(name)
            right = candidate.get(name)
            if not values_match(left, right):
                raise RuntimeError(f"{label} changed multimodal prompt layout field {name}")

    def _shuffled_prompt_inputs(
        self,
        prompts_text: list[str],
        image_inputs: list[Any] | None,
        video_inputs: list[torch.Tensor],
        video_kwargs: dict[str, Any],
        permutation: list[int],
        original: dict[str, Any],
    ) -> dict[str, Any]:
        frame_tensor = video_inputs[0]
        shuffled = [frame_tensor[torch.tensor(permutation, device=frame_tensor.device)]]
        encoded = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=shuffled,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **video_kwargs,
        )
        candidate = {
            name: self._to_device(value, self.accelerator.device)
            for name, value in dict(encoded).items()
        }
        self._assert_same_layout(original, candidate, "temporal counterfactual")
        return candidate

    def _record_selected_tokens(
        self,
        inputs: list[dict[str, Any]],
        completion_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        entropy: torch.Tensor,
        visual_delta: torch.Tensor | None,
        temporal_delta: torch.Tensor | None,
        entropy_mask: torch.Tensor | None,
        visual_mask: torch.Tensor | None,
        temporal_mask: torch.Tensor | None,
        union_mask: torch.Tensor,
    ) -> None:
        if not self.output_selected_token or self.token_record_limit == 0:
            return
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        records: list[dict[str, Any]] = []
        for completion_index in range(completion_ids.shape[0]):
            valid_positions = torch.nonzero(
                valid_mask[completion_index], as_tuple=False
            ).flatten().tolist()
            decoded_pieces: dict[int, str] = {}
            offsets: dict[int, tuple[int, int]] = {}
            text_parts: list[str] = []
            cursor = 0
            for candidate_position in valid_positions:
                candidate_token_id = int(
                    completion_ids[completion_index, candidate_position].item()
                )
                try:
                    candidate_piece = tokenizer.decode(
                        [candidate_token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                except TypeError:
                    candidate_piece = tokenizer.decode([candidate_token_id])
                decoded_pieces[candidate_position] = candidate_piece
                offsets[candidate_position] = (cursor, cursor + len(candidate_piece))
                text_parts.append(candidate_piece)
                cursor += len(candidate_piece)
            completion_text = "".join(text_parts)
            think_start = completion_text.find("<think>")
            think_end = completion_text.find("</think>")
            if think_end < 0:
                think_end = len(completion_text)
            selected = torch.nonzero(union_mask[completion_index], as_tuple=False).flatten().tolist()
            for position in selected[: self.token_record_limit]:
                token_id = int(completion_ids[completion_index, position].item())
                piece = decoded_pieces.get(position, "")
                start, end = offsets.get(position, (0, 0))
                records.append(
                    {
                        "global_step_before_update": int(self.state.global_step),
                        "rank": self._local_rank,
                        "problem_id": inputs[0].get("problem_id"),
                        "media_path": inputs[0].get("path"),
                        "completion_index": completion_index,
                        "position": position,
                        "token_id": token_id,
                        "piece": piece,
                        "completion_text": completion_text,
                        "token_context": completion_text[
                            max(0, start - 80) : min(len(completion_text), end + 80)
                        ],
                        "in_think": (
                            think_start >= 0
                            and start >= think_start + len("<think>")
                            and end <= think_end
                        ),
                        "valid": bool(valid_mask[completion_index, position].item()),
                        "entropy": float(entropy[completion_index, position].item()),
                        "visual_delta": (
                            float(visual_delta[completion_index, position].item())
                            if visual_delta is not None
                            else None
                        ),
                        "temporal_delta": (
                            float(temporal_delta[completion_index, position].item())
                            if temporal_delta is not None
                            else None
                        ),
                        "entropy_selected": bool(entropy_mask[completion_index, position].item()) if entropy_mask is not None else False,
                        "visual_selected": bool(visual_mask[completion_index, position].item()) if visual_mask is not None else False,
                        "temporal_selected": bool(temporal_mask[completion_index, position].item()) if temporal_mask is not None else False,
                        "union_selected": True,
                        "selection_mode": self.selection_mode,
                        "selection_scope": self.selection_scope,
                    }
                )
        if not records:
            return
        self._token_record_path.parent.mkdir(parents=True, exist_ok=True)
        with self._token_record_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _distributed_mean(self, values: torch.Tensor) -> float:
        gathered = self.accelerator.gather_for_metrics(values.detach().float())
        return float(gathered.mean().item())

    def compute_loss(self, model: Any, inputs: list[dict[str, Any]], return_outputs: bool = False, num_items_in_batch: Any = None) -> torch.Tensor:
        if return_outputs:
            raise ValueError("VideoKTRGRPOTrainer does not support returning model outputs")
        del num_items_in_batch
        data_type = self._collective_homogeneous_data_type(inputs)
        self._rank_trace("modality_agreement", data_type=data_type)
        prompts = [example["prompt"] for example in inputs]
        (
            prompts_text,
            image_inputs,
            video_inputs,
            video_kwargs,
            prompt_inputs,
            prepared_data_type,
        ) = self._prepare_prompt(inputs)
        if prepared_data_type != data_type:
            raise RuntimeError(
                "dataset data_type changed between collective validation and prompt preparation"
            )

        synced_generation = self._synced_generation_enabled()
        if synced_generation and self.accelerator.is_main_process and not self._generation_sync_reported:
            print(
                "[ktr] generation uses synced_gpus=True to keep distributed ranks "
                "on the same ZeRO collective sequence",
                flush=True,
            )
            self._generation_sync_reported = True
        self._rank_trace("generate_start", synced_gpus=synced_generation)
        with unwrap_model_for_generation(
            model,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            prompt_completion_ids = generate_with_synced_gpus(
                unwrapped_model,
                prompt_inputs,
                self.generation_config,
                synced_generation,
            )
        self._rank_trace("generate_done", generated_shape=list(prompt_completion_ids.shape))
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        completion_ids = prompt_completion_ids[:, prompt_length:]
        valid_mask = completion_valid_mask(
            completion_ids,
            getattr(self.processing_class, "eos_token_id", None),
            getattr(self.processing_class, "pad_token_id", None),
        ).to(self.accelerator.device)
        if not bool(valid_mask.any().item()):
            raise RuntimeError("all generated completion tokens were padding/EOS; cannot compute GRPO")

        completion_length = int(completion_ids.shape[1])
        teacher_inputs = self._teacher_inputs(
            prompt_inputs, prompt_completion_ids, prompt_length, valid_mask
        )
        position_ids = self._original_position_ids(model, teacher_inputs)
        self._rank_trace("policy_start", completion_length=completion_length)
        policy_logps, entropy = self._forward_completion(
            model,
            teacher_inputs,
            position_ids,
            prompt_length,
            completion_length,
            with_grad=True,
            need_entropy=self.video_ktr,
        )
        self._rank_trace("policy_done")

        entropy_mask: torch.Tensor | None = None
        visual_mask: torch.Tensor | None = None
        temporal_mask: torch.Tensor | None = None
        visual_delta: torch.Tensor | None = None
        temporal_delta: torch.Tensor | None = None
        union_mask = valid_mask

        if self.video_ktr:
            assert entropy is not None
            rope_model = self._unwrap_rope_model(model)
            image_token_id = int(getattr(rope_model.config, "image_token_id", 151655))
            video_token_id = int(getattr(rope_model.config, "video_token_id", 151656))
            visual_teacher = {
                name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                for name, value in teacher_inputs.items()
            }
            visual_positions = prompt_completion_ids.eq(image_token_id) | prompt_completion_ids.eq(video_token_id)
            masked_count = int(visual_positions.sum().item())
            if masked_count == 0:
                raise RuntimeError("no visual placeholder token found; cannot calculate visual attribution")
            visual_teacher["attention_mask"][visual_positions] = 0
            visual_logps, _ = self._forward_completion(
                model,
                visual_teacher,
                position_ids,
                prompt_length,
                completion_length,
                with_grad=False,
            )
            self._rank_trace("visual_done")
            delta_mode = "absolute" if self.selection_mode == "paper" else "signed"
            visual_delta = compute_logp_delta(policy_logps.detach(), visual_logps, mode=delta_mode)
            entropy_mask = select_top_ratio(
                entropy,
                valid_mask,
                self.entropy_ratio,
                mode=self.selection_mode,
                scope=self.selection_scope,
            )
            visual_mask = select_top_ratio(
                visual_delta,
                valid_mask,
                self.visual_ratio,
                mode=self.selection_mode,
                scope=self.selection_scope,
            )

            # A zero image temporal mask preserves E/V-only selection for
            # images, while keeping the metrics collective sequence identical
            # if a future caller bypasses the homogeneous sampler.  The
            # sampler assertion above still rejects that unsupported mixed
            # configuration before any model forward.
            temporal_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            masks: list[torch.Tensor] = [entropy_mask, visual_mask]
            if data_type == "video":
                if not video_inputs:
                    raise RuntimeError("video record has no decoded video tensor")
                frame_count = int(video_inputs[0].shape[0])
                permutations = nonidentity_frame_permutations(
                    frame_count,
                    self.temporal_permutations,
                    self.temporal_seed + self._local_rank + int(self.state.global_step) * 1009,
                    include_reverse=self.temporal_include_reverse,
                )
                accumulated = torch.zeros_like(policy_logps.detach())
                for permutation in permutations:
                    shuffled_prompt = self._shuffled_prompt_inputs(
                        prompts_text,
                        image_inputs,
                        video_inputs,
                        video_kwargs,
                        permutation,
                        prompt_inputs,
                    )
                    shuffled_teacher = self._teacher_inputs(
                        shuffled_prompt,
                        prompt_completion_ids,
                        prompt_length,
                        valid_mask,
                    )
                    temporal_logps, _ = self._forward_completion(
                        model,
                        shuffled_teacher,
                        position_ids,
                        prompt_length,
                        completion_length,
                        with_grad=False,
                    )
                    accumulated.add_(
                        compute_logp_delta(policy_logps.detach(), temporal_logps, mode=delta_mode)
                    )
                    self._rank_trace("temporal_done", permutation=permutation)
                temporal_delta = accumulated.div_(len(permutations))
                temporal_mask = select_top_ratio(
                    temporal_delta,
                    valid_mask,
                    self.temporal_ratio,
                    mode=self.selection_mode,
                    scope=self.selection_scope,
                )
            masks.append(temporal_mask)
            union_mask = union_masks(*masks, valid_mask=valid_mask)
            if not bool(union_mask.any().item()):
                raise RuntimeError("KTR E/V/T union is empty; no token would receive GRPO or KL")
            self._record_selected_tokens(
                inputs,
                completion_ids,
                valid_mask,
                entropy,
                visual_delta,
                temporal_delta,
                entropy_mask,
                visual_mask,
                temporal_mask,
                union_mask,
            )
            if self.accelerator.is_main_process:
                print(
                    "[ktr] "
                    f"protocol={self.selection_mode}/{self.selection_scope} "
                    f"E={int(entropy_mask.sum())} V={int(visual_mask.sum())} "
                    f"T={int(temporal_mask.sum())} "
                    f"U={int(union_mask.sum())} masked_visual={masked_count}",
                    flush=True,
                )

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_logps, _ = self._forward_completion(
                    self.ref_model,
                    teacher_inputs,
                    position_ids,
                    prompt_length,
                    completion_length,
                    with_grad=False,
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_logps, _ = self._forward_completion(
                        model,
                        teacher_inputs,
                        position_ids,
                        prompt_length,
                        completion_length,
                        with_grad=False,
                    )
        self._rank_trace("ref_done")

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        repeated_prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(
            len(repeated_prompts), len(self.reward_funcs), device=self.accelerator.device
        )
        for reward_index, reward_func in enumerate(self.reward_funcs):
            reward_kwargs = {
                key: []
                for key in inputs[0]
                # The collator's CPU payload is transport state, not dataset
                # metadata; keep reward function inputs unchanged.
                if key not in {"prompt", "completion", "_video_ktr_preprocessed_media"}
            }
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            reward_values = reward_func(
                prompts=repeated_prompts, completions=completions, **reward_kwargs
            )
            rewards_per_func[:, reward_index] = torch.as_tensor(
                reward_values, dtype=torch.float32, device=self.accelerator.device
            )
        rewards = rewards_per_func.sum(dim=1)

        if self.len_control:
            correct = rewards_per_func[:, 0] > 0.1
            # Length control is a reward on the generated completion, not on
            # the subset chosen for the KTR loss.  Using ``union_mask`` here
            # would make the KTR variant ineligible whenever the union has
            # fewer than 320 tokens, while a baseline with the same generated
            # completion could receive the bonus.  Keep reward/advantage
            # semantics identical across the two variants.
            completion_lengths = valid_mask.sum(dim=1)
            if int(correct.sum().item()) > 1:
                eligible = correct & completion_lengths.ge(320) & completion_lengths.le(512)
                rewards = rewards + eligible.to(rewards.dtype) * 0.2

        grouped_mean = rewards.view(-1, self.num_generations).mean(dim=1).repeat_interleave(self.num_generations)
        grouped_std = rewards.view(-1, self.num_generations).std(dim=1).repeat_interleave(self.num_generations)
        advantages = (rewards - grouped_mean) / (grouped_std + 1e-4)
        x_clamped = torch.clamp(ref_logps - policy_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        per_token_loss = -(torch.exp(policy_logps - policy_logps.detach()) * advantages.unsqueeze(1) - self.beta * per_token_kl)
        loss_mask = union_mask.to(per_token_loss.dtype)
        sequence_losses = (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)
        loss = sequence_losses.mean()

        completion_length_metric = self._distributed_mean(valid_mask.sum(dim=1))
        self._metrics["completion_length"].append(completion_length_metric)
        latest_step_metrics: dict[str, float] = {
            "completion_length": completion_length_metric,
        }
        for reward_index, reward_func in enumerate(self.reward_funcs):
            reward_name = getattr(reward_func, "__name__", reward_func.__class__.__name__)
            reward_value = self._distributed_mean(rewards_per_func[:, reward_index])
            self._metrics[f"rewards/{reward_name}"].append(reward_value)
            latest_step_metrics[f"rewards/{reward_name}"] = reward_value
        gathered_rewards = self.accelerator.gather_for_metrics(rewards.detach())
        reward_groups = gathered_rewards.view(-1, self.num_generations)
        all_wrong = float((reward_groups <= 1).all(dim=1).float().mean().item())
        all_correct = float((reward_groups >= 2).all(dim=1).float().mean().item())
        reward_mean = float(gathered_rewards.float().mean().item())
        reward_std = self._distributed_mean(grouped_std.detach())
        kl_value = self._distributed_mean(
            (per_token_kl.detach() * loss_mask).sum(dim=1)
            / loss_mask.sum(dim=1).clamp_min(1)
        )
        self._metrics["all_wrong"].append(all_wrong)
        self._metrics["all_correct"].append(all_correct)
        self._metrics["reward"].append(reward_mean)
        self._metrics["reward_std"].append(reward_std)
        self._metrics["kl"].append(kl_value)
        latest_step_metrics.update(
            {
                "all_wrong": all_wrong,
                "all_correct": all_correct,
                "reward": reward_mean,
                "reward_std": reward_std,
                "kl": kl_value,
            }
        )
        if self.video_ktr:
            assert (
                entropy_mask is not None
                and visual_mask is not None
                and temporal_mask is not None
            )
            ktr_metrics = {
                "ktr/entropy_tokens": self._distributed_mean(entropy_mask.sum(dim=1)),
                "ktr/visual_tokens": self._distributed_mean(visual_mask.sum(dim=1)),
                "ktr/temporal_tokens": self._distributed_mean(temporal_mask.sum(dim=1)),
                "ktr/union_tokens": self._distributed_mean(union_mask.sum(dim=1)),
                "ktr/update_ratio": self._distributed_mean(
                    union_mask.sum(dim=1).float() / valid_mask.sum(dim=1).clamp_min(1)
                ),
            }
            for metric_name, metric_value in ktr_metrics.items():
                self._metrics[metric_name].append(metric_value)
            latest_step_metrics.update(ktr_metrics)
        self.latest_step_metrics = latest_step_metrics
        self._rank_trace("metrics_done")
        return loss
