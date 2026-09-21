#!/usr/bin/env python3
"""Run the non-GRPO Video-KTR key-token attribution smoke test.

The program deliberately performs inference only:

1. generate a CoT for one video/question pair;
2. teacher-force that same completion on the original input;
3. counterfactually mask visual-token attention and shuffle video frames;
4. select high-entropy, visual-sensitive, temporal-sensitive tokens, and
   serialize human-readable evidence.

It does not instantiate a trainer, compute a loss, or modify model weights.
Use ``src/scripts/run_key_token_smoke.sh`` on a GPU node for the supported
offline invocation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from key_token_attribution import (
    attribute_key_tokens,
    compute_logp_delta,
    target_token_logps,
)
from qwen_vl_utils import process_vision_info


QUESTION_TEMPLATE = (
    "{question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', "
    "'wait', 'Hmm', 'oh, I see', 'let's break it down', or other natural "
    "language thought expressions. It is encouraged to include self-reflection "
    "or verification in the reasoning process. Provide your detailed reasoning "
    "between the <think> and </think> tags, and then give your final answer "
    "between the <answer> and </answer> tags. Please provide your text answer "
    "within the <answer> </answer> tags."
)

DEFAULT_QUESTION = "Which move motion in the video lose the system energy?"


def progress(stage: str, message: str) -> None:
    """Print a line suitable for a live terminal log."""

    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{stamp}] [{stage}] {message}", flush=True)


@contextlib.contextmanager
def timed_stage(stage: str, heartbeat_seconds: int) -> Iterator[None]:
    """Emit start/end plus a heartbeat while a long GPU step is running."""

    started = time.monotonic()
    finished = threading.Event()

    def heartbeat() -> None:
        while not finished.wait(heartbeat_seconds):
            elapsed = int(time.monotonic() - started)
            progress(stage, f"still running ({elapsed}s elapsed)")

    progress(stage, "started")
    thread = threading.Thread(target=heartbeat, name=f"heartbeat-{stage}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        finished.set()
        thread.join(timeout=max(1, heartbeat_seconds + 1))
        elapsed = time.monotonic() - started
        progress(stage, f"finished ({elapsed:.1f}s)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def values_match(left: Any, right: Any) -> bool:
    """Compare processor metadata without treating tensor equality as a scalar."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(values_match(first, second) for first, second in zip(left, right))
        )
    return bool(left == right)


def assert_prompt_layout_matches(
    original: dict[str, Any], candidate: dict[str, Any], *, counterfactual: str
) -> None:
    """Ensure a video perturbation did not change token/mRoPE layout metadata.

    Pixel values are deliberately omitted: they are the thing a visual or
    temporal counterfactual changes.  These fields must remain identical so a
    log-probability delta cannot silently include a tokenization, grid, mask,
    or timestamp-layout change.
    """

    for field in (
        "input_ids",
        "attention_mask",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    ):
        original_value = original.get(field)
        candidate_value = candidate.get(field)
        if not values_match(original_value, candidate_value):
            raise RuntimeError(f"{counterfactual} changed prompt layout field: {field}")


def completion_valid_mask(completion_ids: torch.Tensor, eos_token_id: int | None, pad_token_id: int | None) -> torch.Tensor:
    """Exclude padding, EOS, and anything after the first EOS from selection."""

    valid = torch.ones_like(completion_ids, dtype=torch.bool)
    if pad_token_id is not None:
        valid &= completion_ids.ne(pad_token_id)
    if eos_token_id is not None:
        eos = completion_ids.eq(eos_token_id)
        # The EOS token itself and all later padding/positions are excluded.
        valid &= eos.cumsum(dim=-1).eq(0)
    return valid


def build_teacher_forcing_inputs(
    processed_prompt: dict[str, Any],
    generated_ids: torch.Tensor,
    prompt_length: int,
    completion_mask: torch.Tensor,
) -> dict[str, Any]:
    """Append a generated completion and preserve all Qwen multimodal inputs."""

    if generated_ids.shape[0] != 1:
        raise ValueError("the standalone smoke selector currently expects batch size 1")
    result: dict[str, Any] = {}
    for name, value in processed_prompt.items():
        if name == "input_ids":
            result[name] = generated_ids
        elif name == "attention_mask":
            prefix = value[:, :prompt_length].to(dtype=torch.long)
            result[name] = torch.cat((prefix, completion_mask.to(dtype=torch.long)), dim=1)
        else:
            result[name] = value
    if "attention_mask" not in result:
        result["attention_mask"] = torch.cat(
            (
                torch.ones_like(generated_ids[:, :prompt_length], dtype=torch.long),
                completion_mask.to(dtype=torch.long),
            ),
            dim=1,
        )
    return result


def forward_completion(
    model: Qwen2_5_VLForConditionalGeneration,
    processed_prompt: dict[str, Any],
    generated_ids: torch.Tensor,
    prompt_length: int,
    completion_mask: torch.Tensor,
    *,
    retain_logits: bool,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Teacher-force a completion and return logits/logp aligned to its tokens.

    ``position_ids`` is deliberately injectable: a visual-attention ablation
    must use the original, unmasked mRoPE layout rather than asking Qwen to
    rebuild positions from a modified attention mask.
    """

    model_inputs = build_teacher_forcing_inputs(
        processed_prompt, generated_ids, prompt_length, completion_mask
    )
    if position_ids is not None:
        model_inputs["position_ids"] = position_ids.clone()
    with torch.inference_mode():
        output = model(**model_inputs, use_cache=False)
        completion_length = completion_mask.shape[1]
        logits = output.logits[:, prompt_length - 1 : prompt_length - 1 + completion_length, :]
        target_ids = generated_ids[:, prompt_length : prompt_length + completion_length]
        logps = target_token_logps(logits.float(), target_ids)
        retained = logits.float().cpu() if retain_logits else None
    return retained, logps.float().cpu()


def original_position_ids(
    model: Qwen2_5_VLForConditionalGeneration,
    teacher_inputs: dict[str, Any],
) -> torch.Tensor:
    """Build the one mRoPE layout shared by all counterfactual forwards.

    Qwen2.5-VL's native ``get_rope_index`` filters ``input_ids`` with the
    attention mask.  If a visual ablation zeros the video placeholders and
    lets the model recompute positions, every later text token changes
    position too.  Computing it once from the original unmasked input avoids
    that confound while preserving the model's normal layout exactly.
    """

    with torch.inference_mode():
        position_ids, _ = model.get_rope_index(
            input_ids=teacher_inputs["input_ids"],
            image_grid_thw=teacher_inputs.get("image_grid_thw"),
            video_grid_thw=teacher_inputs.get("video_grid_thw"),
            second_per_grid_ts=teacher_inputs.get("second_per_grid_ts"),
            attention_mask=teacher_inputs["attention_mask"],
        )
    if position_ids.ndim != 3 or position_ids.shape[1:] != teacher_inputs["input_ids"].shape:
        raise RuntimeError(
            "Qwen returned position_ids with unexpected shape "
            f"{tuple(position_ids.shape)} for input_ids {tuple(teacher_inputs['input_ids'].shape)}"
        )
    return position_ids


def nonidentity_frame_permutations(
    frame_count: int,
    count: int,
    seed: int,
    *,
    include_reverse: bool,
) -> list[list[int]]:
    """Return reproducible, distinct, non-identity frame permutations."""

    if frame_count < 2:
        raise ValueError("temporal attribution needs at least two frames")
    if count < 1:
        raise ValueError("temporal permutation count must be at least one")
    available = math.factorial(frame_count) - 1
    if count > available:
        raise ValueError(
            f"requested {count} distinct non-identity permutations but only {available} exist for {frame_count} frames"
        )

    identity = tuple(range(frame_count))
    permutations: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    if include_reverse:
        reverse = tuple(reversed(identity))
        if reverse != identity:
            permutations.append(list(reverse))
            seen.add(reverse)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    while len(permutations) < count:
        candidate = tuple(torch.randperm(frame_count, generator=generator).tolist())
        if candidate == identity or candidate in seen:
            continue
        permutations.append(list(candidate))
        seen.add(candidate)
    return permutations


def visual_masked_prompt(
    processed_prompt: dict[str, Any],
    generated_ids: torch.Tensor,
    prompt_length: int,
    completion_mask: torch.Tensor,
    image_token_id: int,
    video_token_id: int,
) -> dict[str, Any]:
    """Mask attention to image/video placeholder tokens, matching the trainer idea."""

    result = build_teacher_forcing_inputs(
        processed_prompt, generated_ids, prompt_length, completion_mask
    )
    attention_mask = result["attention_mask"].clone()
    visual_positions = generated_ids.eq(image_token_id) | generated_ids.eq(video_token_id)
    count = int(visual_positions.sum().item())
    if count == 0:
        raise RuntimeError(
            "no image/video placeholder token occurred in the prompt; cannot perform visual perturbation"
        )
    attention_mask[visual_positions] = 0
    result["attention_mask"] = attention_mask
    result["_masked_visual_token_count"] = count
    return result


def prepare_video_prompt(
    processor: Any,
    video_path: Path,
    question: str,
    nframes: int,
    max_pixels: int,
    device: torch.device,
    video_override: torch.Tensor | None = None,
) -> tuple[dict[str, Any], str, torch.Tensor, dict[str, Any]]:
    """Construct Qwen processor inputs, optionally with a supplied frame order."""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "nframes": nframes,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": QUESTION_TEMPLATE.format(question=question)},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    if not video_inputs:
        raise RuntimeError("the video processor returned no video tensor")
    video_tensor = video_override if video_override is not None else video_inputs[0]
    encoded = processor(
        text=[prompt],
        images=image_inputs,
        videos=[video_tensor],
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    processed = {name: tensor_to_device(value, device) for name, value in encoded.items()}
    return processed, prompt, video_tensor, video_kwargs


def decode_raw_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def decode_piece(tokenizer: Any, token_id: int) -> str:
    """Return a table-safe individual token piece while preserving raw CoT elsewhere."""

    return decode_raw_piece(tokenizer, token_id).replace("\n", "\\n")


def completion_text_and_offsets(tokenizer: Any, token_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """Decode per-token pieces and use their concatenation as stable display context."""

    pieces = [decode_raw_piece(tokenizer, token_id) for token_id in token_ids]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return "".join(pieces), offsets


def in_think_tag(text: str, offset: tuple[int, int]) -> bool:
    start = text.find("<think>")
    end = text.find("</think>")
    if start < 0:
        return False
    if end < 0:
        end = len(text)
    return offset[0] >= start + len("<think>") and offset[1] <= end


def excerpt(text: str, offset: tuple[int, int], radius: int = 55) -> str:
    left = max(0, offset[0] - radius)
    right = min(len(text), offset[1] + radius)
    return text[left:right].replace("\n", "\\n")


def finite_score(value: torch.Tensor | None, row: int, column: int) -> float | None:
    if value is None:
        return None
    return float(value[row, column].item())


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


@dataclass
class RunMetadata:
    created_at_utc: str
    case_id: str
    question: str
    model_path: str
    model_revision: str
    video_path: str
    video_sha256: str
    transformers_version: str
    transformers_source_commit: str
    torch_version: str
    cuda_version: str | None
    device: str
    physical_gpu_id: str | None
    seed: int
    nframes: int
    max_pixels: int
    max_new_tokens: int
    min_new_tokens: int
    generated_token_count: int
    generation_stop_reason: str
    temperature: float
    top_p: float
    selection_mode: str
    selection_scope: str
    entropy_ratio: float
    visual_ratio: float
    temporal_ratio: float
    frame_permutation: list[int]
    temporal_permutation_seed: int
    temporal_permutations: list[list[int]]
    temporal_probe_statistics: list[dict[str, float | int | list[int]]]
    masked_visual_token_count: int
    visual_perturbation: str
    shared_original_position_ids: bool
    control_tolerance: float
    control_metrics: dict[str, float]
    training_or_weight_update: bool
    repository_commit: str | None
    repository_dirty: bool | None
    selector_sha256: str
    attribution_sha256: str
    launcher_sha256: str | None
    run_config_sha256: str | None


def write_report(
    output_dir: Path,
    metadata: RunMetadata,
    completion: str,
    records: list[dict[str, Any]],
    counts: dict[str, int],
) -> None:
    """Write both lossless JSON and a compact, immediately readable Markdown report."""

    payload = {
        "metadata": asdict(metadata),
        "completion": completion,
        "counts": counts,
        "tokens": records,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # One completion is one JSONL record in the smoke run.  Keeping JSONL in
    # addition to the pretty JSON makes later multi-video aggregation trivial.
    (output_dir / "result.jsonl").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Video-KTR 非 GRPO 关键 Token Smoke Test",
        "",
        "本报告只执行生成与 teacher forcing 归因；`training_or_weight_update=false`，未运行 GRPO、loss、backward、optimizer 或权重保存。",
        "",
        "## 运行元数据",
        "",
        "```json",
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Counterfactual controls",
        "",
        "- visual perturbation: `%s`" % metadata.visual_perturbation,
        "- shared original mRoPE positions: `%s`" % metadata.shared_original_position_ids,
        "- temporal permutations: `%d`" % len(metadata.temporal_permutations),
        "- generation stop reason: `%s`" % metadata.generation_stop_reason,
    ]
    if metadata.control_metrics:
        lines.extend(
            [
                "- control metrics:",
                "```json",
                json.dumps(metadata.control_metrics, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    lines.extend(
        [
            "",
        "## 生成的 CoT / 回答",
        "",
        "```text",
        completion,
        "```",
        "",
        "## 选择统计",
        "",
        "| 有效 completion token | 高熵 E | 视觉 V | 时序 T | 并集 U |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {counts['valid']} | {counts['entropy']} | {counts['visual']} | {counts['temporal']} | {counts['union']} |",
        "",
        "选择模式：`%s` / `%s`；E、V、T 分别取有效 token 的 %.0f%%、%.0f%%、%.0f%%。"
        % (
            metadata.selection_mode,
            metadata.selection_scope,
            metadata.entropy_ratio * 100,
            metadata.visual_ratio * 100,
            metadata.temporal_ratio * 100,
        ),
            "",
        ]
    )

    categories = [
        ("高熵 token（E）", "entropy_selected", "entropy"),
        ("视觉敏感 token（V）", "visual_selected", "visual_delta"),
        ("时序敏感 token（T）", "temporal_selected", "temporal_delta"),
    ]
    for title, flag, score_name in categories:
        selected = [record for record in records if record[flag]]
        selected.sort(key=lambda item: item[score_name], reverse=True)
        lines.extend(
            [
                f"## {title}",
                "",
                "| 位置 | token | 分数 | think 中 | 上下文 | 标签 |",
                "| ---: | --- | ---: | :---: | --- | --- |",
            ]
        )
        for record in selected[:8]:
            labels = "/".join(record["labels"])
            lines.append(
                "| {position} | `{piece}` | {score:.6f} | {in_think} | {context} | {labels} |".format(
                    position=record["position"],
                    piece=markdown_escape(record["piece"]),
                    score=record[score_name],
                    in_think="是" if record["in_think"] else "否",
                    context=markdown_escape(record["context"]),
                    labels=labels,
                )
            )
        if not selected:
            lines.append("| — | （无） | — | — | — | — |")
        lines.append("")

    lines.extend(
        [
            "## 并集 U（前 12 个位置）",
            "",
            "| 位置 | token | E | V | T | 上下文 |",
            "| ---: | --- | :---: | :---: | :---: | --- |",
        ]
    )
    union_records = [record for record in records if record["union_selected"]]
    for record in union_records[:12]:
        lines.append(
            "| {position} | `{piece}` | {e} | {v} | {t} | {context} |".format(
                position=record["position"],
                piece=markdown_escape(record["piece"]),
                e="✓" if record["entropy_selected"] else "",
                v="✓" if record["visual_selected"] else "",
                t="✓" if record["temporal_selected"] else "",
                context=markdown_escape(record["context"]),
            )
        )
    if not union_records:
        lines.append("| — | （无） | — | — | — | — |")
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    # Written last: launchers may safely treat this marker as a completed,
    # resumable case rather than mistaking a partially written result for one.
    status = {
        "schema_version": 1,
        "status": "success",
        "case_id": metadata.case_id,
        "result_sha256": sha256_file(output_dir / "result.json"),
        "run_config_sha256": metadata.run_config_sha256,
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="local checkpoint directory; launchers pass MODEL_PATH explicitly",
    )
    parser.add_argument(
        "--video-path", type=Path, default=Path("src/example_video/video1.mp4")
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--case-id",
        default="single-video",
        help="stable manifest identifier used by batch validation and reports",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nframes", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entropy-ratio", type=float, default=0.2)
    parser.add_argument("--visual-ratio", type=float, default=0.2)
    parser.add_argument("--temporal-ratio", type=float, default=0.2)
    parser.add_argument(
        "--temporal-seed",
        type=int,
        default=None,
        help="random source for frame permutations; defaults to seed + 1",
    )
    parser.add_argument(
        "--temporal-permutations",
        type=int,
        default=1,
        help="number of distinct non-identity frame shuffles to average",
    )
    parser.add_argument(
        "--temporal-include-reverse",
        action="store_true",
        help="make reverse-order one of the temporal probes",
    )
    parser.add_argument(
        "--visual-perturbation",
        choices=("attention_mask_fixed_position",),
        default="attention_mask_fixed_position",
        help="visual counterfactual protocol; fixed original mRoPE is required",
    )
    parser.add_argument(
        "--verify-controls",
        action="store_true",
        help="run auto-position, legacy visual-position, and identity-video sanity controls",
    )
    parser.add_argument(
        "--control-tolerance",
        type=float,
        default=1e-4,
        help="maximum valid-token logp deviation accepted by deterministic controls",
    )
    parser.add_argument("--selection-mode", choices=("paper", "repo"), default="paper")
    parser.add_argument("--selection-scope", choices=("per_completion", "batch"), default=None)
    parser.add_argument(
        "--model-revision",
        default="f71f0f1e22c015007fccd080eef87824fe292a10",
        help="resolved revision of the local Video-R1 checkpoint",
    )
    parser.add_argument(
        "--transformers-source-commit",
        default="336dc69d63d56f232a183a3e7f52790429b871ef",
        help="source commit used to build the compatible Transformers wheel",
    )
    parser.add_argument(
        "--launcher-path",
        type=Path,
        default=None,
        help="optional shell launcher path whose SHA256 is recorded in the artifact",
    )
    parser.add_argument(
        "--run-config-sha256",
        default=None,
        help="optional launcher-computed fingerprint required for safe batch resume",
    )
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.18)
    parser.add_argument(
        "--physical-gpu-id",
        default=None,
        help="physical GPU identifier recorded by the shell launcher for auditability",
    )
    parser.add_argument("--heartbeat-seconds", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.nframes < 2:
        raise ValueError("--nframes must be at least 2 for temporal perturbation")
    if args.min_new_tokens > args.max_new_tokens:
        raise ValueError("--min-new-tokens cannot exceed --max-new-tokens")
    if args.heartbeat_seconds < 1:
        raise ValueError("--heartbeat-seconds must be at least 1")
    if args.temporal_permutations < 1:
        raise ValueError("--temporal-permutations must be at least 1")
    if args.control_tolerance < 0:
        raise ValueError("--control-tolerance must be non-negative")
    if args.run_config_sha256 is not None and (
        len(args.run_config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.run_config_sha256.lower())
    ):
        raise ValueError("--run-config-sha256 must be a 64-character hexadecimal SHA256 digest")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0, 1]")
    for name in ("entropy_ratio", "visual_ratio", "temporal_ratio"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")

    model_path = args.model_path.resolve()
    video_path = args.video_path.resolve()
    output_dir = args.output_dir.resolve()
    # The launcher creates the timestamped directory first so it can tee its
    # own preflight output into ``terminal.log``.  A direct Python invocation
    # may create it here instead.
    output_dir.mkdir(parents=True, exist_ok=True)
    progress("preflight", f"writing run artifacts to {output_dir}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"video file does not exist: {video_path}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this smoke test requires an available CUDA GPU")
    # PyTorch's allocator-limit API rejects the shorthand ``torch.device('cuda')``.
    # The launcher maps its selected physical card to logical device 0.
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    progress(
        "preflight",
        "device=%s (%s); PyTorch allocator cap=%.0f%%; free=%d MiB"
        % (
            device,
            torch.cuda.get_device_name(device),
            args.gpu_memory_fraction * 100,
            torch.cuda.mem_get_info(device)[0] // (1024 * 1024),
        ),
    )
    progress("preflight", "offline-only model loading is enabled; no training APIs are used")
    set_seed(args.seed)

    with timed_stage("model", args.heartbeat_seconds):
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        model.to(device)
        model.eval()

    with timed_stage("video", args.heartbeat_seconds):
        processed_prompt, prompt, video_tensor, _ = prepare_video_prompt(
            processor,
            video_path,
            args.question,
            args.nframes,
            args.max_pixels,
            device,
        )
    prompt_length = int(processed_prompt["input_ids"].shape[1])
    progress(
        "video",
        "prompt_tokens=%d; decoded_video_shape=%s; processor_keys=%s"
        % (
            prompt_length,
            tuple(video_tensor.shape),
            ", ".join(sorted(processed_prompt)),
        ),
    )

    with timed_stage("generate", args.heartbeat_seconds):
        with torch.inference_mode():
            generated_ids = model.generate(
                **processed_prompt,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                min_new_tokens=args.min_new_tokens,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
    completion_ids = generated_ids[:, prompt_length:]
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    valid_mask = completion_valid_mask(completion_ids, eos_token_id, pad_token_id)
    if not bool(valid_mask.any().item()):
        raise RuntimeError("generation contained no valid completion token")
    generation_stop_reason = (
        "eos_token"
        if eos_token_id is not None and bool(completion_ids.eq(eos_token_id).any().item())
        else "max_new_tokens"
    )
    progress(
        "generate",
        "generated=%d tokens; valid_for_attribution=%d tokens; stop_reason=%s"
        % (completion_ids.shape[1], int(valid_mask.sum().item()), generation_stop_reason),
    )

    original_teacher_inputs = build_teacher_forcing_inputs(
        processed_prompt, generated_ids, prompt_length, valid_mask
    )
    with timed_stage("position-reference", args.heartbeat_seconds):
        reference_position_ids = original_position_ids(model, original_teacher_inputs)
    progress(
        "position-reference",
        "using original unmasked mRoPE positions with shape=%s for every counterfactual"
        % (tuple(reference_position_ids.shape),),
    )

    with timed_stage("original-teacher-forcing", args.heartbeat_seconds):
        original_logits, original_logps = forward_completion(
            model,
            processed_prompt,
            generated_ids,
            prompt_length,
            valid_mask,
            retain_logits=True,
            position_ids=reference_position_ids,
        )
    if original_logits is None:
        raise AssertionError("original teacher-forcing logits were unexpectedly discarded")
    valid_mask_cpu = valid_mask.cpu()
    control_metrics: dict[str, float] = {}
    if args.verify_controls:
        with timed_stage("control-auto-position", args.heartbeat_seconds):
            _, automatic_position_logps = forward_completion(
                model,
                processed_prompt,
                generated_ids,
                prompt_length,
                valid_mask,
                retain_logits=False,
                position_ids=None,
            )
        auto_position_difference = float(
            (automatic_position_logps[valid_mask_cpu] - original_logps[valid_mask_cpu])
            .abs()
            .max()
            .item()
        )
        control_metrics["original_auto_vs_fixed_position_max_abs_logp"] = auto_position_difference
        if auto_position_difference > args.control_tolerance:
            raise RuntimeError(
                "fixed original mRoPE positions do not reproduce the native original forward: "
                f"max |Δlogp|={auto_position_difference:.6g} > {args.control_tolerance:.6g}"
            )

    image_token_id = int(getattr(model.config, "image_token_id", 151655))
    video_token_id = int(getattr(model.config, "video_token_id", 151656))
    masked_visual_count = 0
    with timed_stage("visual-counterfactual", args.heartbeat_seconds):
        masked_prompt = visual_masked_prompt(
            processed_prompt,
            generated_ids,
            prompt_length,
            valid_mask,
            image_token_id,
            video_token_id,
        )
        masked_visual_count = int(masked_prompt.pop("_masked_visual_token_count"))
        _, visual_logps = forward_completion(
            model,
            masked_prompt,
            generated_ids,
            prompt_length,
            valid_mask,
            retain_logits=False,
            position_ids=reference_position_ids,
        )
        if args.verify_controls:
            _, legacy_visual_logps = forward_completion(
                model,
                masked_prompt,
                generated_ids,
                prompt_length,
                valid_mask,
                retain_logits=False,
                position_ids=None,
            )
            control_metrics["legacy_visual_auto_position_vs_fixed_max_abs_logp"] = float(
                (legacy_visual_logps[valid_mask_cpu] - visual_logps[valid_mask_cpu])
                .abs()
                .max()
                .item()
            )
    progress("visual-counterfactual", f"masked {masked_visual_count} visual placeholder tokens")

    temporal_seed = args.temporal_seed if args.temporal_seed is not None else args.seed + 1
    frame_permutations = nonidentity_frame_permutations(
        video_tensor.shape[0],
        args.temporal_permutations,
        temporal_seed,
        include_reverse=args.temporal_include_reverse,
    )
    temporal_logps_permutation: list[torch.Tensor] = []
    temporal_probe_statistics: list[dict[str, float | int | list[int]]] = []
    for permutation_index, frame_permutation in enumerate(frame_permutations, start=1):
        shuffled_video = video_tensor[frame_permutation].contiguous()
        if torch.equal(video_tensor, shuffled_video):
            raise RuntimeError(
                f"frame permutation {frame_permutation} did not alter the decoded video tensor"
            )
        with timed_stage(
            f"temporal-counterfactual-{permutation_index}/{len(frame_permutations)}",
            args.heartbeat_seconds,
        ):
            shuffled_prompt, shuffled_rendered_prompt, _, _ = prepare_video_prompt(
                processor,
                video_path,
                args.question,
                args.nframes,
                args.max_pixels,
                device,
                video_override=shuffled_video,
            )
            if shuffled_rendered_prompt != prompt:
                raise RuntimeError("frame shuffling changed prompt text unexpectedly")
            assert_prompt_layout_matches(
                processed_prompt,
                shuffled_prompt,
                counterfactual="frame shuffling",
            )
            _, temporal_logps = forward_completion(
                model,
                shuffled_prompt,
                generated_ids,
                prompt_length,
                valid_mask,
                retain_logits=False,
                position_ids=reference_position_ids,
            )
        absolute_delta = compute_logp_delta(original_logps, temporal_logps, mode="absolute")
        temporal_logps_permutation.append(temporal_logps)
        temporal_probe_statistics.append(
            {
                "index": permutation_index,
                "permutation": frame_permutation,
                "mean_abs_logp_delta": float(absolute_delta[valid_mask_cpu].mean().item()),
                "max_abs_logp_delta": float(absolute_delta[valid_mask_cpu].max().item()),
            }
        )

    if args.verify_controls:
        with timed_stage("control-identity-video", args.heartbeat_seconds):
            identity_prompt, identity_rendered_prompt, _, _ = prepare_video_prompt(
                processor,
                video_path,
                args.question,
                args.nframes,
                args.max_pixels,
                device,
                video_override=video_tensor,
            )
            if identity_rendered_prompt != prompt:
                raise RuntimeError("identity-video control changed prompt text")
            assert_prompt_layout_matches(
                processed_prompt,
                identity_prompt,
                counterfactual="identity-video control",
            )
            _, identity_logps = forward_completion(
                model,
                identity_prompt,
                generated_ids,
                prompt_length,
                valid_mask,
                retain_logits=False,
                position_ids=reference_position_ids,
            )
        identity_difference = float(
            (identity_logps[valid_mask_cpu] - original_logps[valid_mask_cpu]).abs().max().item()
        )
        control_metrics["identity_video_max_abs_logp"] = identity_difference
        if identity_difference > args.control_tolerance:
            raise RuntimeError(
                "identity-video control differs from original: "
                f"max |Δlogp|={identity_difference:.6g} > {args.control_tolerance:.6g}"
            )

    delta_mode = "repo" if args.selection_mode == "repo" else "paper"
    temporal_delta_scores = torch.stack(
        [
            compute_logp_delta(original_logps, logps, mode=delta_mode)
            for logps in temporal_logps_permutation
        ],
        dim=0,
    ).mean(dim=0)
    temporal_logps_mean = torch.stack(temporal_logps_permutation, dim=0).mean(dim=0)

    with timed_stage("select", args.heartbeat_seconds):
        attribution = attribute_key_tokens(
            original_logits,
            original_logps,
            visual_logps,
            valid_mask.cpu(),
            temporal_delta_scores=temporal_delta_scores,
            entropy_ratio=args.entropy_ratio,
            visual_ratio=args.visual_ratio,
            temporal_ratio=args.temporal_ratio,
            mode=args.selection_mode,
            scope=args.selection_scope,
        )
    if attribution.temporal_mask is None or attribution.temporal_delta is None:
        raise AssertionError("video smoke test must produce temporal attribution")
    if not torch.isfinite(attribution.entropy[valid_mask.cpu()]).all():
        raise RuntimeError("non-finite entropy at a valid token")
    if not torch.isfinite(attribution.visual_delta[valid_mask.cpu()]).all():
        raise RuntimeError("non-finite visual delta at a valid token")
    if not torch.isfinite(attribution.temporal_delta[valid_mask.cpu()]).all():
        raise RuntimeError("non-finite temporal delta at a valid token")

    valid_ids = completion_ids[0, valid_mask[0]].detach().cpu().tolist()
    completion, offsets = completion_text_and_offsets(processor.tokenizer, valid_ids)
    entropy_mask = attribution.entropy_mask[0]
    visual_mask = attribution.visual_mask[0]
    temporal_mask = attribution.temporal_mask[0]
    union_mask = attribution.union_mask[0]
    records: list[dict[str, Any]] = []
    for position, token_id in enumerate(valid_ids):
        labels: list[str] = []
        if bool(entropy_mask[position]):
            labels.append("E")
        if bool(visual_mask[position]):
            labels.append("V")
        if bool(temporal_mask[position]):
            labels.append("T")
        if bool(union_mask[position]):
            labels.append("U")
        records.append(
            {
                "position": position,
                "token_id": int(token_id),
                "token": processor.tokenizer.convert_ids_to_tokens(int(token_id)),
                "piece": decode_piece(processor.tokenizer, int(token_id)),
                "offset": list(offsets[position]),
                "context": excerpt(completion, offsets[position]),
                "in_think": in_think_tag(completion, offsets[position]),
                "entropy": finite_score(attribution.entropy, 0, position),
                "original_logp": finite_score(original_logps, 0, position),
                "visual_logp": finite_score(visual_logps, 0, position),
                "temporal_logp_mean": finite_score(temporal_logps_mean, 0, position),
                "visual_delta": finite_score(attribution.visual_delta, 0, position),
                "temporal_delta": finite_score(attribution.temporal_delta, 0, position),
                "entropy_selected": bool(entropy_mask[position]),
                "visual_selected": bool(visual_mask[position]),
                "temporal_selected": bool(temporal_mask[position]),
                "union_selected": bool(union_mask[position]),
                "labels": labels,
            }
        )

    counts = {
        "valid": int(valid_mask.sum().item()),
        "entropy": int(attribution.entropy_mask.sum().item()),
        "visual": int(attribution.visual_mask.sum().item()),
        "temporal": int(attribution.temporal_mask.sum().item()),
        "union": int(attribution.union_mask.sum().item()),
    }
    if counts["valid"] <= 0 or counts["union"] <= 0:
        raise RuntimeError(f"selection unexpectedly produced no valid union token: {counts}")

    import transformers  # Kept here so the CLI can give preflight feedback first.

    try:
        repository_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        repository_commit = None

    try:
        repository_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        repository_dirty = None

    selector_path = Path(__file__).resolve()
    attribution_path = selector_path.with_name("key_token_attribution.py")
    launcher_path = args.launcher_path.resolve() if args.launcher_path is not None else None
    if launcher_path is not None and not launcher_path.is_file():
        raise FileNotFoundError(f"launcher path does not exist: {launcher_path}")

    metadata = RunMetadata(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        case_id=args.case_id,
        question=args.question,
        model_path=str(model_path),
        model_revision=args.model_revision,
        video_path=str(video_path),
        video_sha256=sha256_file(video_path),
        transformers_version=transformers.__version__,
        transformers_source_commit=args.transformers_source_commit,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device=str(device),
        physical_gpu_id=args.physical_gpu_id,
        seed=args.seed,
        nframes=args.nframes,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        generated_token_count=int(completion_ids.shape[1]),
        generation_stop_reason=generation_stop_reason,
        temperature=args.temperature,
        top_p=args.top_p,
        selection_mode=attribution.mode,
        selection_scope=attribution.scope,
        entropy_ratio=args.entropy_ratio,
        visual_ratio=args.visual_ratio,
        temporal_ratio=args.temporal_ratio,
        frame_permutation=frame_permutations[0],
        temporal_permutation_seed=temporal_seed,
        temporal_permutations=frame_permutations,
        temporal_probe_statistics=temporal_probe_statistics,
        masked_visual_token_count=masked_visual_count,
        visual_perturbation=args.visual_perturbation,
        shared_original_position_ids=True,
        control_tolerance=args.control_tolerance,
        control_metrics=control_metrics,
        training_or_weight_update=False,
        repository_commit=repository_commit,
        repository_dirty=repository_dirty,
        selector_sha256=sha256_file(selector_path),
        attribution_sha256=sha256_file(attribution_path),
        launcher_sha256=sha256_file(launcher_path) if launcher_path is not None else None,
        run_config_sha256=args.run_config_sha256,
    )
    with timed_stage("write-artifacts", args.heartbeat_seconds):
        write_report(output_dir, metadata, completion, records, counts)
    progress(
        "complete",
        "E=%d V=%d T=%d U=%d; report=%s"
        % (counts["entropy"], counts["visual"], counts["temporal"], counts["union"], output_dir / "report.md"),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        progress("failed", f"{type(error).__name__}: {error}")
        raise
