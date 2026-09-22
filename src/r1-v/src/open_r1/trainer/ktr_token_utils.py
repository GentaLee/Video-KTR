"""Self-contained tensor helpers used by the direct Video-KTR trainer.

The standalone attribution report lives under the repository-level ``src/``
directory.  A normal upstream launcher, however, imports ``open_r1`` with
only ``src/r1-v/src`` on ``sys.path``.  Keeping this small, dependency-free
copy inside the package makes the training entrypoint portable.  The unit
tests compare its deterministic outputs with ``key_token_attribution``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import Tensor


def requires_synced_generation(
    distributed_available: bool, distributed_initialized: bool, world_size: int
) -> bool:
    """Return whether generation must advance in lockstep across ranks.

    ZeRO-3 can issue parameter collectives from each generation forward.  If
    one rank stops generation at EOS while another continues, the ranks can
    enter a different collective sequence and hang.  ``generate`` supports
    ``synced_gpus`` specifically for this case.  Keeping the predicate pure
    makes the distributed boundary explicit and unit-testable.
    """

    if world_size < 1:
        raise ValueError("world_size must be at least one")
    return bool(distributed_available and distributed_initialized and world_size > 1)


def generate_with_synced_gpus(
    model: Any,
    prompt_inputs: dict[str, Any],
    generation_config: Any,
    synced_gpus: bool,
) -> Tensor:
    """Call ``generate`` with the explicit distributed synchronization flag.

    Keeping the small call boundary here lets the unit test assert that the
    flag reaches Transformers, rather than only testing the predicate that
    computes it.
    """

    return model.generate(
        **prompt_inputs,
        generation_config=generation_config,
        synced_gpus=bool(synced_gpus),
    )


def values_match(left: Any, right: Any) -> bool:
    """Compare processor metadata without reducing tensor equality to a scalar."""

    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(left, right)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(values_match(first, second) for first, second in zip(left, right))
        )
    return bool(left == right)


def completion_valid_mask(
    completion_ids: Tensor, eos_token_id: int | None, pad_token_id: int | None
) -> Tensor:
    """Exclude padding, EOS, and positions after the first EOS."""

    valid = torch.ones_like(completion_ids, dtype=torch.bool)
    if pad_token_id is not None:
        valid &= completion_ids.ne(pad_token_id)
    if eos_token_id is not None:
        valid &= completion_ids.eq(eos_token_id).cumsum(dim=-1).eq(0)
    return valid


def nonidentity_frame_permutations(
    frame_count: int, count: int, seed: int, *, include_reverse: bool
) -> list[list[int]]:
    """Return reproducible, distinct non-identity frame permutations."""

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


def compute_token_entropy(logits: Tensor) -> Tensor:
    """Return numerically stable Shannon entropy over the vocabulary."""

    if logits.ndim < 2 or not logits.is_floating_point():
        raise ValueError("logits must be a floating tensor with a vocabulary dimension")
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def target_token_logps(logits: Tensor, target_token_ids: Tensor) -> Tensor:
    """Extract aligned teacher-forced target-token log probabilities."""

    if logits.ndim < 2 or target_token_ids.shape != logits.shape[:-1]:
        raise ValueError("target_token_ids must match logits.shape[:-1]")
    if target_token_ids.device != logits.device:
        raise ValueError("target_token_ids and logits must be on the same device")
    return torch.gather(
        torch.log_softmax(logits, dim=-1),
        dim=-1,
        index=target_token_ids.unsqueeze(-1).to(dtype=torch.long),
    ).squeeze(-1)


def compute_logp_delta(original_logps: Tensor, perturbed_logps: Tensor, *, mode: str = "absolute") -> Tensor:
    """Return an absolute (paper) or signed (legacy repo) log-probability delta."""

    if original_logps.shape != perturbed_logps.shape:
        raise ValueError("original_logps and perturbed_logps must have equal shapes")
    if original_logps.device != perturbed_logps.device:
        raise ValueError("original_logps and perturbed_logps must be on the same device")
    difference = original_logps - perturbed_logps
    normalised = mode.lower().replace("-", "_")
    if normalised in {"signed", "repo", "original_minus_perturbed"}:
        return difference
    if normalised in {"absolute", "abs", "paper"}:
        return difference.abs()
    raise ValueError("mode must be 'absolute'/'paper' or 'signed'/'repo'")


def _select_group(scores: Tensor, group_mask: Tensor, output: Tensor, ratio: float, mode: str) -> None:
    values = scores[group_mask].float()
    count = values.numel()
    if count == 0:
        return
    if mode == "repo":
        output[group_mask] = values >= torch.quantile(values, 1.0 - ratio)
        return
    keep = min(count, int(math.ceil(count * ratio)))
    if keep == 0:
        return
    try:
        order = torch.argsort(values, descending=True, stable=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch only
        order = torch.argsort(values, descending=True)
    selected = torch.zeros(count, dtype=torch.bool, device=scores.device)
    selected[order[:keep]] = True
    output[group_mask] = selected


def select_top_ratio(
    scores: Tensor,
    valid_mask: Tensor,
    ratio: float,
    *,
    mode: str = "paper",
    scope: Optional[str] = None,
) -> Tensor:
    """Select top attribution scores using the documented repo/paper protocol."""

    if scores.ndim != 2 or valid_mask.shape != scores.shape:
        raise ValueError("scores and valid_mask must have matching [batch, completion_tokens] shapes")
    if scores.device != valid_mask.device:
        raise ValueError("scores and valid_mask must be on the same device")
    if not 0.0 <= float(ratio) <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    normalised_mode = mode.lower().replace("-", "_")
    if normalised_mode in {"repo", "repo_batch"}:
        selection_mode = "repo"
    elif normalised_mode in {"paper", "paper_per_completion"}:
        selection_mode = "paper"
    else:
        raise ValueError("mode must be 'repo' or 'paper'")
    selection_scope = scope.lower().replace("-", "_") if scope else (
        "batch" if selection_mode == "repo" else "per_completion"
    )
    if selection_scope in {"repo_batch"}:
        selection_scope = "batch"
    if selection_scope in {"paper_per_completion"}:
        selection_scope = "per_completion"
    if selection_scope not in {"batch", "per_completion"}:
        raise ValueError("scope must be 'batch' or 'per_completion'")

    mask = valid_mask.to(dtype=torch.bool)
    output = torch.zeros_like(mask, dtype=torch.bool)
    if ratio == 0.0 or not bool(mask.any().item()):
        return output
    if selection_scope == "batch":
        _select_group(scores, mask, output, float(ratio), selection_mode)
    else:
        for row in range(scores.shape[0]):
            _select_group(scores[row], mask[row], output[row], float(ratio), selection_mode)
    return output


def union_masks(*masks: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
    """Return a boolean union, optionally restricted to valid completion tokens."""

    if not masks:
        raise ValueError("at least one mask is required")
    result = masks[0].to(dtype=torch.bool).clone()
    for mask in masks[1:]:
        if mask.shape != result.shape or mask.device != result.device:
            raise ValueError("all masks must share shape and device")
        result |= mask.to(dtype=torch.bool)
    if valid_mask is not None:
        if valid_mask.shape != result.shape or valid_mask.device != result.device:
            raise ValueError("valid_mask must share mask shape and device")
        result &= valid_mask.to(dtype=torch.bool)
    return result
