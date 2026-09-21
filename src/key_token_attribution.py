"""Pure tensor utilities for selecting Video-KTR key tokens.

This module intentionally has no trainer, model, processor, or filesystem
dependency.  A caller supplies aligned teacher-forced logits/log-probabilities
for the generated completion and a completion-valid mask.  The utilities then
produce the entropy, visual/temporal dependency scores, individual masks, and
their union.

Two selection conventions are supported explicitly:

* ``repo``: signed ``original - perturbed`` dependency scores and the original
  repository's batch-wide quantile threshold (ties are retained).
* ``paper``: absolute dependency scores and an exact top-k selection for each
  completion (``ceil(ratio * valid_tokens)``; ties are resolved by position).

The distinction is deliberately not hidden: callers should record both ``mode``
and ``scope`` with any reported examples.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor


SelectionMode = Literal["repo", "paper"]
SelectionScope = Literal["batch", "per_completion"]


@dataclass(frozen=True)
class KeyTokenAttribution:
    """Scores and selection masks for one batch of generated completions.

    All tensors except the optional temporal fields have ``[batch, tokens]``
    shape.  Masks are boolean and are always false outside ``valid_mask``.
    """

    entropy: Tensor
    visual_delta: Tensor
    temporal_delta: Optional[Tensor]
    entropy_mask: Tensor
    visual_mask: Tensor
    temporal_mask: Optional[Tensor]
    union_mask: Tensor
    mode: SelectionMode
    scope: SelectionScope


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _require_floating(name: str, value: Tensor) -> None:
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must have a real floating-point dtype, got {value.dtype}")


def _require_finite(name: str, value: Tensor) -> None:
    if value.numel() and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains NaN or infinity")


def _coerce_bool_mask(mask: object, *, expected_shape: torch.Size, device: torch.device, name: str) -> Tensor:
    """Validate a boolean or 0/1 numeric mask and return a bool tensor."""

    tensor = _require_tensor(name, mask)
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")

    if tensor.dtype == torch.bool:
        return tensor
    if tensor.is_complex() or not (tensor.is_floating_point() or tensor.dtype in _INTEGER_DTYPES):
        raise TypeError(f"{name} must be boolean or a numeric 0/1 tensor, got {tensor.dtype}")
    _require_finite(name, tensor) if tensor.is_floating_point() else None
    is_binary = torch.logical_or(tensor == 0, tensor == 1)
    if tensor.numel() and not is_binary.all().item():
        raise ValueError(f"{name} must contain only 0 and 1 values")
    return tensor.to(dtype=torch.bool)


def _normalise_mode(mode: str) -> SelectionMode:
    if not isinstance(mode, str):
        raise TypeError(f"mode must be a string, got {type(mode).__name__}")
    normalised = mode.lower().replace("-", "_")
    aliases = {
        "repo": "repo",
        "repo_batch": "repo",
        "paper": "paper",
        "paper_per_completion": "paper",
    }
    try:
        return aliases[normalised]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError("mode must be 'repo' or 'paper'") from exc


def _normalise_scope(scope: Optional[str], mode: SelectionMode) -> SelectionScope:
    if scope is None:
        return "batch" if mode == "repo" else "per_completion"
    if not isinstance(scope, str):
        raise TypeError(f"scope must be a string or None, got {type(scope).__name__}")
    normalised = scope.lower().replace("-", "_")
    aliases = {
        "batch": "batch",
        "repo_batch": "batch",
        "per_completion": "per_completion",
        "paper_per_completion": "per_completion",
    }
    try:
        return aliases[normalised]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError("scope must be 'batch' or 'per_completion'") from exc


def _validate_ratio(ratio: object) -> float:
    if isinstance(ratio, bool) or not isinstance(ratio, Real):
        raise TypeError(f"ratio must be a real number in [0, 1], got {type(ratio).__name__}")
    ratio_float = float(ratio)
    if not math.isfinite(ratio_float) or not 0.0 <= ratio_float <= 1.0:
        raise ValueError(f"ratio must be finite and in [0, 1], got {ratio!r}")
    return ratio_float


def _validate_score_matrix(scores: object, valid_mask: object) -> tuple[Tensor, Tensor]:
    scores_tensor = _require_tensor("scores", scores)
    _require_floating("scores", scores_tensor)
    if scores_tensor.ndim != 2:
        raise ValueError(
            "scores must have shape [batch, completion_tokens]; "
            f"got {tuple(scores_tensor.shape)}"
        )
    if scores_tensor.shape[0] == 0:
        raise ValueError("scores must contain at least one completion")

    mask = _coerce_bool_mask(
        valid_mask,
        expected_shape=scores_tensor.shape,
        device=scores_tensor.device,
        name="valid_mask",
    )
    # Invalid/padded positions may legitimately carry a sentinel such as -inf.
    # Any non-finite *candidate* score would make its ranking undefined.
    if mask.any().item() and not torch.isfinite(scores_tensor[mask]).all().item():
        raise ValueError("scores contains NaN or infinity at a valid position")
    return scores_tensor, mask


def compute_token_entropy(logits: Tensor) -> Tensor:
    """Return Shannon entropy over the vocabulary for every logit position.

    ``logits`` must have at least two dimensions, with vocabulary on its last
    dimension.  The returned tensor has shape ``logits.shape[:-1]``.  The
    calculation uses ``log_softmax`` for numerical stability.
    """

    logits = _require_tensor("logits", logits)
    _require_floating("logits", logits)
    if logits.ndim < 2:
        raise ValueError(
            "logits must have at least [tokens, vocabulary] dimensions; "
            f"got {tuple(logits.shape)}"
        )
    if logits.shape[-1] == 0:
        raise ValueError("logits vocabulary dimension must be non-empty")
    _require_finite("logits", logits)

    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def target_token_logps(logits: Tensor, target_token_ids: Tensor) -> Tensor:
    """Extract teacher-forced target-token log-probabilities from logits.

    ``target_token_ids`` must match ``logits.shape[:-1]`` exactly.  This helps
    make token alignment failures explicit before visual/temporal deltas are
    computed.
    """

    logits = _require_tensor("logits", logits)
    target_token_ids = _require_tensor("target_token_ids", target_token_ids)
    _require_floating("logits", logits)
    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions")
    if logits.shape[-1] == 0:
        raise ValueError("logits vocabulary dimension must be non-empty")
    if target_token_ids.shape != logits.shape[:-1]:
        raise ValueError(
            "target_token_ids must match logits.shape[:-1]; "
            f"expected {tuple(logits.shape[:-1])}, got {tuple(target_token_ids.shape)}"
        )
    if target_token_ids.device != logits.device:
        raise ValueError(
            "target_token_ids and logits must be on the same device; "
            f"got {target_token_ids.device} and {logits.device}"
        )
    if target_token_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            "target_token_ids must have an integer dtype, "
            f"got {target_token_ids.dtype}"
        )
    _require_finite("logits", logits)
    if target_token_ids.numel():
        min_id = int(target_token_ids.min().item())
        max_id = int(target_token_ids.max().item())
        if min_id < 0 or max_id >= logits.shape[-1]:
            raise ValueError(
                "target_token_ids must be in [0, vocabulary_size); "
                f"got range [{min_id}, {max_id}] for vocabulary size {logits.shape[-1]}"
            )

    return torch.gather(
        torch.log_softmax(logits, dim=-1),
        dim=-1,
        index=target_token_ids.unsqueeze(-1).to(dtype=torch.long),
    ).squeeze(-1)


def compute_logp_delta(
    original_logps: Tensor,
    perturbed_logps: Tensor,
    *,
    mode: str = "absolute",
) -> Tensor:
    """Compute target-token log-probability changes under a perturbation.

    ``mode='absolute'`` (also ``'paper'``) returns
    ``abs(original - perturbed)``.  ``mode='signed'`` (also ``'repo'``)
    returns ``original - perturbed``, matching the existing trainer's sign.
    """

    original_logps = _require_tensor("original_logps", original_logps)
    perturbed_logps = _require_tensor("perturbed_logps", perturbed_logps)
    _require_floating("original_logps", original_logps)
    _require_floating("perturbed_logps", perturbed_logps)
    if original_logps.shape != perturbed_logps.shape:
        raise ValueError(
            "original_logps and perturbed_logps must have equal shapes; "
            f"got {tuple(original_logps.shape)} and {tuple(perturbed_logps.shape)}"
        )
    if original_logps.device != perturbed_logps.device:
        raise ValueError(
            "original_logps and perturbed_logps must be on the same device; "
            f"got {original_logps.device} and {perturbed_logps.device}"
        )
    _require_finite("original_logps", original_logps)
    _require_finite("perturbed_logps", perturbed_logps)

    if not isinstance(mode, str):
        raise TypeError(f"mode must be a string, got {type(mode).__name__}")
    normalised = mode.lower().replace("-", "_")
    signed_aliases = {"signed", "repo", "original_minus_perturbed"}
    absolute_aliases = {"absolute", "abs", "paper"}
    difference = original_logps - perturbed_logps
    if normalised in signed_aliases:
        return difference
    if normalised in absolute_aliases:
        return difference.abs()
    raise ValueError("mode must be 'absolute'/'paper' or 'signed'/'repo'")


def _select_repo_group(scores: Tensor, group_mask: Tensor, output: Tensor, ratio: float) -> None:
    """Apply the trainer's quantile rule to one group, retaining ties."""

    values = scores[group_mask].float()
    if values.numel() == 0:
        return
    threshold = torch.quantile(values, 1.0 - ratio)
    output[group_mask] = values >= threshold


def _select_paper_group(scores: Tensor, group_mask: Tensor, output: Tensor, ratio: float) -> None:
    """Select exact top-k scores in a group, with positional tie-breaking."""

    values = scores[group_mask].float()
    count = values.numel()
    if count == 0:
        return
    keep = min(count, int(math.ceil(count * ratio)))
    if keep == 0:
        return
    # ``stable=True`` makes equal scores choose the earlier completion position,
    # which gives a repeatable exact top-k mask instead of tie-driven expansion.
    try:
        order = torch.argsort(values, descending=True, stable=True)
    except TypeError:  # pragma: no cover - for very old PyTorch releases only
        order = torch.argsort(values, descending=True)
    chosen = torch.zeros(count, dtype=torch.bool, device=scores.device)
    chosen[order[:keep]] = True
    output[group_mask] = chosen


def select_top_ratio(
    scores: Tensor,
    valid_mask: Tensor,
    ratio: float,
    *,
    mode: str = "paper",
    scope: Optional[str] = None,
) -> Tensor:
    """Return a valid-position-only mask for the selected top-score tokens.

    Args:
        scores: Floating ``[batch, completion_tokens]`` attribution scores.
        valid_mask: Same-shaped bool or binary mask.  Padding/EOS-excluded
            positions are never selected.
        ratio: Fraction in ``[0, 1]``.  Zero selects nothing; a positive paper
            ratio selects at least one token in each non-empty completion.
        mode: ``'repo'`` uses quantile thresholding and keeps ties; ``'paper'``
            uses exact top-k.
        scope: ``'batch'`` ranks all valid batch tokens together, while
            ``'per_completion'`` ranks each row independently.  If omitted it
            defaults to batch for repo mode and per-completion for paper mode.

    The score matrix is never mutated.
    """

    scores, mask = _validate_score_matrix(scores, valid_mask)
    ratio = _validate_ratio(ratio)
    selection_mode = _normalise_mode(mode)
    selection_scope = _normalise_scope(scope, selection_mode)
    output = torch.zeros_like(mask, dtype=torch.bool)
    if ratio == 0.0 or not mask.any().item():
        return output

    select_group = _select_repo_group if selection_mode == "repo" else _select_paper_group
    if selection_scope == "batch":
        select_group(scores, mask, output, ratio)
    else:
        for row_index in range(scores.shape[0]):
            select_group(scores[row_index], mask[row_index], output[row_index], ratio)
    return output


def union_masks(*masks: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
    """Return the logical union of compatible selection masks.

    When ``valid_mask`` is supplied, the union is additionally restricted to it.
    This is useful as a final invariant check before serialising selected tokens.
    """

    if not masks:
        raise ValueError("at least one mask is required")
    first = _require_tensor("masks[0]", masks[0])
    shape = first.shape
    device = first.device
    result = _coerce_bool_mask(first, expected_shape=shape, device=device, name="masks[0]").clone()
    for index, candidate in enumerate(masks[1:], start=1):
        candidate_mask = _coerce_bool_mask(
            candidate,
            expected_shape=shape,
            device=device,
            name=f"masks[{index}]",
        )
        result |= candidate_mask
    if valid_mask is not None:
        result &= _coerce_bool_mask(
            valid_mask,
            expected_shape=shape,
            device=device,
            name="valid_mask",
        )
    return result


def attribute_key_tokens(
    logits: Tensor,
    original_logps: Tensor,
    visual_logps: Tensor,
    valid_mask: Tensor,
    *,
    temporal_logps: Optional[Tensor] = None,
    temporal_delta_scores: Optional[Tensor] = None,
    entropy_ratio: float = 0.2,
    visual_ratio: float = 0.2,
    temporal_ratio: float = 0.2,
    mode: str = "paper",
    scope: Optional[str] = None,
) -> KeyTokenAttribution:
    """Compute and select high-entropy, visual, and temporal key tokens.

    ``logits`` and all log-probability tensors must be aligned as
    ``[batch, completion_tokens, ...]`` and ``[batch, completion_tokens]``.
    Passing both ``temporal_logps`` and ``temporal_delta_scores`` is invalid.
    ``temporal_delta_scores`` is for a caller that has aggregated multiple
    frame-shuffle probes already: it must have the selected mode's semantics
    (absolute for ``paper``, signed for ``repo``).  Passing neither is the
    explicit image/no-temporal case; its temporal score and mask are ``None``.
    """

    selection_mode = _normalise_mode(mode)
    selection_scope = _normalise_scope(scope, selection_mode)
    entropy = compute_token_entropy(logits)
    entropy, normalised_valid_mask = _validate_score_matrix(entropy, valid_mask)

    original_logps, _ = _validate_score_matrix(original_logps, normalised_valid_mask)
    visual_logps, _ = _validate_score_matrix(visual_logps, normalised_valid_mask)
    if original_logps.shape != entropy.shape:
        raise ValueError(
            "original_logps must align with logits positions; "
            f"expected {tuple(entropy.shape)}, got {tuple(original_logps.shape)}"
        )
    if visual_logps.shape != entropy.shape:
        raise ValueError(
            "visual_logps must align with logits positions; "
            f"expected {tuple(entropy.shape)}, got {tuple(visual_logps.shape)}"
        )

    delta_mode = "repo" if selection_mode == "repo" else "paper"
    visual_delta = compute_logp_delta(original_logps, visual_logps, mode=delta_mode)
    entropy_mask = select_top_ratio(
        entropy,
        normalised_valid_mask,
        entropy_ratio,
        mode=selection_mode,
        scope=selection_scope,
    )
    visual_mask = select_top_ratio(
        visual_delta,
        normalised_valid_mask,
        visual_ratio,
        mode=selection_mode,
        scope=selection_scope,
    )

    if temporal_logps is not None and temporal_delta_scores is not None:
        raise ValueError("pass either temporal_logps or temporal_delta_scores, not both")

    temporal_delta: Optional[Tensor] = None
    temporal_mask: Optional[Tensor] = None
    masks: list[Tensor] = [entropy_mask, visual_mask]
    if temporal_logps is not None:
        temporal_logps, _ = _validate_score_matrix(temporal_logps, normalised_valid_mask)
        if temporal_logps.shape != entropy.shape:
            raise ValueError(
                "temporal_logps must align with logits positions; "
                f"expected {tuple(entropy.shape)}, got {tuple(temporal_logps.shape)}"
            )
        temporal_delta = compute_logp_delta(original_logps, temporal_logps, mode=delta_mode)
    elif temporal_delta_scores is not None:
        temporal_delta, _ = _validate_score_matrix(
            temporal_delta_scores, normalised_valid_mask
        )
        if temporal_delta.shape != entropy.shape:
            raise ValueError(
                "temporal_delta_scores must align with logits positions; "
                f"expected {tuple(entropy.shape)}, got {tuple(temporal_delta.shape)}"
            )

    if temporal_delta is not None:
        temporal_mask = select_top_ratio(
            temporal_delta,
            normalised_valid_mask,
            temporal_ratio,
            mode=selection_mode,
            scope=selection_scope,
        )
        masks.append(temporal_mask)

    return KeyTokenAttribution(
        entropy=entropy,
        visual_delta=visual_delta,
        temporal_delta=temporal_delta,
        entropy_mask=entropy_mask,
        visual_mask=visual_mask,
        temporal_mask=temporal_mask,
        union_mask=union_masks(*masks, valid_mask=normalised_valid_mask),
        mode=selection_mode,
        scope=selection_scope,
    )


# Names matching the existing trainer and concise names useful to a standalone
# selector.  ``keep_token_by_ratio`` defaults to the trainer-compatible rule.
def keep_token_by_ratio(scores: Tensor, valid_mask: Tensor, ratio: float) -> Tensor:
    """Trainer-compatible alias for repo batch-wide top-ratio selection."""

    return select_top_ratio(scores, valid_mask, ratio, mode="repo", scope="batch")


token_entropy = compute_token_entropy
logp_delta = compute_logp_delta
top_ratio_mask = select_top_ratio


__all__ = [
    "KeyTokenAttribution",
    "SelectionMode",
    "SelectionScope",
    "attribute_key_tokens",
    "compute_logp_delta",
    "compute_token_entropy",
    "keep_token_by_ratio",
    "logp_delta",
    "select_top_ratio",
    "target_token_logps",
    "token_entropy",
    "top_ratio_mask",
    "union_masks",
]
