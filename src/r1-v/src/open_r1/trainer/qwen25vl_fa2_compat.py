"""Narrow Qwen2.5-VL FlashAttention-2 rotary dtype compatibility shim.

The pinned Transformers 4.49.0.dev0 snapshot used by the supplied checkpoint
promotes Q/K to FP32 inside ``apply_rotary_pos_emb_flashatt`` but can leave the
position ``cos``/``sin`` tensors in BF16 under ZeRO-3.  FlashAttention 2's
Triton rotary implementation requires all three inputs to share a dtype.

Transformers later fixed this exact path by passing ``cos.float()`` and
``sin.float()`` alongside ``q.float()``/``k.float()`` (upstream commit
8ee50537fe7613b87881cd043a85971c85e99519).  We retain the checkpoint's pinned
package version and install that minimal, opt-in equivalent at runtime instead
of mutating site-packages or silently changing the attention backend.
"""

from __future__ import annotations

import importlib
import inspect
from functools import wraps
from types import ModuleType
from typing import Any, Callable


_MODEL_MODULE = "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
_PATCH_MARKER = "_video_ktr_qwen25vl_fa2_rotary_dtype_compat"


def apply_rotary_pos_emb_flashatt_dtype_safe(
    q: Any,
    k: Any,
    cos: Any,
    sin: Any,
    apply_rotary_emb: Callable[[Any, Any, Any], Any],
) -> tuple[Any, Any]:
    """Match the upstream FP32 Qwen FA2 RoPE calculation exactly.

    This deliberately does *not* downcast Q/K to BF16: the legacy helper
    already chose FP32 RoPE arithmetic.  It instead promotes the trig tensors
    to the same FP32 dtype before invoking the FlashAttention rotary kernel,
    then restores each result to the original Q/K dtype.
    """

    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed


def install_qwen25vl_fa2_rotary_dtype_compat(module: ModuleType | None = None) -> bool:
    """Install the scoped helper into the Qwen2.5-VL modeling module.

    Returns ``True`` only when this invocation performs the installation;
    returns ``False`` for a previously installed shim.  The function rejects a
    different helper signature rather than guessing across an unreviewed
    Transformers implementation.
    """

    if module is None:
        module = importlib.import_module(_MODEL_MODULE)

    current = getattr(module, "apply_rotary_pos_emb_flashatt", None)
    if not callable(current):
        raise RuntimeError(
            f"{_MODEL_MODULE} has no callable apply_rotary_pos_emb_flashatt; "
            "refusing to install an unverified FA2 compatibility shim"
        )
    if getattr(current, _PATCH_MARKER, False):
        return False

    parameters = tuple(inspect.signature(current).parameters)
    if parameters != ("q", "k", "cos", "sin"):
        raise RuntimeError(
            "unexpected Qwen2.5-VL FA2 rotary helper signature "
            f"{parameters!r}; refusing to patch an unverified implementation"
        )
    if not callable(getattr(module, "apply_rotary_emb", None)):
        raise RuntimeError(
            f"{_MODEL_MODULE} has no callable apply_rotary_emb; "
            "cannot install the FA2 compatibility shim"
        )

    @wraps(current)
    def patched(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
        return apply_rotary_pos_emb_flashatt_dtype_safe(
            q,
            k,
            cos,
            sin,
            module.apply_rotary_emb,
        )

    setattr(patched, _PATCH_MARKER, True)
    module.apply_rotary_pos_emb_flashatt = patched
    return True
