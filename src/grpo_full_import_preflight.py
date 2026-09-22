#!/usr/bin/env python3
"""Fail-fast import and runtime gate used by the full launcher.

This is a standalone program rather than a shell heredoc so ``run_full.sh``
can supervise it as a tracked process group.  It intentionally does not load
the checkpoint or start distributed training.
"""

from __future__ import annotations

import os
from pathlib import Path

import tokenizers
import torch
import transformers
import trl

from grpo import legacy_deepspeed_resume_safe_globals
from trainer.video_ktr_grpo_trainer import VideoKTRGRPOTrainer


def main() -> int:
    expected = int(os.environ["GRPO_EXPECTED_GPU_COUNT"])
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
    assert transformers.__version__ == "4.49.0.dev0", transformers.__version__
    assert tokenizers.__version__ == "0.21.4", tokenizers.__version__
    assert trl.__version__ == "0.16.0", trl.__version__
    resume_probe = os.environ.get("GRPO_RESUME_CHECKPOINT_PROBE")
    if resume_probe:
        probe_path = Path(resume_probe)
        assert probe_path.is_file(), f"resume probe is unavailable: {probe_path}"
        expected_unsafe = {
            "deepspeed.runtime.fp16.loss_scaler.LossScaler",
            "deepspeed.runtime.zero.config.ZeroStageEnum",
        }
        scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
        optimizer_shards = sorted(probe_path.parent.glob("*_optim_states.pt"))
        assert len(optimizer_shards) == expected, (
            f"expected {expected} optimizer shards next to resume probe, found {len(optimizer_shards)}"
        )
        if scanner is not None:
            for shard in optimizer_shards:
                unsafe = set(scanner(shard))
                assert unsafe == expected_unsafe, (
                    f"unexpected unsafe globals in {shard.name}: {sorted(unsafe)}"
                )
        # Enter the same restricted context that each torchrun rank will use,
        # then exercise DeepSpeed's exact checkpoint-engine load call without
        # loading the 24 GiB optimizer shards or beginning a training step.
        from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import (
            TorchCheckpointEngine,
        )

        with legacy_deepspeed_resume_safe_globals(str(probe_path)):
            probe_state = TorchCheckpointEngine().load(probe_path, map_location="cpu")
        assert isinstance(probe_state, dict) and probe_state, "invalid DeepSpeed resume probe"
        resume_status = "restricted_deepspeed_resume_probe=passed"
    else:
        resume_status = "restricted_deepspeed_resume_probe=not_applicable"
    print(
        "[full-preflight] "
        f"torch={torch.__version__}; transformers={transformers.__version__}; "
        f"trl={trl.__version__}; devices={[torch.cuda.get_device_name(i) for i in range(expected)]}; "
        f"trainer={VideoKTRGRPOTrainer.__module__}; "
        f"video_reader={os.environ['FORCE_QWENVL_VIDEO_READER']}; {resume_status}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
