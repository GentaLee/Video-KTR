#!/usr/bin/env python3
"""Fail-fast import and runtime gate used by the full launcher.

This is a standalone program rather than a shell heredoc so ``run_full.sh``
can supervise it as a tracked process group.  It intentionally does not load
the checkpoint or start distributed training.
"""

from __future__ import annotations

import os

import tokenizers
import torch
import transformers
import trl

from trainer.video_ktr_grpo_trainer import VideoKTRGRPOTrainer


def main() -> int:
    expected = int(os.environ["GRPO_EXPECTED_GPU_COUNT"])
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
    assert transformers.__version__ == "4.49.0.dev0", transformers.__version__
    assert tokenizers.__version__ == "0.21.4", tokenizers.__version__
    assert trl.__version__ == "0.16.0", trl.__version__
    print(
        "[full-preflight] "
        f"torch={torch.__version__}; transformers={transformers.__version__}; "
        f"trl={trl.__version__}; devices={[torch.cuda.get_device_name(i) for i in range(expected)]}; "
        f"trainer={VideoKTRGRPOTrainer.__module__}; "
        f"video_reader={os.environ['FORCE_QWENVL_VIDEO_READER']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
