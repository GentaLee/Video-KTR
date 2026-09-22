#!/usr/bin/env python3
"""Import the exact GRPO entrypoint in a torchrun worker without training.

The B200 full launcher runs this on eight CPU-only ranks before the expensive
media preflight.  It catches worker-only dependency failures such as an Apex
installation that is visible to one interpreter but lacks ``apex.amp``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--expected-python", type=Path, required=True)
    args = parser.parse_args()
    entrypoint = args.entrypoint.resolve(strict=True)
    if not entrypoint.is_file():
        raise RuntimeError(f"GRPO entrypoint is not a file: {entrypoint}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("B200 import gate must not expose GPUs")
    # Do not resolve these symlinks: both the project and platform venv Python
    # executables can resolve to /usr/bin/python3.12 on this image.
    observed_python = os.path.abspath(sys.executable)
    expected_python = os.path.abspath(args.expected_python)
    if observed_python != expected_python:
        raise RuntimeError(
            f"torchrun worker interpreter mismatch: {observed_python} != {expected_python}"
        )
    runpy.run_path(str(entrypoint), run_name="video_ktr_b200_import_gate")
    print(
        "[b200-training-import] PASS "
        f"rank={os.environ.get('LOCAL_RANK', 'unknown')} "
        f"python={sys.executable}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
