#!/usr/bin/env python3
"""Sample per-GPU utilization and memory into a flushed JSONL artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


STOP = False


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def numeric(value: str) -> float | int | None:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def query_gpus() -> tuple[list[dict[str, Any]], str | None]:
    fields = (
        "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,"
        "utilization.memory,temperature.gpu,power.draw,power.limit"
    )
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}"
        return [], detail
    names = (
        "gpu_index",
        "gpu_uuid",
        "gpu_name",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "gpu_utilization_pct",
        "memory_utilization_pct",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
    )
    records: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(names):
            continue
        record: dict[str, Any] = {}
        for name, value in zip(names, values):
            record[name] = value if name in {"gpu_uuid", "gpu_name"} else numeric(value)
        records.append(record)
    return records, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    samples = 0
    print(
        f"[resource-monitor] started interval={args.interval_seconds}s output={args.output}",
        flush=True,
    )
    with args.output.open("a", encoding="utf-8", buffering=1) as handle:
        while not STOP:
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            gpu_records, error = query_gpus()
            common = {
                "timestamp_utc": timestamp,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "run_label": args.run_label,
            }
            if error is not None:
                handle.write(json.dumps({**common, "event": "monitor_error", "error": error}) + "\n")
            else:
                for record in gpu_records:
                    handle.write(json.dumps({**common, "event": "gpu_sample", **record}) + "\n")
                samples += len(gpu_records)
            handle.flush()
            deadline = time.monotonic() + args.interval_seconds
            while not STOP and time.monotonic() < deadline:
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    print(f"[resource-monitor] stopped gpu_records={samples}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
