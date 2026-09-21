#!/usr/bin/env python3
"""Summarize a GRPO launch's duration, trainer metrics, and GPU telemetry."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def finite_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resource-log", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--started-epoch-seconds", type=float, required=True)
    parser.add_argument("--ended-epoch-seconds", type=float, required=True)
    args = parser.parse_args()

    samples_by_gpu: dict[str, list[dict[str, Any]]] = defaultdict(list)
    monitor_errors: list[str] = []
    if args.resource_log.is_file():
        for line in args.resource_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "gpu_sample":
                samples_by_gpu[str(record.get("gpu_index"))].append(record)
            elif record.get("event") == "monitor_error":
                monitor_errors.append(str(record.get("error")))

    gpu_summary: dict[str, dict[str, float | int | None]] = {}
    for gpu, records in sorted(samples_by_gpu.items(), key=lambda item: int(item[0])):
        memory = finite_values(records, "memory_used_mib")
        utilization = finite_values(records, "gpu_utilization_pct")
        power = finite_values(records, "power_draw_w")
        gpu_summary[gpu] = {
            "samples": len(records),
            "memory_used_mib_peak": max(memory) if memory else None,
            "memory_used_mib_mean": fmean(memory) if memory else None,
            "gpu_utilization_pct_peak": max(utilization) if utilization else None,
            "gpu_utilization_pct_mean": fmean(utilization) if utilization else None,
            "power_draw_w_peak": max(power) if power else None,
            "power_draw_w_mean": fmean(power) if power else None,
        }

    training_complete = load_json(args.run_dir / "training" / "training_complete.json")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "variant": args.variant,
        "exit_code": args.exit_code,
        "succeeded": args.exit_code == 0,
        "duration_seconds": round(max(0.0, args.ended_epoch_seconds - args.started_epoch_seconds), 3),
        "resource_log": str(args.resource_log),
        "monitor_errors": monitor_errors,
        "gpus": gpu_summary,
        "training_complete": training_complete,
    }
    json_path = args.run_dir / "training_summary.json"
    markdown_path = args.run_dir / "training_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# GRPO {args.variant} 训练资源摘要",
        "",
        f"- 状态：`{'PASS' if args.exit_code == 0 else 'FAIL'}`（exit={args.exit_code}）",
        f"- 外层运行时长：`{summary['duration_seconds']:.3f}s`",
        f"- 采样日志：`{args.resource_log}`",
        "",
        "## GPU 峰值与均值",
        "",
        "| GPU | 样本数 | 显存峰值 MiB | 平均显存 MiB | 峰值利用率 % | 平均利用率 % |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for gpu, record in gpu_summary.items():
        def fmt(value: float | int | None) -> str:
            return "—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)
        lines.append(
            f"| {gpu} | {record['samples']} | {fmt(record['memory_used_mib_peak'])} | "
            f"{fmt(record['memory_used_mib_mean'])} | {fmt(record['gpu_utilization_pct_peak'])} | "
            f"{fmt(record['gpu_utilization_pct_mean'])} |"
        )
    if training_complete is not None:
        lines.extend([
            "",
            "## Trainer 完成记录",
            "",
            f"- global_step：`{training_complete.get('global_step', '—')}`",
            f"- train_runtime：`{training_complete.get('train_metrics', {}).get('train_runtime', '—')}`",
            f"- train_loss：`{training_complete.get('train_metrics', {}).get('train_loss', '—')}`",
        ])
    if monitor_errors:
        lines.extend(["", "## 监控告警", "", *[f"- `{item}`" for item in monitor_errors]])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[grpo-run-summary] wrote {markdown_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
