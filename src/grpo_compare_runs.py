#!/usr/bin/env python3
"""Create a compact baseline-vs-KTR duration and GPU-resource comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "training_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing run summary: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"summary is not an object: {path}")
    return loaded


def numeric(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def max_gpu_field(summary: dict[str, Any], field: str) -> float | None:
    values = [
        numeric(record.get(field))
        for record in summary.get("gpus", {}).values()
        if isinstance(record, dict)
    ]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def field(summary: dict[str, Any], name: str) -> float | None:
    completion = summary.get("training_complete")
    if not isinstance(completion, dict):
        return None
    latest = completion.get("latest_step_metrics")
    if not isinstance(latest, dict):
        return None
    return numeric(latest.get(name))


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else right - left


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ktr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    ktr_dir = args.ktr_dir.resolve()
    output_dir = args.output_dir.resolve()
    baseline = load_summary(baseline_dir)
    ktr = load_summary(ktr_dir)
    if args.require_pass and (not baseline.get("succeeded") or not ktr.get("succeeded")):
        raise RuntimeError("cannot create a passing comparison from a failed baseline or KTR run")

    duration_baseline = numeric(baseline.get("duration_seconds"))
    duration_ktr = numeric(ktr.get("duration_seconds"))
    peak_memory_baseline = max_gpu_field(baseline, "memory_used_mib_peak")
    peak_memory_ktr = max_gpu_field(ktr, "memory_used_mib_peak")
    peak_util_baseline = max_gpu_field(baseline, "gpu_utilization_pct_peak")
    peak_util_ktr = max_gpu_field(ktr, "gpu_utilization_pct_peak")
    comparison = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline_dir": str(baseline_dir),
        "ktr_dir": str(ktr_dir),
        "baseline_succeeded": bool(baseline.get("succeeded")),
        "ktr_succeeded": bool(ktr.get("succeeded")),
        "resources": {
            "duration_seconds": {
                "baseline": duration_baseline,
                "ktr": duration_ktr,
                "ktr_minus_baseline": delta(duration_baseline, duration_ktr),
            },
            "max_per_gpu_memory_used_mib_peak": {
                "baseline": peak_memory_baseline,
                "ktr": peak_memory_ktr,
                "ktr_minus_baseline": delta(peak_memory_baseline, peak_memory_ktr),
            },
            "max_per_gpu_utilization_pct_peak": {
                "baseline": peak_util_baseline,
                "ktr": peak_util_ktr,
                "ktr_minus_baseline": delta(peak_util_baseline, peak_util_ktr),
            },
        },
        "ktr_latest_step_metrics": {
            name: field(ktr, name)
            for name in (
                "completion_length",
                "reward",
                "kl",
                "ktr/entropy_tokens",
                "ktr/visual_tokens",
                "ktr/temporal_tokens",
                "ktr/union_tokens",
                "ktr/update_ratio",
            )
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# GRPO baseline 与 KTR 对比",
        "",
        "| 指标 | baseline | KTR | KTR − baseline |",
        "| --- | ---: | ---: | ---: |",
        "| 外层训练时长（秒） | %s | %s | %s |"
        % (
            fmt(duration_baseline),
            fmt(duration_ktr),
            fmt(delta(duration_baseline, duration_ktr)),
        ),
        "| 单卡显存峰值（MiB，跨已监控 GPU 最大） | %s | %s | %s |"
        % (
            fmt(peak_memory_baseline, 2),
            fmt(peak_memory_ktr, 2),
            fmt(delta(peak_memory_baseline, peak_memory_ktr), 2),
        ),
        "| 单卡 GPU 利用率峰值（%%，跨已监控 GPU 最大） | %s | %s | %s |"
        % (
            fmt(peak_util_baseline, 2),
            fmt(peak_util_ktr, 2),
            fmt(delta(peak_util_baseline, peak_util_ktr), 2),
        ),
        "",
        "## KTR 最后一个优化步的 token 统计",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
    ]
    for name, value in comparison["ktr_latest_step_metrics"].items():
        lines.append(f"| {name} | {fmt(value)} |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[grpo-compare-runs] wrote {output_dir / 'comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
