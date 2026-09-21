#!/usr/bin/env python3
"""Build a resumable, evidence-oriented summary of sequential key-token runs.

This program is intentionally CPU-only.  The validation launcher calls it
after each manually started GPU batch, but it can also be rerun independently
on a shared artifact directory after an interruption.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return nonzero unless every expected run has a valid success marker",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    if not data:
        return {"count": 0, "mean": None, "p50": None, "p80": None, "p95": None, "max": None}
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "p50": percentile(data, 0.5),
        "p80": percentile(data, 0.8),
        "p95": percentile(data, 0.95),
        "max": max(data),
    }


def result_fingerprint(result: dict[str, Any]) -> str:
    """Fingerprint every value that should be deterministic for a repeated run."""

    metadata = result["metadata"]
    token_view = [
        (
            token["token_id"],
            token["entropy"],
            token["visual_delta"],
            token["temporal_delta"],
            token["entropy_selected"],
            token["visual_selected"],
            token["temporal_selected"],
            token["union_selected"],
        )
        for token in result["tokens"]
    ]
    payload = {
        "case_id": metadata.get("case_id"),
        "seed": metadata.get("seed"),
        "temporal_seed": metadata.get("temporal_permutation_seed"),
        "video_sha256": metadata.get("video_sha256"),
        "selection_mode": metadata.get("selection_mode"),
        "selection_scope": metadata.get("selection_scope"),
        "visual_perturbation": metadata.get("visual_perturbation"),
        "temporal_permutations": metadata.get("temporal_permutations"),
        "completion": result["completion"],
        "tokens": token_view,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def load_successes(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    successes: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for status_path in sorted(run_dir.glob("cases/**/status.json")):
        result_path = status_path.with_name("result.json")
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            result_bytes = result_path.read_bytes()
            result = json.loads(result_bytes)
            if status.get("status") != "success":
                raise ValueError(f"status is {status.get('status')!r}")
            if status.get("result_sha256") != sha256_bytes(result_bytes):
                raise ValueError("result SHA256 does not match success marker")
            metadata = result["metadata"]
            if status.get("case_id") != metadata.get("case_id"):
                raise ValueError("case id differs between result and success marker")
            if status.get("run_config_sha256") != metadata.get("run_config_sha256"):
                raise ValueError("run configuration digest differs between result and success marker")
            if not isinstance(result.get("tokens"), list) or not result["tokens"]:
                raise ValueError("result has no token records")
            successes.append(
                {
                    "directory": str(status_path.parent.relative_to(run_dir)),
                    "result": result,
                    "fingerprint": result_fingerprint(result),
                }
            )
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as error:
            invalid.append({"directory": str(status_path.parent.relative_to(run_dir)), "error": str(error)})
    return successes, invalid


def safe_jaccard(left: set[int], right: set[int]) -> float | None:
    union = left | right
    return len(left & right) / len(union) if union else None


def make_summary(run_dir: Path, expected_runs: int | None) -> dict[str, Any]:
    successes, invalid = load_successes(run_dir)
    failures_path = run_dir / "failures.tsv"
    failures = []
    if failures_path.is_file():
        for line in failures_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            failures.append(
                {
                    "run_index": fields[0] if len(fields) > 0 else "",
                    "case_id": fields[1] if len(fields) > 1 else "",
                    "seed": fields[2] if len(fields) > 2 else "",
                    "exit_code": fields[3] if len(fields) > 3 else "",
                    "directory": fields[4] if len(fields) > 4 else "",
                }
            )

    stop_reasons: Counter[str] = Counter()
    control_failures: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repeat_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    all_visual: list[float] = []
    all_temporal: list[float] = []
    all_entropy: list[float] = []

    for item in successes:
        result = item["result"]
        metadata = result["metadata"]
        tokens = result["tokens"]
        case_id = str(metadata.get("case_id", "unknown"))
        by_case[case_id].append(item)
        repeat_key = (
            case_id,
            metadata.get("seed"),
            metadata.get("temporal_permutation_seed"),
            metadata.get("video_sha256"),
            metadata.get("selector_sha256"),
            metadata.get("attribution_sha256"),
        )
        repeat_groups[repeat_key].append(item)
        stop_reasons[str(metadata.get("generation_stop_reason", "unknown"))] += 1

        valid = [token for token in tokens if token.get("union_selected") is not None]
        visual_values = [float(token["visual_delta"]) for token in valid]
        temporal_values = [float(token["temporal_delta"]) for token in valid]
        entropy_values = [float(token["entropy"]) for token in valid]
        all_visual.extend(visual_values)
        all_temporal.extend(temporal_values)
        all_entropy.extend(entropy_values)
        metrics = metadata.get("control_metrics", {})
        tolerance = float(metadata.get("control_tolerance", 1e-4))
        for metric_name in ("original_auto_vs_fixed_position_max_abs_logp", "identity_video_max_abs_logp"):
            if metric_name in metrics and float(metrics[metric_name]) > tolerance:
                control_failures.append(
                    {
                        "directory": item["directory"],
                        "metric": metric_name,
                        "value": metrics[metric_name],
                        "tolerance": tolerance,
                    }
                )
        rows.append(
            {
                "directory": item["directory"],
                "case_id": case_id,
                "seed": metadata.get("seed"),
                "temporal_seed": metadata.get("temporal_permutation_seed"),
                "stop_reason": metadata.get("generation_stop_reason"),
                "counts": result.get("counts", {}),
                "visual": numeric_summary(visual_values),
                "temporal": numeric_summary(temporal_values),
                "entropy": numeric_summary(entropy_values),
                "controls": metrics,
                "fingerprint": item["fingerprint"],
            }
        )

    case_summaries = []
    for case_id, items in sorted(by_case.items()):
        union_sets = [
            {token["position"] for token in item["result"]["tokens"] if token["union_selected"]}
            for item in items
        ]
        jaccards = [
            value
            for index, first in enumerate(union_sets)
            for second in union_sets[index + 1 :]
            if (value := safe_jaccard(first, second)) is not None
        ]
        case_summaries.append(
            {
                "case_id": case_id,
                "runs": len(items),
                "union_jaccard": numeric_summary(jaccards),
                "seeds": [item["result"]["metadata"].get("seed") for item in items],
            }
        )

    repeat_summary = []
    for key, items in sorted(repeat_groups.items(), key=lambda pair: str(pair[0])):
        if len(items) < 2:
            continue
        fingerprints = {item["fingerprint"] for item in items}
        repeat_summary.append(
            {
                "case_id": key[0],
                "seed": key[1],
                "temporal_seed": key[2],
                "runs": [item["directory"] for item in items],
                "identical": len(fingerprints) == 1,
                "fingerprints": sorted(fingerprints),
            }
        )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir.resolve()),
        "expected_runs": expected_runs,
        "completed_runs": len(successes),
        "missing_runs": max(0, expected_runs - len(successes)) if expected_runs is not None else None,
        "unexpected_runs": max(0, len(successes) - expected_runs) if expected_runs is not None else None,
        "invalid_success_markers": invalid,
        "failures": failures,
        "stop_reasons": dict(stop_reasons),
        "control_failures": control_failures,
        "global_scores": {
            "entropy": numeric_summary(all_entropy),
            "visual_delta": numeric_summary(all_visual),
            "temporal_delta": numeric_summary(all_temporal),
        },
        "cases": case_summaries,
        "repeat_seed_checks": repeat_summary,
        "runs": rows,
    }


def write_outputs(run_dir: Path, summary: dict[str, Any]) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Key-token validation summary",
        "",
        "This is an inference-only attribution aggregate; it does not run GRPO or update weights.",
        "",
        "| expected | completed | missing | unexpected | failed launcher cases | invalid markers |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        "| %s | %s | %s | %s | %s | %s |"
        % (
            summary["expected_runs"] if summary["expected_runs"] is not None else "—",
            summary["completed_runs"],
            summary["missing_runs"] if summary["missing_runs"] is not None else "—",
            summary["unexpected_runs"] if summary["unexpected_runs"] is not None else "—",
            len(summary["failures"]),
            len(summary["invalid_success_markers"]),
        ),
        "",
        "## Stop reasons",
        "",
        "```json",
        json.dumps(summary["stop_reasons"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Score distributions over valid completion tokens",
        "",
        "| score | n | mean | p50 | p80 | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in summary["global_scores"].items():
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                name,
                stats["count"],
                _format(stats["mean"]),
                _format(stats["p50"]),
                _format(stats["p80"]),
                _format(stats["p95"]),
                _format(stats["max"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per-case consistency",
            "",
            "| case | runs | union Jaccard mean | union Jaccard p50 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for case in summary["cases"]:
        stats = case["union_jaccard"]
        lines.append(
            "| %s | %s | %s | %s |"
            % (case["case_id"], case["runs"], _format(stats["mean"]), _format(stats["p50"]))
        )
    lines.extend(["", "## Repeated-seed determinism", ""])
    if summary["repeat_seed_checks"]:
        lines.extend(["| case | seed | temporal seed | identical |", "| --- | ---: | ---: | :---: |"])
        for repeat in summary["repeat_seed_checks"]:
            lines.append(
                "| %s | %s | %s | %s |"
                % (repeat["case_id"], repeat["seed"], repeat["temporal_seed"], "yes" if repeat["identical"] else "NO")
            )
    else:
        lines.append("No duplicate `(case, generation seed, temporal seed)` run was found.")
    if summary["control_failures"]:
        lines.extend(["", "## Control failures", "", "```json", json.dumps(summary["control_failures"], indent=2), "```"])
    lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _format(value: float | int | None) -> str:
    return "—" if value is None else f"{float(value):.6g}"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    if args.expected_runs is not None and args.expected_runs < 0:
        raise ValueError("--expected-runs must be non-negative")
    summary = make_summary(run_dir, args.expected_runs)
    write_outputs(run_dir, summary)
    complete = (
        not summary["invalid_success_markers"]
        and not summary["failures"]
        and not summary["control_failures"]
        and (summary["missing_runs"] in (None, 0))
        and (summary["unexpected_runs"] in (None, 0))
    )
    print(
        "[validation-summary] completed=%d expected=%s failures=%d controls=%d"
        % (
            summary["completed_runs"],
            summary["expected_runs"],
            len(summary["failures"]),
            len(summary["control_failures"]),
        ),
        flush=True,
    )
    return 0 if complete or not args.require_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
