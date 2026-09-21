#!/usr/bin/env python3
"""Turn rank-local GRPO E/V/T token JSONL evidence into a readable report."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CATEGORIES = (
    ("entropy", "entropy_selected", "高熵 E"),
    ("visual", "visual_selected", "视觉敏感 V"),
    ("temporal", "temporal_selected", "时序敏感 T"),
)


def escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "\\n")


def load_records(training_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(training_dir.glob("selected_tokens_rank*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(loaded)
    return records


def record_key(record: dict[str, Any]) -> tuple[object, ...]:
    return (
        record.get("rank"),
        record.get("problem_id"),
        record.get("completion_index"),
        record.get("position"),
    )


def category_examples(
    records: list[dict[str, Any]], flag: str, limit: int
) -> list[dict[str, Any]]:
    selected = [record for record in records if bool(record.get(flag))]
    # Prefer actual reasoning-token examples, then deterministic rank/sample/
    # position order so a rerun is easy to compare.
    selected.sort(
        key=lambda record: (
            not bool(record.get("in_think")),
            int(record.get("rank", 0)),
            int(record.get("problem_id", 0)),
            int(record.get("completion_index", 0)),
            int(record.get("position", 0)),
        )
    )
    output: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for record in selected:
        key = record_key(record)
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
        if len(output) >= limit:
            break
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--examples-per-category", type=int, default=5)
    parser.add_argument(
        "--require-all-categories",
        action="store_true",
        help="return nonzero unless E, V, and T all have at least one saved example",
    )
    args = parser.parse_args()
    if args.examples_per_category < 1:
        parser.error("--examples-per-category must be positive")

    run_dir = args.run_dir.resolve()
    training_dir = run_dir / "training"
    records = load_records(training_dir)
    if not records:
        raise RuntimeError(f"no selected-token JSONL records found under {training_dir}")

    counts = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    for name, flag, _label in CATEGORIES:
        counts[name] = sum(bool(record.get(flag)) for record in records)
        examples[name] = category_examples(records, flag, args.examples_per_category)
    counts["union"] = sum(bool(record.get("union_selected")) for record in records)
    counts["in_think"] = sum(bool(record.get("in_think")) for record in records)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "records": len(records),
        "counts": dict(counts),
        "examples": examples,
    }
    (run_dir / "token_examples.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# GRPO KTR 三类 token 样例",
        "",
        "这些样例来自训练时实际生成的 completion；`in_think=true` 表示该 token 位于 `<think>` CoT 中。",
        "",
        "| 保存的 union token 记录 | E 标记 | V 标记 | T 标记 | CoT 内记录 |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| %d | %d | %d | %d | %d |"
        % (
            counts["union"],
            counts["entropy"],
            counts["visual"],
            counts["temporal"],
            counts["in_think"],
        ),
    ]
    for name, _flag, label in CATEGORIES:
        lines.extend(
            [
                "",
                f"## {label} token",
                "",
                "| problem | rank / completion / pos | token | CoT 内 | E / V / T | 局部 CoT 上下文 |",
                "| ---: | --- | --- | :---: | --- | --- |",
            ]
        )
        if not examples[name]:
            lines.append("| — | — | — | — | — | 未记录到该类 token |")
        for record in examples[name]:
            flags = "/".join(
                "Y" if bool(record.get(flag)) else "-" for _name, flag, _label in CATEGORIES
            )
            lines.append(
                "| %s | %s / %s / %s | `%s` | %s | %s | %s |"
                % (
                    escape_markdown(record.get("problem_id", "—")),
                    escape_markdown(record.get("rank", "—")),
                    escape_markdown(record.get("completion_index", "—")),
                    escape_markdown(record.get("position", "—")),
                    escape_markdown(record.get("piece", "")),
                    "Y" if bool(record.get("in_think")) else "-",
                    flags,
                    escape_markdown(record.get("token_context", "")),
                )
            )
    (run_dir / "token_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[grpo-token-report] wrote {run_dir / 'token_examples.md'}", flush=True)

    if args.require_all_categories:
        absent = [name for name, _flag, _label in CATEGORIES if counts[name] == 0]
        if absent:
            raise SystemExit("missing selected token categories: " + ", ".join(absent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
