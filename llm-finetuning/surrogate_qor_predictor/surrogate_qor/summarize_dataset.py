from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _get(record: dict[str, Any], path: str) -> Any:
    node: Any = record
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def summarize(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_generator[str(record.get("generator_id", ""))].append(record)

    def pass_rate(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if not values:
            return None
        return sum(1 for value in values if value) / len(values)

    runtimes = defaultdict(list)
    areas: list[float] = []
    drc_rules: Counter[str] = Counter()
    lvs_status: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for record in records:
        for name, value in (record.get("timings_s") or {}).items():
            if isinstance(value, (int, float)):
                runtimes[name].append(float(value))
        area = _get(record, "features.geometric.area_um2")
        if isinstance(area, (int, float)):
            areas.append(float(area))
        for rule, count in (_get(record, "drc.rule_counts") or {}).items():
            drc_rules[str(rule)] += int(count)
        status = _get(record, "lvs.status")
        if status:
            lvs_status[str(status)] += 1
        if record.get("error"):
            errors[str(record.get("error")).splitlines()[0][:160]] += 1

    per_generator = {}
    for generator, rows in sorted(by_generator.items()):
        per_generator[generator] = {
            "samples": len(rows),
            "ok_rate": pass_rate(rows, "ok"),
            "drc_pass_rate": pass_rate(rows, "drc_pass"),
            "lvs_pass_rate": pass_rate(rows, "lvs_pass"),
            "pex_pass_rate": pass_rate(rows, "pex_pass"),
            "avg_total_runtime_s": _mean([float(_get(row, "timings_s.total")) for row in rows if isinstance(_get(row, "timings_s.total"), (int, float))]),
        }

    return {
        "dataset": str(path),
        "samples": len(records),
        "generators": len(by_generator),
        "ok_rate": pass_rate(records, "ok"),
        "drc_pass_rate": pass_rate(records, "drc_pass"),
        "lvs_pass_rate": pass_rate(records, "lvs_pass"),
        "pex_pass_rate": pass_rate(records, "pex_pass"),
        "area_um2": {
            "mean": _mean(areas),
            "min": min(areas) if areas else None,
            "max": max(areas) if areas else None,
        },
        "runtime_s_mean": {name: _mean(values) for name, values in sorted(runtimes.items())},
        "top_drc_rules": drc_rules.most_common(20),
        "lvs_status_counts": dict(lvs_status),
        "top_errors": errors.most_common(20),
        "per_generator": per_generator,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a surrogate QoR JSONL dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = summarize(Path(args.dataset))
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

