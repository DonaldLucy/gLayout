from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


OPERATOR_LABELS = {
    "label_layer_wrong": "Wrong\nlabel layer",
    "label_moved_to_wrong_port": "Label on\nwrong port",
    "label_text_typo": "Label\ntext typo",
    "missing_connect_subnet": "Missing\nsubnet",
    "netlist_pin_swap": "Netlist\npin swap",
    "physical_route_removed": "Removed\nphysical route",
    "route_spacing_violation": "Route\nspacing",
    "top_node_rename": "Top node\nrename",
}

OPERATOR_ORDER = [
    "label_text_typo",
    "top_node_rename",
    "label_layer_wrong",
    "label_moved_to_wrong_port",
    "netlist_pin_swap",
    "missing_connect_subnet",
    "physical_route_removed",
    "route_spacing_violation",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def operator_sort_key(operator: str) -> tuple[int, str]:
    if operator in OPERATOR_ORDER:
        return (OPERATOR_ORDER.index(operator), operator)
    return (len(OPERATOR_ORDER), operator)


def localizer_rates(metrics_dir: Path | None) -> dict[str, float]:
    if metrics_dir is None:
        return {}
    rates: dict[str, float] = {}
    for row in read_csv(metrics_dir / "by_operator.csv"):
        operator = row.get("operator")
        hit_rate = as_float(row.get("localizer_hit_rate"))
        if operator and hit_rate is not None:
            rates[operator] = hit_rate
    return rates


def zero_shot_operator_rows(summary: dict[str, Any], metrics_dir: Path | None) -> list[dict[str, Any]]:
    localizer_by_operator = localizer_rates(metrics_dir)
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in summary.get("results") or []:
        operator = result.get("operator") or "unknown"
        buckets[operator]["total"] += 1
        if result.get("parse_success") is True:
            buckets[operator]["parse_success"] += 1
        if result.get("apply_success") is True:
            buckets[operator]["apply_success"] += 1
        if result.get("exact_expected_action") is True:
            buckets[operator]["exact_expected_action"] += 1
        if result.get("verification_strict_clean") is True:
            buckets[operator]["verification_strict_clean"] += 1
        if result.get("model_error"):
            buckets[operator]["model_error"] += 1

    rows: list[dict[str, Any]] = []
    for operator in sorted(buckets, key=operator_sort_key):
        stats = buckets[operator]
        total = int(stats["total"])
        rows.append(
            {
                "operator": operator,
                "total": total,
                "localizer_hit_rate": localizer_by_operator.get(operator),
                "parse_success_rate": rate(int(stats["parse_success"]), total),
                "apply_success_rate": rate(int(stats["apply_success"]), total),
                "exact_expected_action_rate": rate(int(stats["exact_expected_action"]), total),
                "verification_strict_clean_rate": rate(int(stats["verification_strict_clean"]), total),
                "parse_success": int(stats["parse_success"]),
                "apply_success": int(stats["apply_success"]),
                "exact_expected_action": int(stats["exact_expected_action"]),
                "verification_strict_clean": int(stats["verification_strict_clean"]),
                "model_error": int(stats["model_error"]),
            }
        )
    return rows


def pct_or_zero(value: float | None) -> float:
    return 0.0 if value is None else 100.0 * value


def add_bar_labels(axis: Any, bars: Any, totals: list[int], values: list[float | None]) -> None:
    for index, bar in enumerate(bars):
        value = values[index]
        if value is None:
            label = "n/a"
            y = 2.0
        else:
            label = f"{100.0 * value:.0f}%"
            y = min(103.0, 100.0 * value + 2.0)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
            rotation=90 if len(totals) > 6 else 0,
        )


def plot_by_operator(rows: list[dict[str, Any]], output_prefix: Path, *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install it in the experiment environment first.") from exc

    if not rows:
        raise SystemExit("No zero-shot rows to plot.")

    labels = [OPERATOR_LABELS.get(row["operator"], row["operator"]) for row in rows]
    totals = [int(row["total"]) for row in rows]
    localizer = [row.get("localizer_hit_rate") for row in rows]
    apply_success = [row.get("apply_success_rate") for row in rows]
    verified = [row.get("verification_strict_clean_rate") for row in rows]

    has_localizer = any(value is not None for value in localizer)
    series: list[tuple[str, list[float | None], str]] = []
    if has_localizer:
        series.append(("SMGR top-k localized", localizer, "#4C78A8"))
    series.append(("Patch applied", apply_success, "#F58518"))
    series.append(("Strict DRC/LVS clean", verified, "#54A24B"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    fig_width = max(9.2, 1.05 * len(rows) + 3.5)
    fig, axis = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)
    x = list(range(len(rows)))
    width = min(0.24, 0.78 / len(series))
    offsets = [(idx - (len(series) - 1) / 2) * width for idx in range(len(series))]

    for offset, (name, values, color) in zip(offsets, series):
        bars = axis.bar(
            [i + offset for i in x],
            [pct_or_zero(value) for value in values],
            width=width,
            label=name,
            color=color,
            alpha=0.92,
        )
        add_bar_labels(axis, bars, totals, values)

    for index, total in enumerate(totals):
        axis.text(index, -10.0, f"n={total}", ha="center", va="top", fontsize=8, color="#555555")

    axis.set_ylim(-16, 112)
    axis.set_ylabel("Rate (%)")
    axis.set_xticks(x, labels, rotation=34, ha="right")
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.legend(frameon=False, loc="upper right", ncols=1)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    write_csv(output_prefix.with_suffix(".csv"), rows)
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.svg')}")
    print(f"Wrote {output_prefix.with_suffix('.csv')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot zero-shot repair success by fault operator.")
    parser.add_argument("zero_shot_run_dir", type=Path, help="Directory containing zero_shot_summary.json.")
    parser.add_argument(
        "--localizer-metrics-dir",
        type=Path,
        default=None,
        help="Optional metrics directory containing by_operator.csv for SMGR localizer hit-rate bars.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path without extension. Defaults to <zero_shot_run_dir>/zero_shot_by_operator.",
    )
    parser.add_argument(
        "--title",
        default="Zero-Shot Repair Outcome by Fault Operator",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.zero_shot_run_dir.resolve()
    summary = load_json(run_dir / "zero_shot_summary.json")
    metrics_dir = args.localizer_metrics_dir.resolve() if args.localizer_metrics_dir else None
    output_prefix = args.output_prefix.resolve() if args.output_prefix else run_dir / "zero_shot_by_operator"
    rows = zero_shot_operator_rows(summary, metrics_dir)
    plot_by_operator(rows, output_prefix, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
