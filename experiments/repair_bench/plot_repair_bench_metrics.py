from __future__ import annotations

import argparse
import csv
from pathlib import Path


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value in (None, ""):
        return default
    return int(float(value))


def plot_operator_overview(metrics_dir: Path, output_prefix: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in the experiment environment first."
        ) from exc

    rows = read_csv(metrics_dir / "by_operator.csv")
    if not rows:
        raise SystemExit(f"No rows found in {metrics_dir / 'by_operator.csv'}")

    rows = sorted(rows, key=lambda row: row["operator"])
    labels = [OPERATOR_LABELS.get(row["operator"], row["operator"]) for row in rows]
    totals = [as_int(row, "total") for row in rows]
    hit_rates = [100.0 * as_float(row, "localizer_hit_rate") for row in rows]
    detected_rates = [100.0 * as_float(row, "bug_detected_rate") for row in rows]
    first_ranks = [as_float(row, "avg_first_hit_rank") for row in rows]
    p50_elapsed = [as_float(row, "p50_elapsed_sec") for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    x = list(range(len(rows)))

    width = 0.38
    axes[0].bar([i - width / 2 for i in x], detected_rates, width=width, label="Bug detected", color="#7895CB")
    axes[0].bar([i + width / 2 for i in x], hit_rates, width=width, label="Top-k localized", color="#D67D3E")
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("Detection and Localization")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].legend(frameon=False, loc="lower right")
    for i, total in enumerate(totals):
        axes[0].text(i, 103, f"n={total}", ha="center", va="bottom", fontsize=8, color="#4A4A4A")

    axes[1].bar(x, first_ranks, color="#7AA874")
    axes[1].set_ylabel("Average first-hit rank")
    axes[1].set_title("Localization Rank")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].axhline(1.0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.55)

    axes[2].bar(x, p50_elapsed, color="#B46060")
    axes[2].set_ylabel("Median verification time (s)")
    axes[2].set_title("Verification Cost")
    axes[2].set_xticks(x, labels, rotation=35, ha="right")

    fig.suptitle("SMGR Repair-Bench Localizer Performance by Fault Operator", fontsize=13, fontweight="bold")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.svg')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot repair-bench Localizer metrics.")
    parser.add_argument("metrics_dir", type=Path, help="Directory containing by_operator.csv.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path without extension. Defaults to <metrics_dir>/localizer_operator_overview.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_dir = args.metrics_dir.resolve()
    output_prefix = args.output_prefix.resolve() if args.output_prefix else metrics_dir / "localizer_operator_overview"
    plot_operator_overview(metrics_dir, output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
