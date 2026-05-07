#!/usr/bin/env python3
"""
Single-purpose plotting script for a publication-quality
"Sidecar Size vs. Generator Depth" figure.

Design goals:
- avoid overlapping text labels
- produce one clean standalone figure
- work directly from a completed smgr_regression_full directory
- default to labeling only the most important anchor points
- provide a compact right-side label key for the remaining points

Usage:
python scripts/plot_smgr_sidecar_scaling_clean.py \
  --full-dir /foss/designs/gLayout/build/smgr_regression_full \
  --outdir /foss/designs/gLayout/build/smgr_publication_figures \
  --label-mode key

label-mode options:
- key: label only anchor points on the plot, put all others in a side key
- sparse: label a few more points directly
- all: label every point directly (not recommended for dense figures)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SHORT_LABELS = {
    "diff_pair_default": "DP-N",
    "diff_pair_pmos": "DP-P",
    "diff_pair_generic": "DP-G",
    "current_mirror_nfet": "CM-N",
    "current_mirror_pfet": "CM-P",
    "transmission_gate": "TG",
    "flipped_voltage_follower": "FVF",
    "low_voltage_cmirror": "LVCM",
    "differential_to_single_ended_converter": "D2S",
    "diff_pair_ibias": "DPIB",
    "stacked_nfet_current_mirror": "SCM",
    "diff_pair_stackedcmirror_component": "DPSC",
    "row_csamplifier_diff_to_single_ended_converter": "RCS",
    "opamp_twostage": "OTA2",
    "opamp": "OTA",
    "p_block": "PB",
    "n_block": "NB",
    "fvf_based_ota_low_voltage_cmirror": "LV-LCM",
    "super_class_ab_ota": "AB-OTA",
}

FAMILY_COLORS = {
    "Elementary": "#355C9A",
    "Composite": "#B56A1C",
    "System": "#2F6A4F",
}


def family_of(case_id: str) -> str:
    elementary = {
        "diff_pair_default",
        "diff_pair_pmos",
        "diff_pair_generic",
        "current_mirror_nfet",
        "current_mirror_pfet",
        "transmission_gate",
        "flipped_voltage_follower",
    }
    system = {"opamp_twostage", "opamp", "super_class_ab_ota"}
    if case_id in elementary:
        return "Elementary"
    if case_id in system:
        return "System"
    return "Composite"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_cases(full_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(full_dir.rglob("case_result.json")):
        item = json.loads(path.read_text())
        side = item["sidecar_summary"]
        cid = item["case_id"]
        rows.append(
            {
                "case_id": cid,
                "label": SHORT_LABELS.get(cid, cid),
                "family": family_of(cid),
                "call_count": side["call_count"],
                "sidecar_mb": side["sidecar_bytes"] / (1024 * 1024),
                "object_count": side["object_count"],
            }
        )
    if not rows:
        raise ValueError(f"No case_result.json files found under {full_dir}")
    return rows


def save_all_formats(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def annotate_anchor_points(ax, cases: list[dict]) -> None:
    anchor_offsets = {
        "DP-N": (6, -4),
        "DPSC": (6, 6),
        "OTA2": (6, 6),
        "OTA": (6, -4),
        "AB-OTA": (6, 4),
    }
    for c in cases:
        if c["label"] in anchor_offsets:
            dx, dy = anchor_offsets[c["label"]]
            ax.annotate(
                c["label"],
                (c["call_count"], c["sidecar_mb"]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8,
                fontweight="bold",
            )


def annotate_sparse_points(ax, cases: list[dict]) -> None:
    offsets = {
        "DP-N": (6, -4),
        "CM-N": (6, 4),
        "SCM": (6, 4),
        "DPIB": (6, -6),
        "LVCM": (6, 4),
        "LV-LCM": (6, -6),
        "DPSC": (6, 6),
        "NB": (6, 4),
        "OTA2": (6, 6),
        "OTA": (6, -4),
        "AB-OTA": (6, 4),
    }
    for c in cases:
        if c["label"] in offsets:
            dx, dy = offsets[c["label"]]
            ax.annotate(
                c["label"],
                (c["call_count"], c["sidecar_mb"]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=7.5,
            )


def annotate_all_points(ax, cases: list[dict]) -> None:
    for i, c in enumerate(sorted(cases, key=lambda x: (x["call_count"], x["sidecar_mb"]))):
        dx = 4 if i % 2 == 0 else -22
        dy = 4 if i % 3 == 0 else -8 if i % 3 == 1 else 10
        ax.annotate(
            c["label"],
            (c["call_count"], c["sidecar_mb"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.8,
        )


def draw_side_key(fig, ax, cases: list[dict]) -> None:
    ordered = sorted(cases, key=lambda c: c["call_count"])
    lines = [f"{c['label']}: {c['case_id']}" for c in ordered]
    text = "\n".join(lines)
    ax.text(
        1.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.2,
        family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", linewidth=0.8),
    )


def plot(cases: list[dict], outdir: Path, label_mode: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for family, color in FAMILY_COLORS.items():
        family_cases = [c for c in cases if c["family"] == family]
        ax.scatter(
            [c["call_count"] for c in family_cases],
            [c["sidecar_mb"] for c in family_cases],
            s=[16 + 8 * math.sqrt(c["object_count"]) for c in family_cases],
            color=color,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.6,
            label=family,
        )

    xs = np.array([c["call_count"] for c in cases], dtype=float)
    ys = np.array([c["sidecar_mb"] for c in cases], dtype=float)
    coeff = np.polyfit(xs, ys, 1)
    xfit = np.linspace(xs.min(), xs.max(), 200)
    ax.plot(xfit, coeff[0] * xfit + coeff[1], linestyle="--", linewidth=1.0, color="#64748B")

    ax.set_xlabel("Call records")
    ax.set_ylabel("Sidecar size (MiB)")
    ax.set_title("Sidecar Size vs. Generator Depth", loc="left", fontweight="bold")
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, loc="upper left")

    if label_mode == "all":
        annotate_all_points(ax, cases)
    elif label_mode == "sparse":
        annotate_sparse_points(ax, cases)
    else:
        annotate_anchor_points(ax, cases)
        draw_side_key(fig, ax, cases)

    stem = outdir / "fig_sidecar_scaling_clean"
    save_all_formats(fig, stem)
    (outdir / "fig_sidecar_scaling_clean_notes.txt").write_text(
        "This figure compares sidecar size against generator depth (call count). "
        "Only anchor points are labeled directly on-plot to avoid overlap. "
        "The side key maps shorthand labels to full case names.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a clean standalone SMGR sidecar scaling figure.")
    parser.add_argument("--full-dir", required=True, help="Path to smgr_regression_full directory")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--label-mode", choices=["key", "sparse", "all"], default="key")
    args = parser.parse_args()

    configure_style()
    cases = load_cases(Path(args.full_dir).resolve())
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plot(cases, outdir, args.label_mode)
    print(f"Wrote figure to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
