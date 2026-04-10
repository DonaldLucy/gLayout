from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import TARGET_LABELS


MODEL_PALETTE = {
    "FT-Transformer": "#0B3954",
    "ExtraTrees": "#087E8B",
    "RandomForest": "#B35D3E",
    "Ridge": "#C1A57B",
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.facecolor": "#FBFBF8",
            "figure.facecolor": "#F4F1EA",
            "savefig.facecolor": "#F4F1EA",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
        }
    )


def _target_label(target: str) -> str:
    return TARGET_LABELS.get(target, target)


def save_figure_bundle(fig: plt.Figure, stem: str | Path) -> None:
    target = Path(stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(target.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _model_order(models: list[str]) -> list[str]:
    ordered = [model for model in ("FT-Transformer", "ExtraTrees", "RandomForest", "Ridge") if model in models]
    ordered.extend(sorted(model for model in models if model not in ordered))
    return ordered


def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: str | Path) -> None:
    apply_plot_style()
    models = _model_order(metrics_df["model"].unique().tolist())
    targets = metrics_df["target"].drop_duplicates().tolist()

    r2_table = metrics_df.pivot(index="model", columns="target", values="r2").reindex(index=models, columns=targets)
    mape_table = metrics_df.pivot(index="model", columns="target", values="mape").reindex(index=models, columns=targets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    for axis, table, cmap, title, formatter in (
        (axes[0], r2_table, "YlGnBu", "R² on Hold-out Evaluation Set", "{:.3f}"),
        (axes[1], mape_table, "YlOrRd_r", "MAPE (%) on Hold-out Evaluation Set", "{:.1f}"),
    ):
        matrix = table.to_numpy(dtype=float)
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(targets)), [_target_label(target) for target in targets], rotation=25, ha="right")
        axis.set_yticks(range(len(models)), models)
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if np.isnan(value):
                    text = "NA"
                else:
                    text = formatter.format(value)
                color = "white" if np.isfinite(value) and value > np.nanmean(matrix) else "#1A1A1A"
                axis.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save_figure_bundle(fig, Path(output_dir) / "comparison_heatmaps")


def plot_parity_panels(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
) -> None:
    apply_plot_style()
    subset = predictions_df[predictions_df["model"] == model_name].copy()
    if subset.empty:
        return
    targets = subset["target"].drop_duplicates().tolist()
    num_targets = len(targets)
    ncols = 3
    nrows = math.ceil(num_targets / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 4.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for axis in axes.ravel():
        axis.set_visible(False)

    for axis, target in zip(axes.ravel(), targets):
        axis.set_visible(True)
        target_df = subset[subset["target"] == target]
        stats = metrics_df[(metrics_df["model"] == model_name) & (metrics_df["target"] == target)].iloc[0]
        color = MODEL_PALETTE.get(model_name, "#0B3954")
        actual = target_df["actual"].to_numpy(dtype=float)
        predicted = target_df["predicted"].to_numpy(dtype=float)
        axis.scatter(actual, predicted, s=22, alpha=0.75, c=color, edgecolors="none")
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.0, color="#222222")
        if lower > 0 and ("pex_" in target or "runtime_" in target or target in {"geom_area_um2", "derived_pex_rc_tau_s"}):
            axis.set_xscale("log")
            axis.set_yscale("log")
        axis.set_title(_target_label(target))
        axis.set_xlabel("Measured")
        axis.set_ylabel("Predicted")
        axis.text(
            0.03,
            0.97,
            f"R² = {stats['r2']:.3f}\nMAPE = {stats['mape']:.1f}%",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
    fig.suptitle(f"{model_name} Parity Plots", y=1.01, fontsize=14)
    save_figure_bundle(fig, Path(output_dir) / f"{model_name.lower().replace('-', '_').replace(' ', '_')}_parity")


def plot_relative_error_boxplot(predictions_df: pd.DataFrame, model_name: str, output_dir: str | Path) -> None:
    apply_plot_style()
    subset = predictions_df[predictions_df["model"] == model_name].copy()
    if subset.empty:
        return
    targets = subset["target"].drop_duplicates().tolist()
    data = [
        np.clip(subset.loc[subset["target"] == target, "abs_pct_error"].to_numpy(dtype=float), 1e-4, None)
        for target in targets
    ]

    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    box = axis.boxplot(
        data,
        vert=False,
        patch_artist=True,
        labels=[_target_label(target) for target in targets],
        showfliers=False,
    )
    for patch, target in zip(box["boxes"], targets):
        patch.set_facecolor(MODEL_PALETTE.get(model_name, "#0B3954"))
        patch.set_alpha(0.65)
        patch.set_edgecolor("#202020")
    axis.set_title(f"{model_name} Absolute Percentage Error Distribution")
    axis.set_xlabel("Absolute Percentage Error (%)")
    axis.set_xscale("log")
    save_figure_bundle(fig, Path(output_dir) / f"{model_name.lower().replace('-', '_').replace(' ', '_')}_relative_error")


def plot_learning_curve(history_df: pd.DataFrame, output_dir: str | Path) -> None:
    if history_df.empty:
        return
    apply_plot_style()
    fig, axis = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    axis.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", linewidth=2.0, color="#0B3954")
    axis.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", linewidth=2.0, color="#B35D3E")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Huber Loss (standardized targets)")
    axis.set_title("FT-Transformer Learning Curve")
    axis.legend()
    save_figure_bundle(fig, Path(output_dir) / "ft_transformer_learning_curve")


def load_run_artifacts(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir)
    metrics_df = pd.read_csv(root / "metrics_per_target.csv")
    predictions_df = pd.read_csv(root / "predictions_eval.csv")
    history_path = root / "training_history_ft_transformer.csv"
    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    profile = json.loads((root / "selection_report.json").read_text(encoding="utf-8"))
    return {
        "metrics": metrics_df,
        "predictions": predictions_df,
        "history": history_df,
        "profile": profile,
    }
