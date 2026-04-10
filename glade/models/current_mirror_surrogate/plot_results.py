from __future__ import annotations

import argparse
from pathlib import Path

from .plotting import (
    load_run_artifacts,
    plot_learning_curve,
    plot_model_comparison,
    plot_parity_panels,
    plot_relative_error_boxplot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate surrogate-model evaluation plots from a completed run.")
    parser.add_argument("--run-dir", required=True, help="Path to a completed training run directory.")
    parser.add_argument(
        "--parity-model",
        default="FT-Transformer",
        help="Which model to use for parity and residual figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = load_run_artifacts(args.run_dir)
    figure_dir = Path(args.run_dir) / "figures"
    plot_model_comparison(artifacts["metrics"], figure_dir)
    plot_parity_panels(artifacts["predictions"], artifacts["metrics"], args.parity_model, figure_dir)
    plot_relative_error_boxplot(artifacts["predictions"], args.parity_model, figure_dir)
    plot_learning_curve(artifacts["history"], figure_dir)
    print(f"Figures regenerated under {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
