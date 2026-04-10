from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    centered = y_true - y_true.mean()
    denom = float(np.sum(centered * centered))
    if denom <= 1e-18:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = np.maximum(np.abs(y_true), 1e-12)
    mape = float(np.mean(np.abs(error) / denom) * 100.0)
    smape = float(np.mean((2.0 * np.abs(error)) / (np.abs(y_true) + np.abs(y_pred) + 1e-12)) * 100.0)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": _safe_r2(y_true, y_pred),
        "mape": mape,
        "smape": smape,
    }


def metrics_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: Iterable[str],
    model_name: str,
) -> pd.DataFrame:
    rows = []
    for index, target in enumerate(target_columns):
        stats = regression_metrics(y_true[:, index], y_pred[:, index])
        stats.update({"model": model_name, "target": target})
        rows.append(stats)
    return pd.DataFrame(rows)


def long_form_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: Iterable[str],
    model_name: str,
    sample_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    sample_list = list(sample_ids) if sample_ids is not None else [str(index) for index in range(len(y_true))]
    rows = []
    for row_index, sample_id in enumerate(sample_list):
        for target_index, target in enumerate(target_columns):
            actual = float(y_true[row_index, target_index])
            predicted = float(y_pred[row_index, target_index])
            abs_pct_error = abs(predicted - actual) / max(abs(actual), 1e-12) * 100.0
            rows.append(
                {
                    "sample_id": sample_id,
                    "model": model_name,
                    "target": target,
                    "actual": actual,
                    "predicted": predicted,
                    "residual": predicted - actual,
                    "abs_pct_error": abs_pct_error,
                }
            )
    return pd.DataFrame(rows)
