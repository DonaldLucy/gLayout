from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glade.common.schema import flatten_entry


TARGET_LABELS = {
    "geom_area_um2": "Area ($\\mu$m$^2$)",
    "pex_total_resistance_ohms": "Parasitic R (Ohm)",
    "pex_total_capacitance_farads": "Parasitic C (F)",
    "derived_pex_rc_tau_s": "Parasitic RC (s)",
    "runtime_total_s": "Runtime (s)",
    "runtime_build_s": "Build Runtime (s)",
    "runtime_drc_s": "DRC Runtime (s)",
    "runtime_lvs_s": "LVS Runtime (s)",
    "runtime_pex_s": "PEX Runtime (s)",
    "geom_sym_h": "Horizontal Symmetry",
    "geom_sym_v": "Vertical Symmetry",
}

DEFAULT_TARGET_COLUMNS = [
    "geom_area_um2",
    "pex_total_resistance_ohms",
    "pex_total_capacitance_farads",
    "derived_pex_rc_tau_s",
    "runtime_total_s",
]

_JSON_EXPANSION_PREFIXES = {
    "params": "param",
    "geom": "geom",
    "pex": "pex",
    "runtime_s": "runtime",
    "structure_features": "struct",
}

_INPUT_PREFIXES = ("param_", "struct_", "derived_")
_LEAKY_PREFIXES = ("geom_", "pex_", "runtime_", "drc_", "lvs_")
_LEAKY_EXACT = {
    "sample_id",
    "generator_id",
    "pdk",
    "build_status",
    "source_netlist",
    "failure_tags",
    "structure_features",
}


@dataclass
class PreparedDataset:
    frame: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]
    target_columns: list[str]
    selection_report: dict[str, Any]
    source_path: str

    @property
    def input_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features


def _looks_like_nested_entry(obj: Any) -> bool:
    return isinstance(obj, dict) and any(key in obj for key in ("params", "geom", "pex", "runtime_s"))


def _parse_json_cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped in {"nan", "None"}:
        return None
    if stripped[0] not in "{[":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _coerce_boolean_like_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        series = frame[column]
        if series.dtype == bool:
            frame[column] = series.astype(int)
            continue
        values = series.dropna().astype(str).str.lower().unique()
        if len(values) == 0:
            continue
        if set(values).issubset({"true", "false"}):
            frame[column] = series.map(lambda item: str(item).lower() == "true" if pd.notna(item) else np.nan)
            frame[column] = frame[column].astype("float64")
    return frame


def _expand_json_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column, prefix in _JSON_EXPANSION_PREFIXES.items():
        if column not in frame.columns:
            continue
        parsed = frame[column].map(_parse_json_cell)
        if parsed.dropna().empty:
            continue
        first_value = parsed.dropna().iloc[0]
        if isinstance(first_value, dict):
            expanded = pd.json_normalize(parsed).add_prefix(f"{prefix}_")
            frame = pd.concat([frame.drop(columns=[column]), expanded], axis=1)
        elif isinstance(first_value, list) and column == "failure_tags":
            frame[f"{column}_count"] = parsed.map(lambda items: len(items) if isinstance(items, list) else 0)
    if "failure_tags" in frame.columns:
        parsed_tags = frame["failure_tags"].map(_parse_json_cell)
        frame["failure_tags_count"] = parsed_tags.map(lambda items: len(items) if isinstance(items, list) else 0)
    return frame


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
    return [flatten_entry(record) if _looks_like_nested_entry(record) else record for record in records]


def load_dataset_table(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path).expanduser().resolve()
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    elif suffix in {".jsonl", ".json"}:
        frame = pd.DataFrame(_load_json_records(dataset_path))
    elif suffix == ".parquet":
        frame = pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    frame.columns = [str(column) for column in frame.columns]
    frame = _expand_json_columns(frame)
    frame = _coerce_boolean_like_columns(frame)
    return frame


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.astype(float).replace(0.0, np.nan)
    result = numerator.astype(float) / denom
    return result.replace([np.inf, -np.inf], np.nan)


def derive_tabular_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if {"param_width", "param_length"}.issubset(frame.columns):
        width = pd.to_numeric(frame["param_width"], errors="coerce")
        length = pd.to_numeric(frame["param_length"], errors="coerce")
        frame["derived_gate_aspect_ratio"] = _safe_divide(width, length)
        frame["derived_inverse_length_um"] = _safe_divide(pd.Series(np.ones(len(frame))), length)
        frame["derived_log_width"] = np.log1p(width.clip(lower=0.0))
        frame["derived_log_length"] = np.log1p(length.clip(lower=0.0))
    if {"param_width", "param_length", "param_numcols"}.issubset(frame.columns):
        width = pd.to_numeric(frame["param_width"], errors="coerce")
        length = pd.to_numeric(frame["param_length"], errors="coerce")
        numcols = pd.to_numeric(frame["param_numcols"], errors="coerce")
        effective_width = width * numcols
        frame["derived_effective_width_um"] = effective_width
        frame["derived_total_channel_area_proxy_um2"] = width * length * numcols
        frame["derived_conductance_proxy"] = _safe_divide(effective_width, length)
        frame["derived_resistance_proxy"] = _safe_divide(length, effective_width)
    if {"pex_total_resistance_ohms", "pex_total_capacitance_farads"}.issubset(frame.columns):
        resistance = pd.to_numeric(frame["pex_total_resistance_ohms"], errors="coerce")
        capacitance = pd.to_numeric(frame["pex_total_capacitance_farads"], errors="coerce")
        frame["derived_pex_rc_tau_s"] = resistance * capacitance
    if "runtime_total_s" not in frame.columns:
        runtime_columns = [
            column
            for column in ("runtime_build_s", "runtime_drc_s", "runtime_lvs_s", "runtime_pex_s")
            if column in frame.columns
        ]
        if runtime_columns:
            runtime_table = frame[runtime_columns].apply(pd.to_numeric, errors="coerce")
            frame["runtime_total_s"] = runtime_table.sum(axis=1)
    return frame


def _is_constant(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return True
    return values.nunique(dropna=True) <= 1


def _non_null_fraction(series: pd.Series) -> float:
    return float(series.notna().mean())


def _display_name(column: str) -> str:
    return TARGET_LABELS.get(column, column)


def _target_is_informative(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return False
    if values.nunique() <= 1:
        return False
    if values.std(ddof=0) <= 1e-12:
        return False
    return True


def select_target_columns(frame: pd.DataFrame, explicit_targets: list[str] | None = None) -> list[str]:
    candidates = explicit_targets or DEFAULT_TARGET_COLUMNS + ["geom_sym_h", "geom_sym_v"]
    selected: list[str] = []
    for column in candidates:
        if column not in frame.columns:
            continue
        if not _target_is_informative(frame[column]):
            continue
        selected.append(column)
    if not selected:
        available = [column for column in frame.columns if column.startswith(_LEAKY_PREFIXES)]
        raise ValueError(
            "No usable target columns were found. "
            f"Available quality/performance columns: {available}"
        )
    return selected


def _sort_feature_name(name: str) -> tuple[int, str]:
    prefix_order = {"param_": 0, "derived_": 1, "struct_": 2}
    for prefix, order in prefix_order.items():
        if name.startswith(prefix):
            return order, name
    return 99, name


def select_input_columns(frame: pd.DataFrame, target_columns: list[str]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    numeric: list[str] = []
    categorical: list[str] = []
    dropped: dict[str, list[str]] = {
        "identifier_or_leakage": [],
        "constant_or_empty": [],
        "missingness": [],
        "redundant": [],
    }

    has_param_device = "param_device" in frame.columns
    for column in frame.columns:
        if column in target_columns:
            continue
        if column in _LEAKY_EXACT or column.startswith(_LEAKY_PREFIXES):
            dropped["identifier_or_leakage"].append(column)
            continue
        if not column.startswith(_INPUT_PREFIXES):
            continue
        if has_param_device and column.startswith("struct_device_is_"):
            dropped["redundant"].append(column)
            continue
        series = frame[column]
        if _non_null_fraction(series) < 0.9:
            dropped["missingness"].append(column)
            continue
        if _is_constant(series):
            dropped["constant_or_empty"].append(column)
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
        else:
            categorical.append(column)

    numeric = sorted(numeric, key=_sort_feature_name)
    categorical = sorted(categorical, key=_sort_feature_name)
    return numeric, categorical, dropped


def _build_stratification_labels(frame: pd.DataFrame) -> pd.Series | None:
    numcols = frame["param_numcols"].astype(str) if "param_numcols" in frame.columns else pd.Series(["0"] * len(frame))
    if "derived_total_channel_area_proxy_um2" in frame.columns:
        complexity = pd.to_numeric(frame["derived_total_channel_area_proxy_um2"], errors="coerce")
    elif {"param_width", "param_length"}.issubset(frame.columns):
        complexity = pd.to_numeric(frame["param_width"], errors="coerce") * pd.to_numeric(
            frame["param_length"], errors="coerce"
        )
    else:
        complexity = None
    if complexity is None or complexity.dropna().nunique() < 4:
        labels = numcols
    else:
        valid = complexity.fillna(complexity.median())
        quantiles = min(5, int(valid.nunique()))
        bins = pd.qcut(valid.rank(method="first"), q=quantiles, labels=False, duplicates="drop").astype(str)
        labels = numcols + "__q" + bins
    if labels.value_counts().min() < 2:
        return None
    return labels


def assign_splits(
    frame: pd.DataFrame,
    eval_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    split_frame = frame.copy()
    labels = _build_stratification_labels(split_frame)
    indices = split_frame.index.to_numpy()
    train_val_idx, eval_idx = train_test_split(
        indices,
        test_size=eval_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels.loc[indices] if labels is not None else None,
    )

    if val_size > 0:
        inner_labels = labels.loc[train_val_idx] if labels is not None else None
        adjusted_val_size = val_size / (1.0 - eval_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=adjusted_val_size,
            random_state=random_state,
            shuffle=True,
            stratify=inner_labels if inner_labels is not None else None,
        )
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=int)

    split_frame["split"] = "train"
    split_frame.loc[val_idx, "split"] = "val"
    split_frame.loc[eval_idx, "split"] = "eval"
    return split_frame


def prepare_current_mirror_dataset(
    dataset_path: str | Path,
    explicit_targets: list[str] | None = None,
    eval_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> PreparedDataset:
    raw_frame = derive_tabular_features(load_dataset_table(dataset_path))
    target_columns = select_target_columns(raw_frame, explicit_targets)
    numeric_features, categorical_features, dropped = select_input_columns(raw_frame, target_columns)
    required_columns = numeric_features + categorical_features + target_columns
    if "sample_id" in raw_frame.columns:
        required_columns = ["sample_id"] + required_columns
    model_frame = raw_frame[required_columns].copy()
    model_frame = model_frame.dropna(subset=target_columns).reset_index(drop=True)
    model_frame = assign_splits(model_frame, eval_size=eval_size, val_size=val_size, random_state=random_state)

    selection_report = {
        "source_path": str(Path(dataset_path).expanduser().resolve()),
        "rows_after_target_filtering": int(len(model_frame)),
        "split_counts": model_frame["split"].value_counts().to_dict(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_columns": target_columns,
        "target_labels": {column: _display_name(column) for column in target_columns},
        "dropped_columns": dropped,
    }

    return PreparedDataset(
        frame=model_frame,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_columns=target_columns,
        selection_report=selection_report,
        source_path=str(Path(dataset_path).expanduser().resolve()),
    )


def write_selection_report_markdown(prepared: PreparedDataset, path: str | Path) -> None:
    report = prepared.selection_report
    lines = [
        "# Current Mirror Surrogate Dataset Profile",
        "",
        f"- Source dataset: `{report['source_path']}`",
        f"- Rows used for modeling: `{report['rows_after_target_filtering']}`",
        f"- Split counts: `{report['split_counts']}`",
        "",
        "## Selected Inputs",
        "",
        f"- Numeric: `{', '.join(report['numeric_features']) or 'None'}`",
        f"- Categorical: `{', '.join(report['categorical_features']) or 'None'}`",
        "",
        "## Selected Targets",
        "",
    ]
    for column in prepared.target_columns:
        lines.append(f"- `{column}` -> {TARGET_LABELS.get(column, column)}")
    lines.extend(["", "## Dropped Columns", ""])
    for group, columns in report["dropped_columns"].items():
        lines.append(f"- {group}: `{', '.join(columns) or 'None'}`")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
