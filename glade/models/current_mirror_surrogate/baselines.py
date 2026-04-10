from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_baseline_models(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_columns: list[str],
    random_state: int,
) -> dict[str, dict[str, object]]:
    x_train = train_frame[numeric_features + categorical_features]
    x_eval = eval_frame[numeric_features + categorical_features]
    y_train = train_frame[target_columns].to_numpy(dtype=float)
    y_eval = eval_frame[target_columns].to_numpy(dtype=float)

    model_specs = {
        "Ridge": Pipeline(
            [
                ("preprocess", _build_preprocessor(numeric_features, categorical_features, scale_numeric=True)),
                ("model", Ridge(alpha=1.5)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("preprocess", _build_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=700,
                        min_samples_leaf=1,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            [
                ("preprocess", _build_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=1000,
                        min_samples_leaf=1,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    outputs: dict[str, dict[str, object]] = {}
    for model_name, pipeline in model_specs.items():
        pipeline.fit(x_train, y_train)
        outputs[model_name] = {
            "pipeline": pipeline,
            "eval_predictions": pipeline.predict(x_eval),
            "eval_targets": y_eval,
        }
    return outputs


def save_baselines(artifacts: dict[str, dict[str, object]], path: str | Path) -> None:
    serializable = {name: payload["pipeline"] for name, payload in artifacts.items()}
    with open(path, "wb") as handle:
        pickle.dump(serializable, handle)
