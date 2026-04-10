from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import fit_baseline_models, save_baselines
from .data import PreparedDataset, TARGET_LABELS, prepare_current_mirror_dataset, write_selection_report_markdown
from .ft_transformer import FTTransformer, FTTransformerConfig
from .metrics import long_form_predictions, metrics_frame
from .plotting import (
    plot_learning_curve,
    plot_model_comparison,
    plot_parity_panels,
    plot_relative_error_boxplot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train current-mirror surrogate models on tabular benchmark data.")
    parser.add_argument("--dataset", default=None, help="Path to dataset CSV, JSONL, JSON, or Parquet file.")
    parser.add_argument(
        "--output-root",
        default="output/glade/current_mirror_surrogate",
        help="Root directory for run artifacts.",
    )
    parser.add_argument("--run-name", default=None, help="Optional run-name override.")
    parser.add_argument("--targets", nargs="*", default=None, help="Explicit target columns to model.")
    parser.add_argument("--eval-size", type=float, default=0.2, help="Hold-out evaluation fraction.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation fraction for early stopping.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--profile-only", action="store_true", help="Stop after dataset profiling.")
    parser.add_argument("--skip-ft", action="store_true", help="Skip FT-Transformer training.")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip traditional baseline models.")
    parser.add_argument("--device", default="auto", help="Torch device override, for example cuda or cpu.")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=4, help="Torch dataloader workers.")
    parser.add_argument("--ft-d-token", type=int, default=192)
    parser.add_argument("--ft-heads", type=int, default=8)
    parser.add_argument("--ft-blocks", type=int, default=6)
    parser.add_argument("--ft-ffn-mult", type=float, default=4.0)
    parser.add_argument("--ft-attn-dropout", type=float, default=0.10)
    parser.add_argument("--ft-ff-dropout", type=float, default=0.15)
    parser.add_argument("--ft-residual-dropout", type=float, default=0.05)
    parser.add_argument("--ft-token-dropout", type=float, default=0.05)
    parser.add_argument("--ft-batch-size", type=int, default=512)
    parser.add_argument("--ft-max-epochs", type=int, default=250)
    parser.add_argument("--ft-patience", type=int, default=30)
    parser.add_argument("--ft-lr", type=float, default=1e-3)
    parser.add_argument("--ft-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--ft-max-train-seconds",
        type=int,
        default=3300,
        help="Hard stop for FT-Transformer wall-clock training time.",
    )
    return parser.parse_args()


def find_default_dataset() -> Path:
    candidates = [
        Path("output/glade/current_mirror/dataset.csv"),
        Path("output/glade/current_mirror/dataset.jsonl"),
        Path("output/glade/current_mirror/dataset.parquet"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "No dataset path was supplied and no default current_mirror dataset file was found under "
        "output/glade/current_mirror/."
    )


def build_run_dir(output_root: str | Path, run_name: str | None) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved_name = run_name or time.strftime("cm_surrogate_%Y%m%d_%H%M%S")
    run_dir = root / resolved_name
    if run_dir.exists():
        suffix = 1
        while True:
            candidate = root / f"{resolved_name}_{suffix:02d}"
            if not candidate.exists():
                run_dir = candidate
                break
            suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    return run_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class NumericScaler:
    columns: list[str]
    mean: dict[str, float]
    std: dict[str, float]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.columns:
            return np.zeros((len(frame), 0), dtype=np.float32)
        values = frame[self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        means = np.array([self.mean[column] for column in self.columns], dtype=np.float32)
        stds = np.array([self.std[column] for column in self.columns], dtype=np.float32)
        standardized = (values - means) / stds
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
        return standardized.astype(np.float32)


def fit_numeric_scaler(frame: pd.DataFrame, columns: list[str]) -> NumericScaler:
    stats_mean: dict[str, float] = {}
    stats_std: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        stats_mean[column] = mean
        stats_std[column] = std if std > 1e-8 else 1.0
    return NumericScaler(columns=columns, mean=stats_mean, std=stats_std)


@dataclass
class CategoricalEncoder:
    columns: list[str]
    mapping: dict[str, dict[str, int]]

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.mapping[column]) + 1 for column in self.columns]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.columns:
            return np.zeros((len(frame), 0), dtype=np.int64)
        encoded_columns = []
        for column in self.columns:
            lookup = self.mapping[column]
            values = frame[column].fillna("__MISSING__").astype(str).map(lookup).fillna(0).astype(np.int64)
            encoded_columns.append(values.to_numpy()[:, None])
        return np.concatenate(encoded_columns, axis=1)


def fit_categorical_encoder(frame: pd.DataFrame, columns: list[str]) -> CategoricalEncoder:
    mapping: dict[str, dict[str, int]] = {}
    for column in columns:
        values = frame[column].fillna("__MISSING__").astype(str)
        categories = sorted(values.unique().tolist())
        mapping[column] = {category: index + 1 for index, category in enumerate(categories)}
    return CategoricalEncoder(columns=columns, mapping=mapping)


def _should_log_transform_target(column: str, values: np.ndarray) -> bool:
    if np.nanmin(values) < 0:
        return False
    if column.startswith("pex_") or column.startswith("runtime_") or column == "geom_area_um2" or column.endswith("_tau_s"):
        return True
    return False


@dataclass
class TargetTransformer:
    columns: list[str]
    log_columns: list[str]
    mean: dict[str, float]
    std: dict[str, float]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = []
        for column in self.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if column in self.log_columns:
                values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
            values = (values - self.mean[column]) / self.std[column]
            transformed.append(values[:, None])
        return np.concatenate(transformed, axis=1).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        restored = np.zeros_like(values, dtype=np.float64)
        for index, column in enumerate(self.columns):
            column_values = values[:, index] * self.std[column] + self.mean[column]
            if column in self.log_columns:
                column_values = np.expm1(column_values)
            restored[:, index] = column_values
        return restored


def fit_target_transformer(frame: pd.DataFrame, columns: list[str]) -> TargetTransformer:
    log_columns: list[str] = []
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if _should_log_transform_target(column, values):
            log_columns.append(column)
            values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
        mean[column] = float(np.nanmean(values))
        col_std = float(np.nanstd(values))
        std[column] = col_std if col_std > 1e-8 else 1.0
    return TargetTransformer(columns=columns, log_columns=log_columns, mean=mean, std=std)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_run_report(
    prepared: PreparedDataset,
    metrics_df: pd.DataFrame,
    run_dir: Path,
    transformer_config: dict[str, object] | None,
) -> None:
    lines = [
        "# Current Mirror Surrogate Report",
        "",
        f"- Dataset: `{prepared.source_path}`",
        f"- Rows modeled: `{len(prepared.frame)}`",
        f"- Split counts: `{prepared.selection_report['split_counts']}`",
        f"- Inputs: `{', '.join(prepared.input_columns)}`",
        f"- Targets: `{', '.join(prepared.target_columns)}`",
        "",
    ]
    if transformer_config is not None:
        lines.extend(
            [
                "## FT-Transformer Configuration",
                "",
                f"- `d_token={transformer_config['d_token']}`, `heads={transformer_config['heads']}`, `blocks={transformer_config['blocks']}`",
                f"- `batch_size={transformer_config['batch_size']}`, `lr={transformer_config['lr']}`, `weight_decay={transformer_config['weight_decay']}`",
                f"- `amp_dtype={transformer_config['amp_dtype']}`, `max_train_seconds={transformer_config['max_train_seconds']}`",
                "",
            ]
        )
    lines.extend(["## Metrics", ""])
    for model_name, model_df in metrics_df.groupby("model", sort=False):
        mean_r2 = model_df["r2"].mean()
        mean_mape = model_df["mape"].mean()
        lines.append(f"- {model_name}: mean R² = {mean_r2:.3f}, mean MAPE = {mean_mape:.2f}%")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _autocast_context(torch, device: str, amp_dtype: str):
    if not device.startswith("cuda") or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _build_scheduler(torch, optimizer, max_epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


def _build_grad_scaler(torch, device: str, amp_dtype: str):
    enabled = device.startswith("cuda") and amp_dtype == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_ft_transformer(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    prepared: PreparedDataset,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.random_state)
    torch.manual_seed(args.random_state)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.random_state)

    if val_frame.empty:
        val_frame = train_frame.copy()

    numeric_scaler = fit_numeric_scaler(train_frame, prepared.numeric_features)
    categorical_encoder = fit_categorical_encoder(train_frame, prepared.categorical_features)
    target_transformer = fit_target_transformer(train_frame, prepared.target_columns)

    x_train_num = numeric_scaler.transform(train_frame)
    x_val_num = numeric_scaler.transform(val_frame)
    x_eval_num = numeric_scaler.transform(eval_frame)

    x_train_cat = categorical_encoder.transform(train_frame)
    x_val_cat = categorical_encoder.transform(val_frame)
    x_eval_cat = categorical_encoder.transform(eval_frame)

    y_train = target_transformer.transform(train_frame)
    y_val = target_transformer.transform(val_frame)

    class ArrayDataset(Dataset):
        def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray):
            self.x_num = torch.from_numpy(x_num)
            self.x_cat = torch.from_numpy(x_cat)
            self.y = torch.from_numpy(y)

        def __len__(self) -> int:
            return self.y.shape[0]

        def __getitem__(self, index: int):
            return self.x_num[index], self.x_cat[index], self.y[index]

    train_loader = DataLoader(
        ArrayDataset(x_train_num, x_train_cat, y_train),
        batch_size=args.ft_batch_size,
        shuffle=True,
        num_workers=args.num_workers if device.startswith("cuda") else 0,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        ArrayDataset(x_val_num, x_val_cat, y_val),
        batch_size=args.ft_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    config = FTTransformerConfig(
        num_numeric_features=len(prepared.numeric_features),
        categorical_cardinalities=categorical_encoder.cardinalities,
        output_dim=len(prepared.target_columns),
        d_token=args.ft_d_token,
        n_blocks=args.ft_blocks,
        attention_n_heads=args.ft_heads,
        attention_dropout=args.ft_attn_dropout,
        ffn_dropout=args.ft_ff_dropout,
        residual_dropout=args.ft_residual_dropout,
        token_dropout=args.ft_token_dropout,
        ffn_hidden_multiplier=args.ft_ffn_mult,
    )
    model = FTTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=args.ft_weight_decay)
    scheduler = _build_scheduler(torch, optimizer, args.ft_max_epochs)
    criterion = nn.HuberLoss(delta=0.75)
    scaler = _build_grad_scaler(torch, device, args.amp_dtype)

    history_rows = []
    best_state = None
    best_val_loss = math.inf
    best_epoch = 0
    start_time = time.time()

    def evaluate(loader: DataLoader) -> float:
        losses = []
        model.eval()
        with torch.no_grad():
            for x_num_batch, x_cat_batch, y_batch in loader:
                x_num_batch = x_num_batch.to(device)
                x_cat_batch = x_cat_batch.to(device)
                y_batch = y_batch.to(device)
                with _autocast_context(torch, device, args.amp_dtype):
                    prediction = model(x_num_batch, x_cat_batch)
                    loss = criterion(prediction.float(), y_batch.float())
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else math.inf

    for epoch in range(1, args.ft_max_epochs + 1):
        if time.time() - start_time > args.ft_max_train_seconds:
            break
        model.train()
        batch_losses = []
        for x_num_batch, x_cat_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x_num_batch = x_num_batch.to(device)
            x_cat_batch = x_cat_batch.to(device)
            y_batch = y_batch.to(device)
            with _autocast_context(torch, device, args.amp_dtype):
                prediction = model(x_num_batch, x_cat_batch)
                loss = criterion(prediction.float(), y_batch.float())
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        scheduler.step()
        train_loss = float(np.mean(batch_losses)) if batch_losses else math.inf
        val_loss = evaluate(val_loader)
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        if epoch - best_epoch >= args.ft_patience:
            break

    if best_state is None:
        raise RuntimeError("FT-Transformer training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    def predict_table(x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        if len(x_num) == 0:
            return np.zeros((0, len(prepared.target_columns)), dtype=np.float32)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x_num), args.ft_batch_size):
                stop = start + args.ft_batch_size
                x_num_batch = torch.from_numpy(x_num[start:stop]).to(device)
                x_cat_batch = torch.from_numpy(x_cat[start:stop]).to(device)
                with _autocast_context(torch, device, args.amp_dtype):
                    prediction = model(x_num_batch, x_cat_batch)
                outputs.append(prediction.detach().float().cpu().numpy())
        stacked = np.concatenate(outputs, axis=0)
        return target_transformer.inverse_transform(stacked)

    eval_prediction = predict_table(x_eval_num, x_eval_cat)
    eval_target = eval_frame[prepared.target_columns].to_numpy(dtype=float)

    metrics_df = metrics_frame(eval_target, eval_prediction, prepared.target_columns, "FT-Transformer")
    predictions_df = long_form_predictions(
        eval_target,
        eval_prediction,
        prepared.target_columns,
        "FT-Transformer",
        sample_ids=eval_frame["sample_id"].tolist() if "sample_id" in eval_frame.columns else None,
    )
    history_df = pd.DataFrame(history_rows)

    checkpoint = {
        "model_state_dict": best_state,
        "config": asdict(config),
        "numeric_scaler": asdict(numeric_scaler),
        "categorical_encoder": {
            "columns": categorical_encoder.columns,
            "mapping": categorical_encoder.mapping,
            "cardinalities": categorical_encoder.cardinalities,
        },
        "target_transformer": asdict(target_transformer),
        "target_columns": prepared.target_columns,
        "numeric_features": prepared.numeric_features,
        "categorical_features": prepared.categorical_features,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "parameter_count": model.parameter_count,
    }
    torch.save(checkpoint, run_dir / "models" / "ft_transformer_best.pt")
    history_df.to_csv(run_dir / "training_history_ft_transformer.csv", index=False)
    return metrics_df, predictions_df, history_df, checkpoint


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else find_default_dataset()
    run_dir = build_run_dir(args.output_root, args.run_name)
    prepared = prepare_current_mirror_dataset(
        dataset_path=dataset_path,
        explicit_targets=args.targets,
        eval_size=args.eval_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    write_json(run_dir / "selection_report.json", prepared.selection_report)
    write_selection_report_markdown(prepared, run_dir / "dataset_profile.md")
    prepared.frame.to_csv(run_dir / "prepared_dataset_with_splits.csv", index=False)
    write_json(
        run_dir / "run_config.json",
        {
            "dataset": str(dataset_path),
            "targets": prepared.target_columns,
            "random_state": args.random_state,
            "eval_size": args.eval_size,
            "val_size": args.val_size,
            "skip_ft": args.skip_ft,
            "skip_baselines": args.skip_baselines,
            "ft_config": {
                "d_token": args.ft_d_token,
                "heads": args.ft_heads,
                "blocks": args.ft_blocks,
                "ffn_multiplier": args.ft_ffn_mult,
                "attention_dropout": args.ft_attn_dropout,
                "ff_dropout": args.ft_ff_dropout,
                "residual_dropout": args.ft_residual_dropout,
                "token_dropout": args.ft_token_dropout,
                "batch_size": args.ft_batch_size,
                "max_epochs": args.ft_max_epochs,
                "patience": args.ft_patience,
                "lr": args.ft_lr,
                "weight_decay": args.ft_weight_decay,
                "max_train_seconds": args.ft_max_train_seconds,
                "amp_dtype": args.amp_dtype,
            },
        },
    )

    if args.profile_only:
        print(f"Dataset profile written under {run_dir}")
        return 0

    train_frame = prepared.frame[prepared.frame["split"] == "train"].reset_index(drop=True)
    val_frame = prepared.frame[prepared.frame["split"] == "val"].reset_index(drop=True)
    eval_frame = prepared.frame[prepared.frame["split"] == "eval"].reset_index(drop=True)

    metrics_tables = []
    prediction_tables = []
    history_df = pd.DataFrame()
    checkpoint_summary = None

    if not args.skip_ft:
        ft_metrics, ft_predictions, history_df, checkpoint = train_ft_transformer(
            train_frame=train_frame,
            val_frame=val_frame,
            eval_frame=eval_frame,
            prepared=prepared,
            args=args,
            run_dir=run_dir,
        )
        metrics_tables.append(ft_metrics)
        prediction_tables.append(ft_predictions)
        checkpoint_summary = {
            "d_token": args.ft_d_token,
            "heads": args.ft_heads,
            "blocks": args.ft_blocks,
            "batch_size": args.ft_batch_size,
            "lr": args.ft_lr,
            "weight_decay": args.ft_weight_decay,
            "amp_dtype": args.amp_dtype,
            "max_train_seconds": args.ft_max_train_seconds,
            "parameter_count": checkpoint["parameter_count"],
            "best_epoch": checkpoint["best_epoch"],
            "best_val_loss": checkpoint["best_val_loss"],
        }

    if not args.skip_baselines:
        baseline_artifacts = fit_baseline_models(
            train_frame=train_frame,
            eval_frame=eval_frame,
            numeric_features=prepared.numeric_features,
            categorical_features=prepared.categorical_features,
            target_columns=prepared.target_columns,
            random_state=args.random_state,
        )
        save_baselines(baseline_artifacts, run_dir / "models" / "baselines.pkl")
        for model_name, payload in baseline_artifacts.items():
            eval_prediction = payload["eval_predictions"]
            eval_target = payload["eval_targets"]
            metrics_tables.append(metrics_frame(eval_target, eval_prediction, prepared.target_columns, model_name))
            prediction_tables.append(
                long_form_predictions(
                    eval_target,
                    eval_prediction,
                    prepared.target_columns,
                    model_name,
                    sample_ids=eval_frame["sample_id"].tolist() if "sample_id" in eval_frame.columns else None,
                )
            )

    if not metrics_tables:
        raise RuntimeError("Nothing was trained. Remove --skip-ft and/or --skip-baselines.")

    metrics_df = pd.concat(metrics_tables, ignore_index=True)
    predictions_df = pd.concat(prediction_tables, ignore_index=True)
    metrics_df.to_csv(run_dir / "metrics_per_target.csv", index=False)
    predictions_df.to_csv(run_dir / "predictions_eval.csv", index=False)

    aggregate_df = (
        metrics_df.groupby("model", as_index=False)[["r2", "mape", "mae", "rmse", "smape"]]
        .mean()
        .sort_values("r2", ascending=False)
    )
    aggregate_df.to_csv(run_dir / "metrics_aggregate.csv", index=False)
    write_json(
        run_dir / "metrics_summary.json",
        {
            "best_model_by_mean_r2": aggregate_df.iloc[0]["model"],
            "targets": {column: TARGET_LABELS.get(column, column) for column in prepared.target_columns},
            "aggregate_metrics": aggregate_df.to_dict(orient="records"),
        },
    )
    build_run_report(prepared, metrics_df, run_dir, checkpoint_summary)

    figure_dir = run_dir / "figures"
    plot_model_comparison(metrics_df, figure_dir)
    if "FT-Transformer" in metrics_df["model"].unique():
        plot_parity_panels(predictions_df, metrics_df, "FT-Transformer", figure_dir)
        plot_relative_error_boxplot(predictions_df, "FT-Transformer", figure_dir)
        plot_learning_curve(history_df, figure_dir)

    print(f"Run completed. Artifacts written under {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
