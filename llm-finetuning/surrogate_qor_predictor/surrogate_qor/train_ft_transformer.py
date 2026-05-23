from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("USER", os.environ.get("LOGNAME", "codex"))
os.environ.setdefault("LOGNAME", os.environ["USER"])
os.environ.setdefault("HOME", f"/tmp/{os.environ['USER']}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/{os.environ['USER']}/.cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/{os.environ['USER']}/torchinductor")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLS_TARGETS = ("drc_pass", "lvs_pass", "pex_pass")
REG_TARGETS = ("area_um2", "total_resistance_ohms", "total_capacitance_farads", "runtime_s")


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _flatten(f"{prefix}.{index}", child, out)
    else:
        out[prefix] = value


def _get_path(record: dict[str, Any], path: str) -> Any:
    node: Any = record
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(float(value))


def load_records(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_rows is not None and len(records) >= max_rows:
                break
    return records


def make_feature_row(record: dict[str, Any], include_verification_summary: bool = False) -> tuple[dict[str, float], dict[str, str]]:
    numeric: dict[str, float] = {}
    categorical: dict[str, str] = {
        "generator_id": str(record.get("generator_id", "")),
        "corpus": str(record.get("corpus", "")),
        "family": str(record.get("family", "")),
        "cost": str(record.get("cost", "")),
        "fixed": str(bool(record.get("fixed", False))),
    }

    flat_params: dict[str, Any] = {}
    _flatten("param", record.get("params", {}), flat_params)
    for key, value in flat_params.items():
        if isinstance(value, bool):
            categorical[key] = str(value)
        elif _is_number(value):
            numeric[key] = float(value)
        elif value is not None:
            categorical[key] = str(value)

    flat_code: dict[str, Any] = {}
    _flatten("code", record.get("code_features", {}) or record.get("features", {}).get("code", {}), flat_code)
    for key, value in flat_code.items():
        if _is_number(value):
            numeric[key] = float(value)

    # These are available after cheap generation/GDS emission, before DRC/LVS.
    pre_paths = {
        "geom.area_um2": "features.geometric.area_um2",
        "geom.bbox_width": "features.geometric.bbox_width",
        "geom.bbox_height": "features.geometric.bbox_height",
        "geom.port_count": "features.geometric.port_count",
        "geom.reference_count": "features.geometric.reference_count",
        "geom.polygon_count": "features.geometric.polygon_count",
        "geom.layer_count": "features.geometric.layer_count",
        "prov.call_count": "features.provenance.call_count",
        "prov.object_count": "features.provenance.object_count",
        "prov.route_intent_count": "features.provenance.route_intent_count",
        "prov.max_call_depth": "features.provenance.max_call_depth",
        "timing.build": "timings_s.build",
        "timing.write_gds": "timings_s.write_gds",
    }
    for name, path in pre_paths.items():
        value = _get_path(record, path)
        if _is_number(value):
            numeric[name] = float(value)

    if include_verification_summary:
        for name, path in {
            "drc.error_count": "drc.error_count",
            "lvs.mismatch_markers": "lvs.mismatch_markers",
            "timing.drc": "timings_s.drc",
            "timing.lvs": "timings_s.lvs",
        }.items():
            value = _get_path(record, path)
            if _is_number(value):
                numeric[name] = float(value)

    return numeric, categorical


def make_targets(record: dict[str, Any]) -> tuple[list[float], list[float], list[float], list[float]]:
    cls_values: list[float] = []
    cls_mask: list[float] = []
    for target in CLS_TARGETS:
        value = record.get(target)
        if value is None and target == "pex_pass":
            status = _get_path(record, "physical.pex.status")
            value = status == "PEX Complete" if status is not None else None
        if value is None:
            cls_values.append(0.0)
            cls_mask.append(0.0)
        else:
            cls_values.append(1.0 if bool(value) else 0.0)
            cls_mask.append(1.0)

    area = _get_path(record, "features.geometric.area_um2")
    resistance = _get_path(record, "physical.pex.total_resistance_ohms")
    capacitance = _get_path(record, "physical.pex.total_capacitance_farads")
    runtime = _get_path(record, "timings_s.total")
    values = [area, resistance, capacitance, runtime]
    reg_values: list[float] = []
    reg_mask: list[float] = []
    for value in values:
        if _is_number(value):
            reg_values.append(float(value))
            reg_mask.append(1.0)
        else:
            reg_values.append(0.0)
            reg_mask.append(0.0)
    return cls_values, cls_mask, reg_values, reg_mask


def split_records(records: list[dict[str, Any]], split: str, test_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    if split == "random":
        rng.shuffle(indices)
        cut = max(1, int(len(indices) * (1.0 - test_fraction)))
        return indices[:cut], indices[cut:]
    if split == "holdout-generator":
        by_generator: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            by_generator[str(record.get("generator_id", ""))].append(index)
        generators = sorted(by_generator)
        rng.shuffle(generators)
        target_test = max(1, int(len(records) * test_fraction))
        test: list[int] = []
        for generator in generators:
            if len(test) >= target_test and test:
                break
            test.extend(by_generator[generator])
        train = [index for index in indices if index not in set(test)]
        return train, test
    raise ValueError(f"Unknown split: {split}")


class FeatureSchema:
    def __init__(
        self,
        numeric_keys: list[str],
        categorical_keys: list[str],
        cat_values: dict[str, list[str]],
        num_mean: np.ndarray,
        num_std: np.ndarray,
        reg_mean: np.ndarray,
        reg_std: np.ndarray,
    ):
        self.numeric_keys = numeric_keys
        self.categorical_keys = categorical_keys
        self.cat_values = cat_values
        self.num_mean = num_mean
        self.num_std = num_std
        self.reg_mean = reg_mean
        self.reg_std = reg_std

    def to_json(self) -> dict[str, Any]:
        return {
            "numeric_keys": self.numeric_keys,
            "categorical_keys": self.categorical_keys,
            "cat_values": self.cat_values,
            "num_mean": self.num_mean.tolist(),
            "num_std": self.num_std.tolist(),
            "reg_targets": REG_TARGETS,
            "reg_mean_log1p": self.reg_mean.tolist(),
            "reg_std_log1p": self.reg_std.tolist(),
            "cls_targets": CLS_TARGETS,
        }


def build_schema(rows: list[tuple[dict[str, float], dict[str, str]]], targets: list[tuple], train_indices: list[int]) -> FeatureSchema:
    numeric_keys = sorted({key for numeric, _ in rows for key in numeric})
    categorical_keys = sorted({key for _, categorical in rows for key in categorical})
    cat_values: dict[str, list[str]] = {}
    for key in categorical_keys:
        values = sorted({rows[index][1].get(key, "<MISSING>") for index in train_indices})
        cat_values[key] = ["<UNK>", "<MISSING>"] + [value for value in values if value not in {"<UNK>", "<MISSING>"}]

    train_num = np.zeros((len(train_indices), len(numeric_keys)), dtype=np.float32)
    for row_index, record_index in enumerate(train_indices):
        numeric = rows[record_index][0]
        for col, key in enumerate(numeric_keys):
            train_num[row_index, col] = float(numeric.get(key, 0.0))
    mean = train_num.mean(axis=0) if len(train_indices) else np.zeros(len(numeric_keys), dtype=np.float32)
    std = train_num.std(axis=0) if len(train_indices) else np.ones(len(numeric_keys), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    reg_logs: list[list[float]] = []
    for index in train_indices:
        _, _, reg_values, reg_mask = targets[index]
        reg_logs.append([math.log1p(max(0.0, value)) if mask else np.nan for value, mask in zip(reg_values, reg_mask)])
    reg_arr = np.array(reg_logs, dtype=np.float32) if reg_logs else np.full((0, len(REG_TARGETS)), np.nan, dtype=np.float32)
    reg_mean_values: list[float] = []
    reg_std_values: list[float] = []
    for col in range(len(REG_TARGETS)):
        values = reg_arr[:, col] if reg_arr.size else np.array([], dtype=np.float32)
        values = values[np.isfinite(values)]
        if len(values):
            reg_mean_values.append(float(values.mean()))
            std_value = float(values.std())
            reg_std_values.append(std_value if std_value >= 1e-6 else 1.0)
        else:
            reg_mean_values.append(0.0)
            reg_std_values.append(1.0)
    reg_mean = np.array(reg_mean_values, dtype=np.float32)
    reg_std = np.array(reg_std_values, dtype=np.float32)
    return FeatureSchema(numeric_keys, categorical_keys, cat_values, mean.astype(np.float32), std.astype(np.float32), reg_mean.astype(np.float32), reg_std.astype(np.float32))


class QorDataset(Dataset):
    def __init__(self, rows: list[tuple[dict[str, float], dict[str, str]]], targets: list[tuple], indices: list[int], schema: FeatureSchema):
        self.rows = rows
        self.targets = targets
        self.indices = indices
        self.schema = schema

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        numeric, categorical = self.rows[index]
        x_num = np.zeros(len(self.schema.numeric_keys), dtype=np.float32)
        for col, key in enumerate(self.schema.numeric_keys):
            x_num[col] = float(numeric.get(key, 0.0))
        if len(x_num):
            x_num = (x_num - self.schema.num_mean) / self.schema.num_std

        x_cat = np.zeros(len(self.schema.categorical_keys), dtype=np.int64)
        for col, key in enumerate(self.schema.categorical_keys):
            values = self.schema.cat_values[key]
            lookup = {value: pos for pos, value in enumerate(values)}
            x_cat[col] = lookup.get(categorical.get(key, "<MISSING>"), 0)

        cls_values, cls_mask, reg_values, reg_mask = self.targets[index]
        y_reg = np.array([math.log1p(max(0.0, value)) for value in reg_values], dtype=np.float32)
        y_reg = (y_reg - self.schema.reg_mean) / self.schema.reg_std
        return {
            "x_num": torch.tensor(x_num, dtype=torch.float32),
            "x_cat": torch.tensor(x_cat, dtype=torch.long),
            "y_cls": torch.tensor(cls_values, dtype=torch.float32),
            "m_cls": torch.tensor(cls_mask, dtype=torch.float32),
            "y_reg": torch.tensor(y_reg, dtype=torch.float32),
            "m_reg": torch.tensor(reg_mask, dtype=torch.float32),
        }


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        d_token: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        if n_num:
            self.num_weight = nn.Parameter(torch.randn(n_num, d_token) * 0.02)
            self.num_bias = nn.Parameter(torch.zeros(n_num, d_token))
        else:
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)
        if cat_cardinalities:
            offsets = np.cumsum([0] + cat_cardinalities[:-1]).astype(np.int64)
            self.register_buffer("cat_offsets", torch.tensor(offsets, dtype=torch.long))
            self.cat_embedding = nn.Embedding(sum(cat_cardinalities), d_token)
        else:
            self.register_buffer("cat_offsets", torch.zeros(0, dtype=torch.long))
            self.cat_embedding = None

        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * 2, len(CLS_TARGETS) + len(REG_TARGETS)),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens: list[torch.Tensor] = [self.cls.expand(x_num.shape[0], -1, -1)]
        if self.n_num:
            tokens.append(x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0))
        if self.n_cat:
            tokens.append(self.cat_embedding(x_cat + self.cat_offsets.unsqueeze(0)))
        x = torch.cat(tokens, dim=1)
        x = self.encoder(x)
        out = self.head(self.norm(x[:, 0]))
        return out[:, : len(CLS_TARGETS)], out[:, len(CLS_TARGETS) :]


def masked_loss(logits: torch.Tensor, y_cls: torch.Tensor, m_cls: torch.Tensor, y_reg_hat: torch.Tensor, y_reg: torch.Tensor, m_reg: torch.Tensor) -> torch.Tensor:
    cls_raw = nn.functional.binary_cross_entropy_with_logits(logits, y_cls, reduction="none")
    cls_loss = (cls_raw * m_cls).sum() / m_cls.sum().clamp_min(1.0)
    reg_raw = nn.functional.smooth_l1_loss(y_reg_hat, y_reg, reduction="none")
    reg_loss = (reg_raw * m_reg).sum() / m_reg.sum().clamp_min(1.0)
    return cls_loss + reg_loss


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def _average_precision(y: np.ndarray, p: np.ndarray) -> float | None:
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if (y == 1).sum() == 0:
        return None
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y == 1)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * (y == 1)).sum() / max(1, (y == 1).sum()))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float | None:
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return None
    total = len(y)
    ece = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1 / bins
        in_bin = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if in_bin.any():
            ece += float(in_bin.sum() / total * abs(y[in_bin].mean() - p[in_bin].mean()))
    return ece


def _spearman(y: np.ndarray, p: np.ndarray) -> float | None:
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) < 2:
        return None
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    pr = np.argsort(np.argsort(p)).astype(np.float64)
    if yr.std() < 1e-9 or pr.std() < 1e-9:
        return None
    return float(np.corrcoef(yr, pr)[0, 1])


@torch.no_grad()
def evaluate(model: FTTransformer, loader: DataLoader, schema: FeatureSchema, device: torch.device) -> dict[str, Any]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_y_cls: list[np.ndarray] = []
    all_m_cls: list[np.ndarray] = []
    all_reg_hat: list[np.ndarray] = []
    all_y_reg: list[np.ndarray] = []
    all_m_reg: list[np.ndarray] = []
    losses: list[float] = []
    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_cat = batch["x_cat"].to(device)
        y_cls = batch["y_cls"].to(device)
        m_cls = batch["m_cls"].to(device)
        y_reg = batch["y_reg"].to(device)
        m_reg = batch["m_reg"].to(device)
        logits, reg_hat = model(x_num, x_cat)
        loss = masked_loss(logits, y_cls, m_cls, reg_hat, y_reg, m_reg)
        losses.append(float(loss.detach().cpu()))
        all_logits.append(logits.detach().cpu().numpy())
        all_y_cls.append(y_cls.cpu().numpy())
        all_m_cls.append(m_cls.cpu().numpy())
        all_reg_hat.append(reg_hat.detach().cpu().numpy())
        all_y_reg.append(y_reg.cpu().numpy())
        all_m_reg.append(m_reg.cpu().numpy())

    if not all_logits:
        return {}
    logits = np.concatenate(all_logits)
    prob = 1 / (1 + np.exp(-logits))
    y_cls = np.concatenate(all_y_cls)
    m_cls = np.concatenate(all_m_cls)
    reg_hat = np.concatenate(all_reg_hat) * schema.reg_std + schema.reg_mean
    y_reg = np.concatenate(all_y_reg) * schema.reg_std + schema.reg_mean
    reg_pred = np.maximum(0.0, np.expm1(reg_hat))
    reg_true = np.maximum(0.0, np.expm1(y_reg))
    m_reg = np.concatenate(all_m_reg)

    metrics: dict[str, Any] = {"loss": float(np.mean(losses))}
    for col, name in enumerate(CLS_TARGETS):
        mask = m_cls[:, col] > 0
        if not mask.any():
            continue
        y = y_cls[mask, col]
        p = prob[mask, col]
        pred = (p >= 0.5).astype(np.float32)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        tn = float(((pred == 0) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        specificity = tn / max(1.0, tn + fp)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        order = np.argsort(-p)
        top_k = max(1, int(math.ceil(0.1 * len(order))))
        clean_rate = float((y == 1).mean())
        top_clean_rate = float((y[order[:top_k]] == 1).mean())
        cumulative_clean = np.cumsum(y[order] == 1)
        total_clean = max(1, int((y == 1).sum()))
        needed_for_95 = int(np.searchsorted(cumulative_clean, math.ceil(0.95 * total_clean)) + 1) if total_clean else len(y)
        metrics[name] = {
            "support": int(mask.sum()),
            "positive_rate": clean_rate,
            "accuracy": float((pred == y).mean()),
            "balanced_accuracy": float((recall + specificity) / 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": _auc(y, p),
            "auprc": _average_precision(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "ece": _ece(y, p),
            "top_decile_clean_rate": top_clean_rate,
            "top_decile_lift": top_clean_rate / clean_rate if clean_rate > 0 else None,
            "verification_savings_at_95pct_clean_recall": float(1.0 - needed_for_95 / len(y)),
        }

    for col, name in enumerate(REG_TARGETS):
        mask = m_reg[:, col] > 0
        if not mask.any():
            continue
        y = reg_true[mask, col]
        p = reg_pred[mask, col]
        mae = float(np.mean(np.abs(p - y)))
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        denom = float(np.sum((y - y.mean()) ** 2))
        metrics[name] = {
            "support": int(mask.sum()),
            "mae": mae,
            "rmse": rmse,
            "r2": float(1 - np.sum((p - y) ** 2) / denom) if denom > 1e-12 else None,
            "spearman": _spearman(y, p),
            "true_mean": float(y.mean()),
            "pred_mean": float(p.mean()),
        }
    return metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(Path(args.dataset), max_rows=args.max_rows)
    records = [record for record in records if record.get("build_ok") is not None]
    if len(records) < 2:
        raise RuntimeError("Need at least two records to train/evaluate.")
    feature_rows = [make_feature_row(record, include_verification_summary=args.include_verification_summary) for record in records]
    targets = [make_targets(record) for record in records]
    train_indices, test_indices = split_records(records, args.split, args.test_fraction, args.seed)
    if not train_indices or not test_indices:
        raise RuntimeError(f"Bad split: train={len(train_indices)} test={len(test_indices)}")
    schema = build_schema(feature_rows, targets, train_indices)
    train_ds = QorDataset(feature_rows, targets, train_indices, schema)
    test_ds = QorDataset(feature_rows, targets, test_indices, schema)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cat_cardinalities = [len(schema.cat_values[key]) for key in schema.categorical_keys]
    model = FTTransformer(
        n_num=len(schema.numeric_keys),
        cat_cardinalities=cat_cardinalities,
        d_token=args.d_token,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_metrics: dict[str, Any] | None = None
    best_loss = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x_num = batch["x_num"].to(device)
            x_cat = batch["x_cat"].to(device)
            y_cls = batch["y_cls"].to(device)
            m_cls = batch["m_cls"].to(device)
            y_reg = batch["y_reg"].to(device)
            m_reg = batch["m_reg"].to(device)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits, reg_hat = model(x_num, x_cat)
                loss = masked_loss(logits, y_cls, m_cls, reg_hat, y_reg, m_reg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break
        metrics = evaluate(model, test_loader, schema, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(epoch_losses)) if epoch_losses else None
        print(json.dumps({"epoch": epoch, "train_loss": metrics["train_loss"], "test_loss": metrics.get("loss")}, sort_keys=True))
        if metrics.get("loss", float("inf")) < best_loss:
            best_loss = float(metrics["loss"])
            best_metrics = metrics
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "schema": schema.to_json(), "args": vars(args)}, output_dir / "model.pt")
        if args.max_steps and global_step >= args.max_steps:
            break

    assert best_metrics is not None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(args.dataset),
        "records": len(records),
        "train_records": len(train_indices),
        "test_records": len(test_indices),
        "split": args.split,
        "device": str(device),
        "best": best_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (output_dir / "feature_schema.json").write_text(json.dumps(schema.to_json(), indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an FT-Transformer surrogate QoR predictor.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("random", "holdout-generator"), default="holdout-generator")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-token", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--include-verification-summary", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    result = train(args)
    print(json.dumps(result["best"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
