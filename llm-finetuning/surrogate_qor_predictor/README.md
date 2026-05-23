# Surrogate QoR Predictor

This directory contains a reproducible first-pass pipeline for the gLayout
surrogate model described in the draft Section 2.4.  It builds labeled records
from the current clean-cell corpus, then trains a multi-task FT-Transformer to
predict DRC/LVS/PEX feasibility and cheap QoR proxies.

## Corpus

The registry joins two clean-cell sources:

- 23 verified OpenFASOC strict-syntax conversation builders from
  `llm-finetuning/verified_convo_samples`.
- 20 gLayout SMGR regression cases from `tests/smgr_cases.py`.

Where the builder exposes stable numeric or categorical parameters, the sampler
uses an enhanced Latin-hypercube sweep.  Fixed or expensive hierarchical cells
are still included as generator-conditioned anchor records so held-out-cell
experiments can measure generalization across the full 43-cell corpus.

## Data Collection

Run inside the IIC-OSIC tools container with `GLdev` active:

```bash
export PYTHONPATH="$PWD/src:$PWD/llm-finetuning/surrogate_qor_predictor"
python llm-finetuning/surrogate_qor_predictor/surrogate_qor/collect_dataset.py \
  --output-dir build/surrogate_qor/pilot \
  --samples-per-parameterized-cell 8 \
  --workers 2 \
  --run-drc \
  --run-lvs \
  --skip-pex
```

For a larger run, shard by `--num-shards/--shard-index` and keep each worker
count below the available CPU headroom.  PEX is optional because it is much
slower than geometry extraction and DRC/LVS; use `--run-pex` for the subset that
should receive RC labels.

## Training

```bash
python llm-finetuning/surrogate_qor_predictor/surrogate_qor/train_ft_transformer.py \
  --dataset build/surrogate_qor/pilot/dataset.jsonl \
  --output-dir build/surrogate_qor/pilot/model \
  --split holdout-generator \
  --epochs 80 \
  --batch-size 256 \
  --d-token 256 \
  --layers 6 \
  --heads 8 \
  --amp
```

The trainer writes:

- `metrics.json` with paper-facing classification, calibration, regression, and
  ranking metrics.
- `model.pt` with model weights.
- `feature_schema.json` with the numeric/categorical token schema.

## Metrics Worth Reporting

Beyond loss, the trainer reports DRC/LVS/PEX accuracy, balanced accuracy, F1,
AUROC, AUPRC, Brier score, expected calibration error, top-decile clean-capture
lift, verification-savings-at-95%-recall, and area/RC/runtime MAE, RMSE, R2, and
Spearman correlation.  Use random split for in-distribution performance and
held-out-generator split for the harder generator-conditioned claim.

