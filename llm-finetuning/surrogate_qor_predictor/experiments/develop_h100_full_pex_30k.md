# Develop H100 Full PEX 30k Overnight Run

## Goal

Train a generator-conditioned surrogate for unseen gLayout code and parameter combinations.  Each record is generated from a clean gLayout generator plus sampled sizing, placement, and routing parameters, then verified with DRC, LVS, and PEX.  The model predicts pass probabilities and analog-QoR proxies before full verification.

## Dataset

```text
host: Develop-Server
container: iic-osic-tools_chipathon_jupyter_uid_1044
repo in container: /foss/designs/gLayout
run directory: build/surrogate_qor/develop_h100_drc_lvs_pex_30k
sampler: deterministic maximin Latin hypercube
seed: 20260523
clean generator corpus: 43 cells
parameterized generators: 30
fixed anchor generators: 13
samples_per_parameterized_cell: 1000
planned records: 30 * 1000 + 13 = 30013
workers: 26
verification: DRC + LVS + PEX
SMGR sidecar: enabled
```

Collection command:

```bash
python llm-finetuning/surrogate_qor_predictor/surrogate_qor/collect_dataset.py \
  --output-dir build/surrogate_qor/develop_h100_drc_lvs_pex_30k \
  --samples-per-parameterized-cell 1000 \
  --workers 26 \
  --run-drc \
  --run-lvs \
  --run-pex \
  --skip-existing
```

## Labels

Classification targets:

```text
drc_pass
lvs_pass
pex_pass
```

Regression targets:

```text
area_um2
bbox_width_um
bbox_height_um
aspect_ratio
symmetry_score_horizontal
symmetry_score_vertical
resistor_count
capacitor_count
total_resistance_ohms
total_capacitance_farads
resistance_per_um2
capacitance_per_um2
parasitic_device_density_per_um2
resistance_per_port
capacitance_per_port
rc_product
runtime_s
```

## Feature Inputs

The predictor uses generator identity, corpus/family metadata, sampled numeric and categorical parameters, source-code summary features, cheap geometry features after generation, and SMGR/provenance summary features.  Verification-result summary features are omitted for the main pre-verification predictor.

## Training Schedule

A tmux watcher starts training checkpoints while collection proceeds, then trains the final model after all 30013 records are collected.

Partial 5k checkpoint:

```bash
python llm-finetuning/surrogate_qor_predictor/surrogate_qor/train_ft_transformer.py \
  --dataset build/surrogate_qor/develop_h100_drc_lvs_pex_30k/dataset.jsonl \
  --output-dir build/surrogate_qor/develop_h100_drc_lvs_pex_30k/model_h100_large_5k \
  --max-rows 5000 \
  --split holdout-generator \
  --epochs 160 \
  --min-epochs 30 \
  --patience 18 \
  --min-delta 0.0005 \
  --batch-size 96 \
  --d-token 2048 \
  --layers 20 \
  --heads 16 \
  --dropout 0.1 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --grad-clip 1.0 \
  --amp
```

Partial 15k checkpoint uses the same parameters with `--max-rows 15000` and output `model_h100_large_15k`.

Final 30k+ model:

```bash
python llm-finetuning/surrogate_qor_predictor/surrogate_qor/train_ft_transformer.py \
  --dataset build/surrogate_qor/develop_h100_drc_lvs_pex_30k/dataset.jsonl \
  --output-dir build/surrogate_qor/develop_h100_drc_lvs_pex_30k/model_h100_xlarge_full \
  --split holdout-generator \
  --epochs 240 \
  --min-epochs 40 \
  --patience 24 \
  --min-delta 0.0003 \
  --batch-size 64 \
  --d-token 2560 \
  --layers 24 \
  --heads 20 \
  --dropout 0.1 \
  --lr 0.0002 \
  --weight-decay 0.0001 \
  --grad-clip 1.0 \
  --amp
```

The final model is intentionally large for the H100: `d_token=2560`, `24` Transformer layers, `20` heads, and batch size `64`.  Early stopping keeps the best validation checkpoint by hold-out-generator loss.

## Metrics

The trainer records classification accuracy, balanced accuracy, F1, AUROC, AUPRC, Brier score, expected calibration error, top-decile clean-capture lift, verification savings at 95% clean recall, and regression MAE/RMSE/R2/Spearman for every area/RC/QoR target.
