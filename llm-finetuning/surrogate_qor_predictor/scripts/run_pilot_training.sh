#!/usr/bin/env bash
set -euo pipefail

cd "${1:-/foss/designs/gLayout}"
source llm-finetuning/surrogate_qor_predictor/scripts/testserver2_container_env.sh

python llm-finetuning/surrogate_qor_predictor/surrogate_qor/train_ft_transformer.py \
  --dataset build/surrogate_qor/pilot/dataset.jsonl \
  --output-dir build/surrogate_qor/pilot/model \
  --split random \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --d-token "${D_TOKEN:-128}" \
  --layers "${LAYERS:-3}" \
  --heads "${HEADS:-4}" \
  --amp

