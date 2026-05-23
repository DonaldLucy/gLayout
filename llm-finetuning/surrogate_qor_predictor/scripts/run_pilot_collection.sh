#!/usr/bin/env bash
set -euo pipefail

cd "${1:-/foss/designs/gLayout}"
source llm-finetuning/surrogate_qor_predictor/scripts/testserver2_container_env.sh

python llm-finetuning/surrogate_qor_predictor/surrogate_qor/collect_dataset.py \
  --output-dir build/surrogate_qor/pilot \
  --samples-per-parameterized-cell "${SAMPLES_PER_CELL:-4}" \
  --limit "${LIMIT:-16}" \
  --workers "${WORKERS:-2}" \
  --run-drc \
  --run-lvs \
  --skip-pex \
  --skip-existing

