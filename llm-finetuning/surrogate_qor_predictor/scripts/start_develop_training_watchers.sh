#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/cmx/eda/designs/gLayout}"
PYTHON_BIN="${PYTHON_BIN:-/home/cmx/venvs/qwen-vllm-h100/bin/python}"
export PYTHONPATH="$REPO_DIR/llm-finetuning/surrogate_qor_predictor:$REPO_DIR/src"

count_lines() {
  local file="$1"
  if [[ -f "$file" ]]; then
    wc -l < "$file"
  else
    echo 0
  fi
}

run_train() {
  local dataset="$1"
  local output_dir="$2"
  local split="$3"
  local epochs="$4"
  local batch="$5"
  local d_token="$6"
  local layers="$7"
  local heads="$8"
  local log="$9"

  "$PYTHON_BIN" "$REPO_DIR/llm-finetuning/surrogate_qor_predictor/surrogate_qor/train_ft_transformer.py" \
    --dataset "$dataset" \
    --output-dir "$output_dir" \
    --split "$split" \
    --epochs "$epochs" \
    --batch-size "$batch" \
    --d-token "$d_token" \
    --layers "$layers" \
    --heads "$heads" \
    --amp > "$log" 2>&1
}

run_main_watcher() {
  cd "$REPO_DIR"
  local run_dir="build/surrogate_qor/develop_h100_drc_lvs_3k"
  mkdir -p "$run_dir"

  while true; do
    local n
    n="$(count_lines "$run_dir/dataset.jsonl")"
    echo "$(date -Iseconds) partial_wait records=$n" >> "$run_dir/trainer_watch.log"
    [[ "$n" -ge 500 ]] && break
    sleep 60
  done

  run_train "$run_dir/dataset.jsonl" "$run_dir/model_big_partial" random 80 256 1024 12 16 "$run_dir/training_big_partial.log" \
    || run_train "$run_dir/dataset.jsonl" "$run_dir/model_big_partial_fallback" random 80 256 768 10 12 "$run_dir/training_big_partial_fallback.log" \
    || echo "$(date -Iseconds) partial_train_failed" >> "$run_dir/trainer_watch.log"

  while ! grep -q collection_done "$run_dir/run_status.txt" 2>/dev/null; do
    echo "$(date -Iseconds) full_wait records=$(count_lines "$run_dir/dataset.jsonl")" >> "$run_dir/trainer_watch.log"
    sleep 120
  done

  run_train "$run_dir/dataset.jsonl" "$run_dir/model_big_full" holdout-generator 120 256 1024 12 16 "$run_dir/training_big_full.log" \
    || run_train "$run_dir/dataset.jsonl" "$run_dir/model_big_full_fallback" holdout-generator 120 256 768 10 12 "$run_dir/training_big_full_fallback.log" \
    || echo "$(date -Iseconds) full_train_failed" >> "$run_dir/trainer_watch.log"

  echo "$(date -Iseconds) trainer_done" >> "$run_dir/trainer_watch.log"
}

run_pex_watcher() {
  cd "$REPO_DIR"
  local run_dir="build/surrogate_qor/develop_h100_pex_313"
  mkdir -p "$run_dir"

  while ! grep -q collection_done "$run_dir/run_status.txt" 2>/dev/null; do
    echo "$(date -Iseconds) pex_wait records=$(count_lines "$run_dir/dataset.jsonl")" >> "$run_dir/trainer_watch.log"
    sleep 120
  done

  run_train "$run_dir/dataset_refreshed.jsonl" "$run_dir/model_big_pex" random 100 128 768 10 12 "$run_dir/training_big_pex.log" \
    || run_train "$run_dir/dataset_refreshed.jsonl" "$run_dir/model_big_pex_fallback" random 100 128 512 8 8 "$run_dir/training_big_pex_fallback.log" \
    || echo "$(date -Iseconds) pex_train_failed" >> "$run_dir/trainer_watch.log"

  echo "$(date -Iseconds) trainer_done" >> "$run_dir/trainer_watch.log"
}

case "${1:-all}" in
  main)
    run_main_watcher
    ;;
  pex)
    run_pex_watcher
    ;;
  all)
    nohup "$0" main >/dev/null 2>&1 &
    echo "main_watcher_pid=$!"
    nohup "$0" pex >/dev/null 2>&1 &
    echo "pex_watcher_pid=$!"
    ;;
  *)
    echo "usage: $0 [main|pex|all]" >&2
    exit 2
    ;;
esac

