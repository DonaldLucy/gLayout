# Seed287 Paper Benchmark Bundle

This directory is the paper-facing control point for the `validated12` seed287
repair benchmark. It keeps the reproducible scripts in git and lets the server
copy only paper-safe artifacts into a single folder after each run.

## What Counts As Reliable

Use `build/repair_bench_validated12_unique287/dataset_drc_lvs_repair.jsonl` as
the primary benchmark input. It excludes mutation attempts that did not activate
a useful DRC/LVS repair record, while preserving the unique seed287 mutation
coverage generated from the 12 strict-clean cells.

Recommended reported sets:

- `seed287_full_repair.jsonl`: all activated DRC/LVS repair rows, currently
  expected to be 277 rows from the 287 seed specs.
- `seed287_smoke_operator3.jsonl`: fast sanity set, up to 3 rows per operator.
- `seed287_balanced_operator5.jsonl`: medium verified set, up to 5 rows per
  operator with case diversity.
- `seed287_representative10.jsonl`: hand-picked qualitative examples for the
  paper narrative and appendix.

## Build The Benchmark Splits

Run this on the server after the seed287 dataset exists:

```bash
python experiments/repair_bench/paper_seed287/make_reliable_benchmark.py \
  --run-root build/repair_bench_validated12_unique287 \
  --output-dir experiments/repair_bench/paper_seed287/benchmark \
  --per-operator 5 \
  --shards 8 \
  --force
```

The `--shards 8` files are optional but useful if full verification needs to be
split across terminals.

## Zero-Shot V2 Commands

Use `policy_v2` for the prompt that adds repair-type guidance learned from the
initial smoke failures without exposing oracle mutation metadata.

Fast verified smoke:

```bash
python experiments/repair_bench/run_zero_shot_baseline.py \
  --dataset experiments/repair_bench/paper_seed287/benchmark/seed287_smoke_operator3.jsonl \
  --output-dir build/repair_bench_validated12_unique287/zero_shot_qwen_policy_v2_smoke30 \
  --limit 0 \
  --max-tokens 1024 \
  --packet-line-budget 350 \
  --prompt-style policy_v2 \
  --run-verification \
  --pdk-root /foss/pdks \
  --force
```

Medium verified benchmark:

```bash
python experiments/repair_bench/run_zero_shot_baseline.py \
  --dataset experiments/repair_bench/paper_seed287/benchmark/seed287_balanced_operator5.jsonl \
  --output-dir build/repair_bench_validated12_unique287/zero_shot_qwen_policy_v2_balanced_operator5 \
  --limit 0 \
  --max-tokens 1024 \
  --packet-line-budget 350 \
  --prompt-style policy_v2 \
  --run-verification \
  --pdk-root /foss/pdks \
  --force
```

Full apply-only pass over all activated repair rows:

```bash
python experiments/repair_bench/run_zero_shot_baseline.py \
  --dataset experiments/repair_bench/paper_seed287/benchmark/seed287_full_repair.jsonl \
  --output-dir build/repair_bench_validated12_unique287/zero_shot_qwen_policy_v2_full_apply_only \
  --limit 0 \
  --max-tokens 1024 \
  --packet-line-budget 350 \
  --prompt-style policy_v2 \
  --force
```

Summarize any run:

```bash
python experiments/repair_bench/summarize_zero_shot_baseline.py \
  build/repair_bench_validated12_unique287/zero_shot_qwen_policy_v2_smoke30 \
  --output build/repair_bench_validated12_unique287/zero_shot_qwen_policy_v2_smoke30/summary.md
```

Inspect failures:

```bash
python experiments/repair_bench/inspect_zero_shot_changes.py \
  build/repair_bench_validated12_unique287/zero_shot_qwen_policy_v2_smoke30 \
  --bucket verification_failed \
  --limit 10 \
  --show-expected \
  --show-diff \
  --text-limit 1200
```

## Collect Paper Artifacts

After a run, collect compact paper-safe evidence:

```bash
python experiments/repair_bench/paper_seed287/collect_seed287_artifacts.py \
  --run-root build/repair_bench_validated12_unique287 \
  --output-dir experiments/repair_bench/paper_seed287/artifacts \
  --include-zero-shot \
  --force
```

This copies summaries, datasets, metrics, representative sample metadata,
repair packets, prompts, responses, and verification case results. It
intentionally does not copy zero-shot `workspace/` directories.
