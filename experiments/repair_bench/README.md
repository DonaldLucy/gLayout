# SMGR Repair Bench

This experiment builds a small supervised repair benchmark from strict-clean
gLayout cells. It injects reversible source-level bugs, runs DRC/LVS plus the
SMGR localizer, measures whether the localizer points back to the mutated source
site, and writes a JSONL dataset for repair-agent training.

The default case list is intentionally conservative, but named profiles are
available. `validated10` is the broader historical candidate set from the
9/19 -> 10 validated-cell discussion, using
`diff_pair_ibias_labeled_candidate` for the repaired/labeled ibias case; the
bench still validates each case on the current branch/machine before using it.
`validated6` is the strict-clean subset observed on the current SKY130
regression setup and is useful for faster sharded overnight runs.

## Pipeline

1. Select strict-clean cells.
2. Apply one mutation operator to the generator source.
3. Run `scripts/run_smgr_verification_locator.py` in an isolated workspace.
4. Save `case_result.json`, `verification_locator.json`, and `repair_packet.json`.
5. Score top-k source localization against the known mutation site.
6. Emit a supervised `replace_text` repair action.
7. Optionally call a Qwen/OpenAI-compatible endpoint as a zero-shot baseline.

## Mutation Operators

- `label_text_typo`: rename a layout label so extracted top-level pins mismatch.
- `label_layer_wrong`: put a label on the wrong label layer.
- `netlist_pin_swap`: connect a child device pin to the wrong schematic net.
- `missing_connect_subnet`: remove a hierarchical schematic internal connection.
- `top_node_rename`: rename a top-level schematic node.
- `label_moved_to_wrong_port`: keep the label text correct but place it on the wrong routed conductor.
- `physical_route_removed`: remove a physical route while leaving the schematic/netlist unchanged.
- `route_spacing_violation`: reduce route spacing below the PDK rule to stress DRC.

For `label_text_typo`, the localizer now emits `source_label_candidates` and
prioritizes source spans around matching `add_label(text=...)` or label-map
entries. This keeps label repairs grounded in Python source rather than only in
LVS net/call rankings.

## Quick Start

```bash
python experiments/repair_bench/run_repair_bench.py \
  --output-dir build/repair_bench_v0 \
  --max-samples 200 \
  --case-profile validated10 \
  --drop-failed-clean-cases \
  --fast-sample-verification \
  --continue-on-error
```

The benchmark intentionally fails closed: if the strict-clean cells cannot run
through DRC/LVS, no training dataset should be trusted. If the PDK is not in one
of the standard locations, pass it explicitly:

```bash
python experiments/repair_bench/run_repair_bench.py \
  --output-dir build/repair_bench_v0 \
  --max-samples 200 \
  --pdk-root /path/to/pdks \
  --continue-on-error
```

The terminal output is intentionally compact. To inspect a completed or failed
run without dumping Magic/Netgen logs into the terminal:

```bash
python experiments/repair_bench/summarize_repair_bench.py build/repair_bench_v0
python experiments/repair_bench/summarize_repair_bench.py build/repair_bench_v0 --show-failed-logs
```

To export plot-ready Localizer metrics:

```bash
python experiments/repair_bench/export_repair_bench_metrics.py \
  build/repair_bench_v0 \
  --print-summary
```

This writes `metrics/metrics.json`, sample-level CSV, by-case/by-operator
aggregates, and `metrics/localizer_failures.csv` for miss triage.

The isolated workspace intentionally keeps source `.spice` references such as
`src/glayout/pdk/sky130_mapped/sky130_fd_sc_hd.spice`, because the regression
runner uses them when resolving the SKY130 LVS setup.

For a fast planning check:

```bash
python experiments/repair_bench/run_repair_bench.py \
  --output-dir /tmp/repair_bench_plan \
  --max-samples 200 \
  --dry-run \
  --force
```

For higher CPU utilization, shard a run across multiple terminals or machines.
Each shard writes a separate output directory and uses a distinct
`--sample-offset`:

```bash
# terminal 1
python experiments/repair_bench/run_repair_bench.py \
  --output-dir build/repair_bench_validated10_shard0 \
  --max-samples 50 \
  --sample-offset 0 \
  --case-profile validated10 \
  --drop-failed-clean-cases \
  --fast-sample-verification \
  --pdk-root /foss/pdks \
  --continue-on-error \
  --force

# terminal 2: use --sample-offset 50, terminal 3: 100, terminal 4: 150.
```

For the current Qwen baseline:

```bash
export QWEN_API_BASE="http://localhost:8000/v1"
export QWEN_API_KEY="EMPTY"
export QWEN_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"

python experiments/repair_bench/run_zero_shot_baseline.py \
  --dataset build/repair_bench_v0/dataset.jsonl \
  --output-dir build/repair_bench_v0/zero_shot_qwen \
  --unique-mutations \
  --limit 0 \
  --write-selected-dataset build/repair_bench_v0/zero_shot_qwen/selected_unique.jsonl
```

Add `--run-verification` to rerun DRC/LVS after the model's proposed repair
actions are applied.

By default, the zero-shot prompt uses only the repair packet and does not expose
oracle mutation metadata such as the injected operator, description, target
line, or exact buggy source context. Use `--include-oracle-mutation-summary` or
`--include-oracle-target-context` only for explicit ablations.

Plot the verified zero-shot repair outcome by operator, optionally overlaying
SMGR Localizer top-k hit rates from an exported metrics directory:

```bash
python experiments/repair_bench/plot_zero_shot_by_operator.py \
  build/repair_bench_validated6_zero_shot/qwen_unique35_compact_verified_512 \
  --localizer-metrics-dir build/repair_bench_validated6_overnight_v2_metrics \
  --output-prefix build/repair_bench_validated6_zero_shot/qwen_unique35_compact_verified_512/by_operator_zero_shot
```

This writes `.png`, `.svg`, and `.csv` files. If Matplotlib cannot create a
font cache in the container, set `MPLCONFIGDIR=/tmp/mpl-cache` first.

## Outputs

- `plan.json`: exact planned mutations.
- `clean_validation.json`: clean-cell verification records unless skipped.
- `dataset.jsonl`: one supervised repair record per generated sample.
- `samples/<sample_id>/sample.json`: full sample metadata.
- `samples/<sample_id>/verification/...`: raw DRC/LVS/localizer artifacts.
- `summary.json`: localizer top-k hit counts by case and mutation operator.
- `metrics/*.csv`: optional plot-ready Localizer and verification aggregates.
- `zero_shot_qwen/zero_shot_summary.json`: baseline parse/apply/verification results.
- `zero_shot_qwen/*by_operator*.png/.svg/.csv`: optional by-operator zero-shot plots.

The v0 dataset intentionally uses exact reversible source replacements. This
makes the repair label unambiguous while we validate whether provenance and the
localizer provide enough evidence for a smaller repair model.
