# Current Mirror Surrogate Stack

This module trains a multi-target surrogate for the GLADE current mirror benchmark.

It is designed for datasets exported by:

- `python -m glade.current_mirror.collect_dataset`

Supported dataset formats:

- flattened `dataset.csv`
- raw or flattened `dataset.jsonl`
- `dataset.parquet`

## What the pipeline models

Default targets are chosen to reflect layout quality and evaluation cost:

- `geom_area_um2`
- `pex_total_resistance_ohms`
- `pex_total_capacitance_farads`
- `derived_pex_rc_tau_s`
- `runtime_total_s`

The preprocessing step intentionally keeps only parameter-space and structure-space inputs, then drops:

- IDs and bookkeeping columns
- quality/status outputs that would leak the target
- constant columns such as single-device indicators when the whole dataset only contains one FET type

Derived inputs are added from the design parameters, including:

- effective width
- total channel-area proxy
- gate aspect ratio
- conductance proxy
- resistance proxy

## Training

Example:

```bash
python3 -m glade.models.current_mirror_surrogate.train \
  --dataset output/glade/current_mirror/dataset.csv \
  --output-root output/glade/current_mirror_surrogate \
  --run-name a100_ftt_default
```

Important outputs:

- `selection_report.json`: exact feature/target selection
- `dataset_profile.md`: human-readable feature report
- `metrics_per_target.csv`: per-model, per-target metrics
- `predictions_eval.csv`: hold-out predictions
- `figures/`: PNG, PDF, and SVG figures for meetings/papers
- `models/ft_transformer_best.pt`: FT-Transformer checkpoint
- `models/baselines.pkl`: classical baseline models

## Plot regeneration

```bash
python3 -m glade.models.current_mirror_surrogate.plot_results \
  --run-dir output/glade/current_mirror_surrogate/a100_ftt_default
```
