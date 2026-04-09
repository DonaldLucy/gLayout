# Current Mirror Benchmark

This folder contains the first GLADE benchmark implementation.

The benchmarked generator is:

- `glayout.cells.elementary.current_mirror.current_mirror`

Declared parameter grid (from the inventory sheet):

- `device`: `["nfet", "pfet"]`
- `numcols`: `1..5`, step `1`
- `width`: `0.5..20.0`, step `0.25`
- `length`: `0.15..4.0`, step `0.2`

This yields `2 * 5 * 79 * 20 = 15800` parameter combinations.

The dataset builder stores:

- generator id and parameters
- source netlist
- build status
- DRC / LVS summaries
- PEX summaries
- geometric features
- runtime
- failure taxonomy

Artifacts and exported datasets are written under `output/glade/current_mirror/`.
