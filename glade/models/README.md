# GLADE Models

This folder contains surrogate-model training code for GLADE benchmarks.

Current contents:

- `current_mirror_surrogate/`: current mirror multi-target surrogate pipeline
  - dataset filtering and feature engineering
  - FT-Transformer training
  - classical baselines (`Ridge`, `RandomForest`, `ExtraTrees`)
  - publication-style evaluation figures and reports
  - surrogate-only environment bootstrap that avoids the legacy full-layout dependency stack

Planned additions:

- feasibility classifiers
- uncertainty calibration
- surrogate-guided search policies
