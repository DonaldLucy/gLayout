## Regression Tests

This folder contains repository-level regression checks for structural changes
such as import-path migrations.

Run the default suite from the repository root:

```bash
tests/run_regression.sh
```

The current suite focuses on:

- canonical imports under `glayout.cells`
- compatibility imports under `glayout.blocks`
- canonical imports under `glayout.verification`
- repository layout checks for the `legacy/atlas` move

Additional SMGR coverage:

- `tests/run_smgr_regression.py`
  Runs baseline vs traced GDS generation, validates the provenance sidecar, and can invoke Magic DRC plus Netgen LVS for the SMGR regression catalog in `tests/smgr_cases.py`.
- `tests/test_smgr_snapshot.py`
  Exercises provenance query and candidate-ranking logic without requiring the full layout stack.
- `tests/test_smgr_case_catalog.py`
  Checks that the SMGR regression catalog still covers the main public cells and key composite sub-blocks.
