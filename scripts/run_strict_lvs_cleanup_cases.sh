#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

unset PDK_ROOT
unset PDKPATH
unset MAGIC_PDK_ROOT
unset NETGEN_PDK_ROOT

export GLAYOUT_SMGR=1
unset GLAYOUT_SMGR_CAPTURE_POLYGONS
unset GLAYOUT_SMGR_CAPTURE_PORT_OBJECTS
unset GLAYOUT_SMGR_CAPTURE_LIVE_REFS

python tests/run_smgr_regression.py \
  --output-dir build/smgr_regression_strict_lvs_cleanup \
  --cases \
    diff_pair_default \
    diff_pair_generic \
    transmission_gate \
    low_voltage_cmirror \
    fvf_based_ota_low_voltage_cmirror \
  --continue-on-error
