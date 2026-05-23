#!/usr/bin/env bash
set -euo pipefail

export CONDA_ROOT=/headless/conda-env/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate GLdev
export PATH="$CONDA_PREFIX/bin:$CONDA_ROOT/condabin:/foss/tools/bin:/foss/tools/sak:/foss/tools/klayout:/foss/tools/libman:/foss/tools/osic-multitool:/foss/tools/rftoolkit/bin:/foss/tools/yosys/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$PWD/src:$PWD/llm-finetuning/surrogate_qor_predictor"
export PDK_ROOT="${PDK_ROOT:-/headless/conda-env/miniconda3/share/pdk}"
export PDKPATH="${PDKPATH:-$PDK_ROOT/sky130A}"
export USER="${USER:-cmx}"
export LOGNAME="${LOGNAME:-$USER}"
export HOME="${HOME:-/tmp/$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/$USER/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/$USER/torchinductor}"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"
