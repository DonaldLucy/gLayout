# Source-Mapped Generator Runtime Runbook

This runbook assumes you are using the standard `IIC-OSIC-TOOLS` container flow and want to validate the Source-Mapped Generator Runtime (SMGR) on `gLayout`.

## 1. Update The Repository

On the server host:

```bash
cd ~/eda/designs/gLayout
git pull
```

If you are working on a feature branch:

```bash
git fetch --all --prune
git checkout <your-branch>
git pull --ff-only
```

## 2. Start And Enter The Container

On the server host:

```bash
cd ~/eda/designs/gLayout/docs/IIC-OSIC-TOOLS
./start_chipathon_jupyter.sh
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Enter the running container:

```bash
sudo docker exec -it \
  -e PYTHONPATH= \
  -e PYTHONHOME= \
  iic-osic-tools_chipathon_jupyter_uid_$(id -u) \
  bash --noprofile --norc
```

## 3. Activate The gLayout Environment

Inside the container:

```bash
export CONDA_ROOT=/headless/conda-env/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate GLdev
export PATH="$CONDA_PREFIX/bin:$CONDA_ROOT/condabin:/foss/tools/bin:/foss/tools/sak:/foss/tools/klayout:/foss/tools/libman:/foss/tools/osic-multitool:/foss/tools/rftoolkit/bin:/foss/tools/yosys/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

cd /foss/designs/gLayout
export PYTHONPATH="$PWD/src"
export PDK_ROOT="/headless/conda-env/miniconda3/share/pdk"
export PDKPATH="$PDK_ROOT/sky130A"
```

Sanity-check the toolchain:

```bash
which python
which magic
which netgen
which ngspice
python -c "import gdsfactory; from glayout import sky130; print(gdsfactory.__version__); print(sky130 is not None)"
```

## 4. Manual SMGR Usage

If you want a normal generator call to emit a provenance sidecar automatically whenever it writes GDS:

```bash
export GLAYOUT_SMGR=1
```

The default mode is intentionally compact: it records generator-level outputs, call hierarchy, ports summary, and instance mapping, but it does not record every transient polygon or every intermediate port event.

If you ever need the old verbose debugging mode for runtime internals, you can opt in explicitly:

```bash
export GLAYOUT_SMGR_CAPTURE_POLYGONS=1
export GLAYOUT_SMGR_CAPTURE_PORT_OBJECTS=1
export GLAYOUT_SMGR_CAPTURE_LIVE_REFS=1
```

Example:

```bash
python - <<'PY'
from glayout import sky130
from glayout.cells.elementary.diff_pair import diff_pair

comp = diff_pair(sky130, width=3, fingers=4)
path = comp.write_gds("build/manual/diff_pair.gds")
print(path)
PY
```

Expected artifacts:

```bash
ls -lh build/manual/diff_pair.gds
ls -lh build/manual/diff_pair.provenance.json
```

Quick JSON inspection:

```bash
python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("build/manual/diff_pair.provenance.json").read_text())
print("format:", payload["format"])
print("calls:", len(payload["calls"]))
print("objects:", len(payload["objects"]))
print("top component:", payload["artifacts"]["top_component_name"])
PY
```

## 5. Full SMGR Regression

The regression runner will:

- build each case twice
- generate baseline GDS with SMGR disabled
- generate traced GDS with SMGR enabled
- assert the two GDS files are byte-identical
- assert the provenance sidecar exists and is queryable
- optionally run Magic DRC and Netgen LVS on both baseline and traced outputs

Run the full regression:

```bash
python tests/run_smgr_regression.py --output-dir build/smgr_regression
```

Run only a subset of cases:

```bash
python tests/run_smgr_regression.py \
  --output-dir build/smgr_regression_subset \
  --cases diff_pair_default current_mirror_nfet opamp
```

Skip physical verification if you only want provenance validation:

```bash
python tests/run_smgr_regression.py \
  --output-dir build/smgr_regression_fast \
  --skip-drc \
  --skip-lvs
```

The runner writes a machine-readable summary here:

```bash
cat build/smgr_regression/summary.json
```

## 6. Lightweight Logic Tests

If you only want to exercise the non-layout logic for the provenance query layer and case catalog:

```bash
python -m pytest -q tests/test_smgr_snapshot.py tests/test_smgr_case_catalog.py
```

If `pytest` is unavailable in the environment, you can still execute the assertions directly:

```bash
python - <<'PY'
from tests.test_smgr_snapshot import test_snapshot_query_and_ranking
from tests.test_smgr_case_catalog import test_case_catalog_contains_public_cells_and_subblocks

test_snapshot_query_and_ranking()
test_case_catalog_contains_public_cells_and_subblocks()
print("SMGR lightweight checks passed")
PY
```

## 7. Reading A Provenance Sidecar

Programmatic query example:

```bash
python - <<'PY'
from glayout import load_provenance

snapshot = load_provenance("build/manual/diff_pair.provenance.json")
print("call ids:", list(snapshot.calls)[:5])
print("top candidates:", snapshot.rank_candidate_calls([0.0, 0.0, 20.0, 20.0], rule_name="manual_probe")[:3])
PY
```

## 8. Common Notes

- SMGR is instrumentation-only. The regression runner compares baseline and traced GDS to catch any geometry drift.
- Magic and Netgen are checked explicitly through their real report files. Missing reports are treated as failures, not as clean passes.
- Some composite generators reuse cached cells internally. The regression runner clears the gdsfactory cache before each build to keep call-tree capture stable.
