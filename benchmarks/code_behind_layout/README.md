# The Code Behind the Layout

Code and frozen configurations for *The Code Behind the Layout: AI Agents for Analog Layout Generators*. The study asks whether a frontier model can make one shared edit to a gLayout generator's source that improves post-extraction performance across a declared family of parameter configurations, while every configuration that passed before still passes. Every candidate is checked on SKY130: GDS generation, DRC, LVS against an independent reference, RC extraction, and SPICE at the tt/ss/ff corners. The editing policies are Claude Opus 5.5 and GPT-6.1.

## Experiments

| | Question | Entry points |
| --- | --- | --- |
| E1 | Which action level works: geometry (G0), primitives (G1) or generator source (G2)? | `scripts/run_e1_scope_suite.py`, `scripts/opus_cohort/run_e1_scope_suite_opus_v2.py` |
| E2 | Which cheap signal ranks candidates before full evaluation: area, a geometric rule, extracted RC, or the calibrated interval F3? | `scripts/e2_four_source.py` |
| E3 | Which test rule finds family regressions within a fixed budget: A anchor, R random, B boundary or C risk? | `scripts/run_e3_suite.py`, `scripts/opus_cohort/run_e3_suite_opus.py` |

The protocol is in [docs/protocol.md](docs/protocol.md), [docs/e1_protocol.md](docs/e1_protocol.md) and [docs/e3_protocol.md](docs/e3_protocol.md). The amendments are in [docs/protocol_amendments.md](docs/protocol_amendments.md). The S0/S1/S2 measurement interface is in [docs/measurement_interface.md](docs/measurement_interface.md).

## Layout

- `src/` contains the libraries: measurement and S0/S1/S2 assessment, task packets, policy adapters, E3 dispatch, risk model and regression selection, and analysis.
- `src/generators/` contains the eight target generators and their independent references.
- `scripts/` contains the scripts for task preparation, physical flows, freezes, suite runners and analysis.
- `scripts/opus_cohort/` runs the frozen E1/E3 suites with the Claude Opus 5.5 policy.
- `configs/` contains the study definition, model configs, task preparations and frozen experiment manifests (`configs/experiments/frozen/`).
- `data/splits/registry.json` is the source-grouped split registry.
- `reproducibility/glayout_fixes.patch` contains the gLayout fixes used by the task circuits.
- `runtime/` contains the two qualified flow modules that the E2/E3 code loads. The data bundle below also extracts here.
- `tests/` contains the unit tests.

## Data bundle

The frozen E1, E2 and E3 task packets are distributed separately (`cbl-data-v1.tar.gz`):

```sh
bash scripts/fetch_data.sh
```

This unpacks `runtime/protocol_freezes/`: the frozen manifests with their task packets, registries, model configs, qualification and baseline plans. It includes the private evaluator side (references, benches, hidden configurations).

## Environment

- **Analysis and unit tests.** Python 3.12:

  ```sh
  python3 -m pip install -r requirements.txt
  python3 -m unittest discover -s tests -p 'test_*.py'
  ```

- **Physical flow.** Runs on Linux with Magic, Netgen, ngspice and SKY130A, using gLayout at commit `5af536b` with `reproducibility/glayout_fixes.patch` applied:

  ```sh
  git checkout 5af536b && git apply benchmarks/code_behind_layout/reproducibility/glayout_fixes.patch
  ```

The scripts that drive the server from a workstation use these placeholders. Recorded paths in the manifests use the same ones.

| Placeholder | Meaning |
| --- | --- |
| `cbl-server` | SSH host alias of the evaluation server (`~/.ssh/config`) |
| `~/.ssh/cbl_server_key` | SSH identity for that host |
| `/srv/code-behind-layout` | Location of this directory on the server |
| `GPT_CLI`, `CLAUDE_CLI` | Command-line clients for the two policies (`src/model_adapter.py`, `src/opus_cli_adapter.py`) |

## Reproducing

**E1 and E3.** Run the suite runners against the frozen manifests:

```sh
python3 scripts/run_e1_scope_suite.py --manifest runtime/protocol_freezes/E1_scope_v2_16k/manifest.json --output runs/E1_gpt
python3 scripts/opus_cohort/run_e1_scope_suite_opus_v2.py --manifest runtime/protocol_freezes/E1_scope_v2_16k_claude_v2/manifest.json --output runs/E1_opus
python3 scripts/run_e3_suite.py --part gpt --manifest runtime/protocol_freezes/E3_primary_v2/manifest.json --output runs/E3_gpt
python3 scripts/opus_cohort/run_e3_suite_opus.py --part gpt --manifest runtime/protocol_freezes/E3_primary_v2_claude/manifest.json --output runs/E3_opus
```

E3 parts:
- `gpt`: the manifest's policy under rules A/R/B/C;
- `parameter`: the parameter baseline P;
- `signals`: the C signal ablations;
- `fidelity`: the calibrated interval F3, with `E3_F3_primary`.

Each runner checks its freeze's bound implementation files before dispatch and reserves a trial identity before launch. Failures are recorded, never retried in place.

**E2.**

```sh
python3 scripts/e2_four_source.py run --manifest runtime/protocol_freezes/E2_four_source_primary/manifest.json --output runs/E2 --method {full_oracle,fixed_area,adaptive_geometry,fixed_rc_product,f3_calibrated} [--pool-id <pool>]
```
