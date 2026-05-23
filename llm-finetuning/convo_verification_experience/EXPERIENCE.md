# gLayout Convo Samples: Generation and DRC/LVS Notes

## Result

Yes: the generated treatment set is fully clean.

- Source prompt split seed: `20260522`
- Generated and repaired samples: `23`
- Held-out experiment prompts: `24`
- Final verification host: `CPU-TestServer`
- Final verification directory: `build/verified_convo_samples_final_23`
- Final verification result: `23/23` builders produced SMGR provenance sidecars and passed Magic DRC plus Netgen LVS
- Final summary file in the server container: `/foss/designs/gLayout/build/verified_convo_samples_final_23/summary.json`
- Final summary key result: `failures: {}`

The corresponding repository-side implementation lives in `llm-finetuning/verified_convo_samples/`. A copy of those files is also archived next to this note under `verified_sample_files/`.

## Directory Contents

This directory is a self-contained work packet for the generated/held-out split:

```text
llm-finetuning/convo_verification_experience/
  EXPERIENCE.md
  split_manifest.json
  source_convos/
    generated_clean_23/
      *.convo
    held_out_experiment_24/
      *.convo
  verified_sample_files/
    README.md
    manifest.json
    verify_convo_samples.py
    convo_layouts/
      __init__.py
      common.py
      generated.py
```

`source_convos/generated_clean_23/` contains the original OpenFASOC prompt files for the 23 samples that were generated and verified. `source_convos/held_out_experiment_24/` contains the original prompt files for the 24 samples reserved for the experiment/control group. The held-out files were only archived; no layout generation or DRC/LVS repair was done for them.

## Generated Clean Set

- `CTATVGen`
- `CascodeCommonGate`
- `CascodeCommonGateCommonCentroid`
- `CascodeCommonSourceInterdigitated`
- `ClassABStage`
- `ClassBPushPull`
- `ClassBPushPullInterdigitated`
- `CommonSourceAmplifier`
- `CommonSourceAmplifierWDiodeLoad`
- `CrossCoupledInverters`
- `CurrentMirrorNtypeInterdigitated`
- `CurrentMirrorPtype`
- `CurrentMirrorPtypeInterdigitated`
- `DiffPair`
- `FourStageIntegrator`
- `Inverter`
- `LowNoiseAmp`
- `MimcapArray`
- `NoiseXDiffConv`
- `PMOSArray2x5`
- `PMOSArray4x3`
- `ULPD`
- `Varactor`

## Held-Out Experiment Set

- `BiasVoltageGenerator`
- `CascodeCommonGateInterdigitated`
- `CascodeCommonSource`
- `CommonSourceAmplifierFoldedDiodeLoad`
- `ConstBiasVoltageGen`
- `CurrentMirrorNtype`
- `CurrentMirrorNtypeCommonCentroid`
- `DegenCommonGate`
- `DegenCommonSource`
- `DeltaSigmaModulator`
- `IntegratorStage`
- `NAND`
- `NMOSArray2x5`
- `NMOSArray4x3`
- `NOR`
- `PTATVoltageGen`
- `PTypeDiffPair`
- `PushPull`
- `RegulatedCascode`
- `SourceFollow`
- `StrongArmLatch`
- `ViaArray3x2`
- `VoltageFollower`
- `WilsonCurrentMirror`

## Workflow That Worked

1. Freeze the split first.

   Use a manifest with a fixed seed, record both the generated set and the held-out set, and do not quietly move prompts between groups after debugging starts. This prevents the verification process from biasing the experiment group.

2. Translate each `.convo` into a small, explicit gLayout builder.

   The prompt usually gives placement and connection intent, but not a complete LVS-ready implementation. The generated builder should create the devices, place them deterministically, add routes, expose useful ports, add top-level labels, and build a matching `Netlist`.

3. Keep verification as close to the generated geometry as possible.

   The helper layer in `convo_layouts/common.py` is deliberately small: device specs, route specs, placement, labels, and netlist creation. Repairs that are generally reusable went there; sample-specific topology stayed in `generated.py`.

4. Run one or a small batch first, then full verification.

   The practical loop was:

   ```bash
   python llm-finetuning/verified_convo_samples/verify_convo_samples.py \
     --samples <SampleName> \
     --output-dir build/<debug_dir> \
     --continue-on-error
   ```

   After individual fixes were clean, run the full set:

   ```bash
   python llm-finetuning/verified_convo_samples/verify_convo_samples.py \
     --output-dir build/verified_convo_samples_final_23 \
     --continue-on-error
   ```

5. Confirm three things for every sample.

   - Magic DRC report has zero errors.
   - Netgen LVS reports a unique match.
   - The SMGR provenance sidecar exists next to the generated GDS.

6. Record repair notes in the manifest.

   Any prompt typo, missing macro, body-tie adjustment, or route workaround should be written down next to the sample. The note is often more valuable than the final code when debugging the next batch.

## Main Pitfalls

### Prompt intent is not always a complete netlist

Some prompts describe placement operations or high-level components, but do not fully specify an LVS-observable topology. The builder still needs an explicit `Netlist`, and the physical labels must match what Magic extracts.

Examples:

- `CurrentMirrorPtype` had a self-route typo in the source prompt. The clean implementation connects `reference_source_E` to `mirror_source_E`.
- `LowNoiseAmp` said to move one device below another without a precise anchor. The implementation chose a deterministic placement and recorded it.

### Body ties and bulk nets can dominate LVS

MOS bulk handling was the most common source of LVS noise. If a prompt does not intend an observable body-tie connection, adding ties can accidentally short nets through the tap ring. For some no-tie MOS-only bulk groups, the clean approach was to keep the bulk as an internal net and omit it from the top-level node list and labels.

Reusable fix:

- Add implicit internal bulk-net handling for no-tie NMOS/PMOS groups.
- Keep explicit body ties where the prompt or extraction needs observable bulk pins.
- Avoid assuming that a visually isolated tie is electrically isolated after Magic extraction.

### Source/drain equivalence still needs consistent labels

Sky130 MOS source and drain can be equivalent in Netgen setup, but the generated netlist and labels still need to be stable. A route that touches the wrong side of a device may pass DRC and still create an LVS mismatch or an unintended merged net.

Useful patterns:

- Prefer explicit east/west/north/south port choices.
- Add short local segments when a source/drain merge is intentional.
- Keep label names derived from the union-find net groups, not ad hoc strings.

### Temporary high-metal routes can create real shorts

Some attempted "escape" or high-metal routes fixed geometry but introduced shorts through endpoint via stacks. This showed up especially in cross-coupled or composite-stage prompts. If a route is only meant to encode logical intent but the physical route is not robust, it should not be faked into a clean sample.

Practical result:

- Add a `logical` route kind for netlist-only intent where needed.
- Use real local shorts only for small, verified merges.
- For high-level composite prompts without clean local macros, prefer a clean prompt-derived placement skeleton and record the limitation.

### Composite macro prompts may not map to local clean builders

`ClassABStage`, `CrossCoupledInverters`, and `FourStageIntegrator` reference higher-level prompt components or macro structure that was not available as clean local builders in this workflow. They were verified as prompt-derived placement skeletons rather than fully routed macro implementations, and this is recorded in `manifest.json`.

This is better than forcing brittle routes that pass one check accidentally and fail later.

### Array and MIM samples are simple only after extraction behavior is understood

Device arrays and MIM capacitors can be compact in Python, but Magic/Netgen may merge parallel devices or capacitances. The expected schematic netlist should match that merging behavior. The final LVS can still say many physical devices collapsed to fewer equivalent devices; that is fine if the final result is a unique match.

### Provenance should be checked as a first-class output

Passing DRC/LVS is not enough for this dataset. The verification script also checks that `write_gds` produced the `.provenance.json` sidecar. A sample without provenance is not a complete training artifact.

## CPU-TestServer Command Pattern

Inside the requested container environment, the important verification shape was:

```bash
export CONDA_ROOT=/headless/conda-env/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate GLdev
export PATH="$CONDA_PREFIX/bin:$CONDA_ROOT/condabin:/foss/tools/bin:/foss/tools/sak:/foss/tools/klayout:/foss/tools/libman:/foss/tools/osic-multitool:/foss/tools/rftoolkit/bin:/foss/tools/yosys/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

cd /foss/designs/gLayout
export PYTHONPATH="$PWD/src"
export PDK_ROOT="/headless/conda-env/miniconda3/share/pdk"
export PDKPATH="$PDK_ROOT/sky130A"

python llm-finetuning/verified_convo_samples/verify_convo_samples.py \
  --output-dir build/verified_convo_samples_final_23 \
  --continue-on-error
```

For local syntax checks before syncing to the server:

```bash
jq empty llm-finetuning/verified_convo_samples/manifest.json
python -m py_compile \
  llm-finetuning/verified_convo_samples/convo_layouts/common.py \
  llm-finetuning/verified_convo_samples/convo_layouts/generated.py \
  llm-finetuning/verified_convo_samples/verify_convo_samples.py
git diff --check -- llm-finetuning/verified_convo_samples
```

## Suggested Process For The Held-Out 24 Later

1. Start from `source_convos/held_out_experiment_24/`.
2. Check whether a prompt is genuinely new or duplicates an already clean primitive.
3. Implement one builder at a time, using the patterns in `verified_sample_files/convo_layouts/common.py`.
4. Run per-sample DRC/LVS before adding it to any full run.
5. Use provenance sidecars, Magic DRC reports, and Netgen LVS reports as required acceptance criteria.
6. Record every prompt ambiguity or repair in the manifest immediately.
7. Only after all selected held-out samples are clean, run a new full final verification directory.

## Rule Of Thumb

Do not optimize for making the prompt "look implemented"; optimize for a layout/netlist pair that Magic and Netgen agree on, with the repair decision documented. The useful dataset sample is the one that is reproducible, provenance-backed, and honest about any prompt-level ambiguity.
