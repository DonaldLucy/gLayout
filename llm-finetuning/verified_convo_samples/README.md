# Verified Convo Layout Samples

This directory contains gLayout Python builders generated from a reproducible half-sample of the OpenFASOC strict-syntax conversations:

<https://github.com/idea-fasoc/OpenFASOC/tree/main/openfasoc/generators/glayout/glayout/llm/syntax_data/convos>

The split uses seed `20260522`. The selected half is the treatment group for generation and verification; the remaining half is kept untouched as an experiment/control group.

Generated builders live in `convo_layouts/generated.py`. They intentionally add explicit top-level labels and matching `Netlist` objects so Magic DRC and Netgen LVS can verify the generated geometry. A few prompts needed small cleanup repairs, recorded in `manifest.json`, such as fixing a self-route typo and adding body ties where the prompt omitted them but LVS needs observable bulk pins. Three higher-level prompts are verified as prompt-derived placement skeletons because their source conversations reference composite macros that are not present as clean local builders.

Run verification in an IIC-OSIC tools container:

```bash
python llm-finetuning/verified_convo_samples/verify_convo_samples.py \
  --output-dir build/verified_convo_samples \
  --continue-on-error
```

The script writes GDS, SMGR provenance sidecars, Magic DRC reports, Netgen LVS reports, and a `summary.json` under the output directory.

Final verification on `CPU-TestServer` used `build/verified_convo_samples_final_23` and passed all 23 generated builders. The selected treatment set, held-out experiment group, and repair notes are recorded in `manifest.json`.
