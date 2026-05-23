# Held-Out Debug Experiment

This directory prepares the 24 held-out `.convo` prompts for a later A/B-style debugging-speed experiment. It does not start the experiment and does not generate or repair any held-out layouts.

The goal is to compare agent debugging speed under two isolated conditions:

- `with_smgr_localizer`: the agent may use SMGR provenance and Localizer outputs.
- `baseline_no_smgr_localizer`: the agent may use normal code inspection plus raw Magic/Netgen reports, but must not use SMGR provenance or Localizer outputs.

Both conditions contain the same 24 held-out prompts so that different agents can be assigned independently without sharing files, logs, or artifacts.

## Layout

```text
llm-finetuning/heldout_debug_experiment/
  README.md
  PROTOCOL.md
  shared/
    split_manifest.json
    heldout_prompt_index.csv
    scorecard_template.csv
  conditions/
    with_smgr_localizer/
      AGENT_INSTRUCTIONS.md
      README.md
      run_log_template.md
      prompts/
      workspace/
      logs/
      results/
      artifacts/
    baseline_no_smgr_localizer/
      AGENT_INSTRUCTIONS.md
      README.md
      run_log_template.md
      prompts/
      workspace/
      logs/
      results/
      artifacts/
```

## Isolation Rules

- Do not let one run read the other condition's `workspace/`, `logs/`, `results/`, or `artifacts/`.
- Use separate git branches or worktrees per agent and condition.
- Use separate verification output directories, for example `build/heldout_debug_experiment/with_smgr_localizer/<agent>/<prompt>` and `build/heldout_debug_experiment/baseline_no_smgr_localizer/<agent>/<prompt>`.
- Record wall-clock time, turns, tool calls, DRC/LVS runs, and final status in the scorecard.
- Keep the held-out prompts unchanged until a formal run starts.

## Current State

- Held-out prompt count per condition: `24`
- Generated clean treatment prompts are not duplicated here; they remain in `llm-finetuning/convo_verification_experience/`.
- The 23 previously generated treatment samples passed DRC/LVS, but their repair history should not be used as hidden feedback during a held-out run unless the protocol explicitly allows both conditions to inspect it.
