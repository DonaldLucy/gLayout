# Agent Instructions: With SMGR + Localizer

You are in the SMGR-enabled condition.

You may use:

- The assigned prompt from `prompts/`.
- Normal repo inspection and examples allowed by the run owner.
- SMGR provenance sidecars produced by your own generated GDS files.
- Localizer reports or rankings produced from your own failed runs.
- Raw Magic DRC and Netgen LVS reports.

You must not use:

- Files from `conditions/baseline_no_smgr_localizer/`.
- Logs, patches, or conclusions from another agent's run.
- Held-out solutions produced before your run starts.

Record:

- Every DRC/LVS attempt.
- Every Localizer invocation.
- Whether Localizer changed your next debugging action.
- Final pass/fail status and where the artifacts were stored.
