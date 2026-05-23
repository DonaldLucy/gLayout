# Agent Instructions: Baseline Without SMGR + Localizer

You are in the baseline condition.

You may use:

- The assigned prompt from `prompts/`.
- Normal repo inspection and examples allowed by the run owner.
- Raw Magic DRC reports.
- Raw Netgen LVS reports.
- Python exceptions, generated code, generated GDS paths, and direct reasoning.

You must not use:

- SMGR provenance sidecars.
- SMGR source maps, trace graphs, component provenance, or provenance-derived summaries.
- Localizer outputs, rankings, or hints.
- Files from `conditions/with_smgr_localizer/`.
- Logs, patches, or conclusions from another agent's run.

Record:

- Every DRC/LVS attempt.
- Any time you were tempted to use provenance or localization but did not.
- Final pass/fail status and where the artifacts were stored.
