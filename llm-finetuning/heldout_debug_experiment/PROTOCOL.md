# Protocol For Later Held-Out Debug Runs

## Purpose

Measure whether SMGR provenance plus a Localizer improves agent debugging speed and success rate on held-out gLayout prompt-to-layout tasks.

This directory only prepares the environment. Do not begin the experiment until the target agents and training setup are ready.

## Conditions

### With SMGR + Localizer

Allowed:

- Original held-out `.convo` prompt for the assigned task.
- Repository source code and existing public examples allowed by the run owner.
- Generated SMGR provenance sidecars for the agent's own attempts.
- Localizer reports or rankings derived from those sidecars and DRC/LVS failures.
- Raw Magic DRC reports and Netgen LVS reports.

Disallowed:

- Baseline condition logs, patches, intermediate artifacts, or result files.
- Manual hints produced by another agent working on the same prompt.

### Baseline Without SMGR + Localizer

Allowed:

- Original held-out `.convo` prompt for the assigned task.
- Repository source code and existing public examples allowed by the run owner.
- Raw Magic DRC reports and Netgen LVS reports.
- Normal code search, local reasoning, and direct inspection of generated Python/GDS outputs.

Disallowed:

- SMGR provenance sidecars, source maps, component trace graphs, or provenance-derived blame information.
- Localizer outputs, rankings, or any message that summarizes Localizer findings.
- With-SMGR condition logs, patches, intermediate artifacts, or result files.

## Recommended Run Shape

1. Assign an agent id, condition, and prompt.
2. Create a fresh git branch or worktree for that agent and condition.
3. Copy only the assigned prompt from that condition's `prompts/` directory into the active workspace.
4. Start a timer before the agent reads the prompt.
5. Let the agent implement, run DRC/LVS, debug, and stop only when it reaches pass/fail criteria or a time budget.
6. Record every DRC/LVS attempt in the run log.
7. Fill one row in `shared/scorecard_template.csv`.
8. Store any generated GDS, netlists, reports, provenance, Localizer output, and patches under that condition's `artifacts/` directory.

## Pass Criteria

A prompt counts as clean only when all of the following are true:

- The generated builder runs without Python errors.
- The generated GDS exists.
- Magic DRC has zero errors.
- Netgen LVS matches uniquely.
- For the with-SMGR condition, the SMGR provenance sidecar exists and any Localizer artifacts used are archived.

## Timing Metrics

Record at least:

- `wall_clock_minutes`: elapsed real time from first prompt read to final decision.
- `active_debug_minutes`: optional estimate excluding setup or waiting time.
- `assistant_turns`: number of agent response turns.
- `tool_calls`: total tool invocations.
- `drc_runs`: Magic DRC attempts.
- `lvs_runs`: Netgen LVS attempts.
- `python_failures`: Python/runtime failures before DRC/LVS.
- `final_status`: `clean`, `timeout`, `blocked`, or `failed`.

## Contamination Controls

- Do not reuse a patched implementation between conditions.
- Do not expose one condition's logs to the other condition.
- If the same model family is used in both conditions, prefer different prompt assignments or run baseline first to reduce memory/context contamination.
- If different agents run the same prompt pair in parallel, keep worktrees and server output directories separate.

## Analysis Ideas

Useful comparisons:

- Median wall-clock time to clean.
- Median number of DRC/LVS attempts.
- Clean rate under a fixed time budget.
- Number of human interventions required.
- Number of wrong hypotheses before the final fix.
- Whether Localizer suggestions reduce time spent reading raw LVS reports.
