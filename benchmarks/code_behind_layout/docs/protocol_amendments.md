# Protocol amendments

Each amendment was approved before any affected dispatch and adds a declared, separately recorded procedure. No frozen manifest, bound source, budget, task identity, hidden configuration or recorded trial is modified. New cohorts use Claude Opus 5.5 as the policy, through its command-line client with all tools disabled.

## A1. Policy cohorts E1-C and E3-C (Claude Opus 5.5)

- The 24 tasks, anchors, eight visible and sixteen hidden configurations, references, benches, specifications, budgets, selection rules and analysis scripts are the same as for the GPT-6.1 cohorts. The budgets are: E1 1,200 s, 10 decisions and 3 full evaluations; E3 4 h, 24 decisions, 8 patches and 8 full evaluations; 16,384-token context and 1,024 output tokens.
- Only `model_config` (backend `claude_cli`, fresh session per decision) and the adapter implementation change. The cohort freeze binds the new adapter.
- Scope:
  - E1: 24 tasks × 3 levels × 2 trials = 144 episodes.
  - E3: A/R/B/C 24 tasks × 4 rules × 3 trials = 288 searches.
  - Parameter baseline P: 4 tasks × 3 trials × 2 variants = 24.
  - C signal ablations: 4 tasks × 3 trials × 2 variants = 24.

## A2. Hidden release per cohort

A cohort's hidden configurations are evaluated once, after every declared search of that cohort is sealed.

## A4. E2 transient-completion pass

Some oracle candidates are unknown because a transient analysis exceeded the 90 s inner limit. Each is re-simulated once with a 4× inner limit, on the retained extracted netlist and an identical deck, and labelled `transient_completion_v1`. The finite-pool oracle is reported both under the frozen rule and under the completion pass.

## A5. Parallel oracle execution

Additional `e2_four_source.py run --method full_oracle --pool-id <pool>` processes run disjoint pools under the same frozen manifest. Per-pool reservation files created with `O_EXCL` prevent duplicate execution.

## A6. Format guidance in the Claude policy prompt (cohort E1-C2)

- **Observation.** Without format guidance, most E1-C episodes ended at the first decision: the action exceeded the 1,024-token output budget by 6 to 41 percent, because of long summaries and context-heavy patches.
- **Change.** Cohort E1-C2 adds format rules to the fixed system prompt only:
  - a one-sentence summary;
  - the whole action within 1,024 tokens;
  - strict unified diffs with exact hunk counts and three context lines.
- **Unchanged.** Tasks, hidden sets, budgets, gateway, selection rules, adapter mechanics and the 144-episode matrix.
- **Reporting.** The GPT-6.1 cohort is reported beside E1-C2 as the no-guidance condition.
