# Qwen Repair Loop Experiment

This folder contains a small closed-loop experiment for using an OpenAI-compatible Qwen2.5 Coder 14B endpoint as a verification repair agent.

The loop starts from an existing failing gLayout generator, runs the SMGR verification locator, builds a compact repair prompt, asks the model for a unified diff, applies the patch in an isolated workspace, and repeats.

## Why This Exists

The full locator artifact can be thousands of lines. A 32K-context coding model should not need the whole report. The experiment keeps context small by combining:

- `glayout_skill_compact.md`: stable gLayout repair rules, intended to stay under 1500 lines.
- `repair_packet.json`: compact per-case DRC/LVS topology, label, source-span, and hint data.
- A short loop summary from previous iterations.
- The model response as a unified diff only.

## Requirements

- Run from the repository root.
- A server environment that can run the existing DRC/LVS flow.
- An OpenAI-compatible model endpoint, for example vLLM serving `Qwen/Qwen2.5-Coder-14B-Instruct`.

Environment variables:

```bash
export QWEN_API_BASE="http://localhost:8000/v1"
export QWEN_API_KEY="EMPTY"
export QWEN_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
```

## Dry Run

Generate the first verification packet and prompt without calling the model:

```bash
python experiments/qwen_repair_loop/run_qwen_repair_loop.py \
  --case diff_pair_ibias \
  --run-root build/qwen_repair_loop_dryrun \
  --max-iters 1 \
  --dry-run
```

## Live Loop

```bash
python experiments/qwen_repair_loop/run_qwen_repair_loop.py \
  --case diff_pair_ibias \
  --run-root build/qwen_repair_loop_diff_pair_ibias \
  --max-iters 4 \
  --workspace-mode worktree
```

Outputs are written under the run root:

- `workspace/`: isolated worktree or copied repo.
- `iter_00/verification.log`: verifier output.
- `iter_00/verification/<case>/repair_packet.json`: compact locator packet.
- `iter_00/prompt.md`: prompt sent to Qwen.
- `iter_00/model_response.txt`: raw model response.
- `iter_00/model.patch`: extracted unified diff.
- `iter_00/apply.log`: patch application result.
- `summary.json`: loop status and artifact paths.

## Analyze a Run

After a live loop, summarize what the model did and which failure modes are useful
for prompt tuning or SFT:

```bash
python experiments/qwen_repair_loop/analyze_qwen_repair_run.py \
  /tmp/qwen_repair_loop_diff_pair_ibias_v4
```

The analyzer writes:

- `analysis.md`: human-readable iteration table, failure categories, and training hooks.
- `analysis.json`: machine-readable per-iteration patch/path/apply diagnostics, including
  source-span file violations and duplicate/stale edit signals.

Useful quick views:

```bash
RUN=/tmp/qwen_repair_loop_diff_pair_ibias_v4
sed -n '1,220p' $RUN/analysis.md
jq '.iterations[] | {iteration, issues, apply_failure_category, changed_files}' $RUN/analysis.json
```

## Notes

- The loop never intentionally edits the original checkout. It patches only the isolated workspace.
- Use `--workspace-mode copy` if Git worktrees are inconvenient.
- Use `--apply-command patch` if the model emits a diff that `git apply` rejects but `patch -p1` can consume.
- By default, patches are guarded before apply: the loop rejects edits outside
  repair-packet source spans and patches that mostly re-add lines already present.
  Use `--allow-non-source-span-files` only for broad refactor/debug experiments.
- If the model repeatedly emits non-diff prose, lower the prompt budgets or add a stronger system prompt in the serving wrapper.
