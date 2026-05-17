from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({numerator / denominator:.3f})"


def short_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def sample_dir_for_result(result: dict[str, Any]) -> Path | None:
    prompt_path = result.get("prompt_path")
    if not isinstance(prompt_path, str):
        return None
    return Path(prompt_path).parent


def load_response(result: dict[str, Any]) -> str | None:
    sample_dir = sample_dir_for_result(result)
    if sample_dir is None:
        return None
    response_path = sample_dir / "response.txt"
    if not response_path.exists():
        return None
    return response_path.read_text(errors="replace")


def load_repair_action(result: dict[str, Any]) -> dict[str, Any] | None:
    sample_dir = sample_dir_for_result(result)
    if sample_dir is None:
        return None
    action_path = sample_dir / "repair_action.json"
    if not action_path.exists():
        return None
    try:
        return load_json(action_path)
    except json.JSONDecodeError:
        return None


def result_bucket(result: dict[str, Any]) -> str:
    if result.get("model_error"):
        return "model_error"
    if result.get("parse_success") is not True:
        return "parse_failed"
    if result.get("apply_success") is not True:
        return "apply_failed"
    if "verification_strict_clean" in result and result.get("verification_strict_clean") is not True:
        return "verification_failed"
    if result.get("verification_strict_clean") is True:
        return "verification_clean"
    return "applied_not_verified"


def summarize(summary: dict[str, Any], show_actions: bool, show_responses: bool, text_limit: int) -> str:
    results = summary.get("results") or []
    lines: list[str] = []
    aggregate = summary.get("aggregate") or {}
    lines.append("# Zero-Shot Repair Baseline Summary")
    lines.append("")
    lines.append(f"- Model: `{summary.get('model')}`")
    lines.append(f"- API base: `{summary.get('api_base')}`")
    lines.append(f"- Dataset: `{summary.get('dataset')}`")
    lines.append(f"- Selected rows: {summary.get('selected_rows')} / available {summary.get('available_rows')}")
    lines.append(f"- Unique mutations: {summary.get('unique_mutations')}")
    lines.append(f"- Run verification: {summary.get('run_verification')}")
    lines.append("")
    lines.append("## Aggregate")
    lines.append(f"- Parse success: {pct(int(aggregate.get('parse_success') or 0), int(aggregate.get('total') or 0))}")
    lines.append(f"- Apply success: {pct(int(aggregate.get('apply_success') or 0), int(aggregate.get('total') or 0))}")
    lines.append(f"- Exact expected action: {pct(int(aggregate.get('exact_expected_action') or 0), int(aggregate.get('total') or 0))}")
    verification_runs = int(aggregate.get("verification_runs") or 0)
    lines.append(
        f"- Verification strict-clean: "
        f"{pct(int(aggregate.get('verification_strict_clean') or 0), verification_runs)}"
    )
    lines.append("")

    by_bucket: dict[str, int] = defaultdict(int)
    by_operator: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in results:
        bucket = result_bucket(result)
        operator = result.get("operator") or "unknown"
        by_bucket[bucket] += 1
        by_operator[operator]["total"] += 1
        by_operator[operator][bucket] += 1
        if result.get("parse_success") is True:
            by_operator[operator]["parse_success"] += 1
        if result.get("apply_success") is True:
            by_operator[operator]["apply_success"] += 1
        if result.get("exact_expected_action") is True:
            by_operator[operator]["exact_expected_action"] += 1
        if result.get("verification_strict_clean") is True:
            by_operator[operator]["verification_strict_clean"] += 1

    lines.append("## Outcome Buckets")
    for bucket, count in sorted(by_bucket.items()):
        lines.append(f"- {bucket}: {count}")
    lines.append("")

    lines.append("## By Operator")
    lines.append("operator,total,parse_success,apply_success,exact_expected_action,verification_strict_clean,model_error,apply_failed")
    for operator, stats in sorted(by_operator.items()):
        lines.append(
            ",".join(
                str(value)
                for value in [
                    operator,
                    stats["total"],
                    stats["parse_success"],
                    stats["apply_success"],
                    stats["exact_expected_action"],
                    stats["verification_strict_clean"],
                    stats["model_error"],
                    stats["apply_failed"],
                ]
            )
        )
    lines.append("")

    lines.append("## Per Sample")
    for result in results:
        bucket = result_bucket(result)
        lines.append(
            f"- {result.get('sample_id')} | {result.get('case_id')} | {result.get('operator')} | "
            f"{result.get('mutation_id')} | {bucket}"
        )
        if result.get("model_error"):
            lines.append(f"  model_error: {short_text(result.get('model_error'), text_limit)}")
        for apply_result in result.get("apply_results") or []:
            action = apply_result.get("action") or {}
            lines.append(
                f"  apply: success={apply_result.get('success')} "
                f"message={short_text(apply_result.get('message'), text_limit)} "
                f"file={short_text(action.get('file'), text_limit)}"
            )
            if show_actions:
                lines.append(f"  find: {short_text(action.get('find'), text_limit)}")
                lines.append(f"  replace: {short_text(action.get('replace'), text_limit)}")
        if show_responses:
            response = load_response(result)
            if response is not None:
                lines.append(f"  response: {short_text(response, text_limit)}")
        elif show_actions:
            action_payload = load_repair_action(result)
            if action_payload is not None:
                lines.append(f"  repair_action_json: {short_text(json.dumps(action_payload, sort_keys=True), text_limit)}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a zero-shot repair baseline run.")
    parser.add_argument("run_dir", type=Path, help="Directory containing zero_shot_summary.json.")
    parser.add_argument("--show-actions", action="store_true", help="Print model-proposed find/replace actions.")
    parser.add_argument("--show-responses", action="store_true", help="Print truncated raw model responses.")
    parser.add_argument("--text-limit", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.run_dir / "zero_shot_summary.json"
    summary = load_json(summary_path)
    text = summarize(summary, args.show_actions, args.show_responses, args.text_limit)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
