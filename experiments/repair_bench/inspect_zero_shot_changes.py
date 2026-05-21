from __future__ import annotations

import argparse
import difflib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def short_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def sample_dir_for_result(result: dict[str, Any], run_dir: Path) -> Path:
    prompt_path = result.get("prompt_path")
    if isinstance(prompt_path, str):
        path = Path(prompt_path)
        if not path.is_absolute():
            path = run_dir / path.relative_to(run_dir) if str(path).startswith(str(run_dir)) else path
        return path.parent
    return run_dir / str(result.get("sample_id", "unknown"))


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


def repair_actions(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    actions = payload.get("repair_actions")
    if isinstance(actions, list):
        return [action for action in actions if isinstance(action, dict)]
    if all(key in payload for key in ("type", "file", "find", "replace")):
        return [payload]
    return []


def resolve_dataset_path(summary: dict[str, Any], run_dir: Path, repo_root: Path) -> Path | None:
    dataset = summary.get("dataset")
    if not isinstance(dataset, str):
        return None
    path = Path(dataset)
    if path.is_absolute() and path.exists():
        return path
    for candidate in (repo_root / path, run_dir / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def workspace_diff(repo_root: Path, sample_dir: Path, relpath: str, max_lines: int) -> str:
    clean_path = repo_root / relpath
    repaired_path = sample_dir / "workspace" / relpath
    if not clean_path.exists():
        return f"[diff unavailable: clean file missing] {clean_path}"
    if not repaired_path.exists():
        return f"[diff unavailable: repaired file missing] {repaired_path}"
    clean_lines = clean_path.read_text(errors="replace").splitlines(keepends=True)
    repaired_lines = repaired_path.read_text(errors="replace").splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            clean_lines,
            repaired_lines,
            fromfile=f"clean/{relpath}",
            tofile=f"qwen/{relpath}",
        )
    )
    if not diff_lines:
        return "[no diff vs clean repo]"
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines] + [f"... truncated {len(diff_lines) - max_lines} diff lines ...\n"]
    return "".join(diff_lines).rstrip()


def action_files(actions: Iterable[dict[str, Any]]) -> list[str]:
    files: list[str] = []
    for action in actions:
        file_path = action.get("file")
        if isinstance(file_path, str) and file_path not in files:
            files.append(file_path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect zero-shot model repairs and diffs.")
    parser.add_argument("run_dir", type=Path, help="Directory containing zero_shot_summary.json.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-id", action="append", default=None, help="Inspect one sample id. Repeatable.")
    parser.add_argument("--bucket", default=None, help="Filter by outcome bucket, e.g. verification_failed.")
    parser.add_argument("--operator", default=None, help="Filter by mutation operator.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--show-response", action="store_true")
    parser.add_argument("--show-expected", action="store_true")
    parser.add_argument("--show-diff", action="store_true")
    parser.add_argument("--text-limit", type=int, default=500)
    parser.add_argument("--diff-lines", type=int, default=160)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    repo_root = args.repo_root.resolve()
    summary = load_json(run_dir / "zero_shot_summary.json") or {}
    results = summary.get("results") or []
    if not isinstance(results, list):
        results = []

    dataset_path = resolve_dataset_path(summary, run_dir, repo_root)
    dataset_by_id = {
        row.get("sample_id"): row
        for row in read_jsonl(dataset_path) if dataset_path is not None
    }

    selected: list[dict[str, Any]] = []
    sample_filter = set(args.sample_id or [])
    for result in results:
        if not isinstance(result, dict):
            continue
        if sample_filter and result.get("sample_id") not in sample_filter:
            continue
        if args.bucket and result_bucket(result) != args.bucket:
            continue
        if args.operator and result.get("operator") != args.operator:
            continue
        selected.append(result)
        if args.limit > 0 and len(selected) >= args.limit:
            break

    print(f"# Inspect Zero-Shot Changes: {run_dir}")
    print(f"selected: {len(selected)}")
    if dataset_path is not None:
        print(f"dataset: {dataset_path}")
    print()

    for result in selected:
        sample_id = str(result.get("sample_id"))
        bucket = result_bucket(result)
        sample_dir = sample_dir_for_result(result, run_dir)
        payload = load_json(sample_dir / "repair_action.json")
        actions = repair_actions(payload)
        dataset_row = dataset_by_id.get(sample_id) or {}
        expected = dataset_row.get("expected_repair_action") or {}
        mutation = dataset_row.get("mutation") or {}

        print(f"## {sample_id}")
        print(f"case={result.get('case_id')} operator={result.get('operator')} mutation={result.get('mutation_id')}")
        print(
            "status="
            f"{bucket} parse={result.get('parse_success')} apply={result.get('apply_success')} "
            f"clean={result.get('verification_strict_clean')}"
        )
        if result.get("model_error"):
            print(f"model_error={short_text(result.get('model_error'), args.text_limit)}")
        for item in result.get("apply_results") or []:
            action = item.get("action") or {}
            print(
                "apply "
                f"success={item.get('success')} message={short_text(item.get('message'), args.text_limit)} "
                f"file={action.get('file')}"
            )
            print(f"  find={short_text(action.get('find'), args.text_limit)}")
            print(f"  replace={short_text(action.get('replace'), args.text_limit)}")
        if args.show_expected and expected:
            print("expected_repair_action:")
            print(f"  file={expected.get('file')}")
            print(f"  find={short_text(expected.get('find'), args.text_limit)}")
            print(f"  replace={short_text(expected.get('replace'), args.text_limit)}")
        if args.show_response:
            response_path = sample_dir / "response.txt"
            if response_path.exists():
                print("response:")
                print(short_text(response_path.read_text(errors="replace"), args.text_limit))
        if args.show_diff:
            files = action_files(actions)
            mutation_file = mutation.get("file_path")
            if isinstance(mutation_file, str) and mutation_file not in files:
                files.append(mutation_file)
            for relpath in files:
                print(f"diff {relpath}:")
                print(workspace_diff(repo_root, sample_dir, relpath, args.diff_lines))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
