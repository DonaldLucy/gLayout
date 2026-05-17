from __future__ import annotations

import argparse
import json
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
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def status_bits(case_result: dict[str, Any]) -> str:
    if not case_result.get("available"):
        return "no case_result"
    bits = []
    for key, label in (
        ("baseline_drc", "base_drc"),
        ("traced_drc", "traced_drc"),
        ("baseline_lvs", "base_lvs"),
        ("traced_lvs", "traced_lvs"),
    ):
        status = case_result.get(key) or {}
        clean = status.get("is_clean")
        detail = status.get("status")
        if detail is None:
            detail = status.get("error_count")
        if detail is None:
            detail = status.get("mismatch_markers")
        bits.append(f"{label}={clean}({detail})")
    return " ".join(bits)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact repair-bench summary.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--show-failed-logs", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    summary = load_json(run_dir / "summary.json") or {}
    clean_validation = load_json(run_dir / "clean_validation.json") or []
    rows = read_jsonl(run_dir / "dataset.jsonl")
    invalid = load_json(run_dir / "invalid_records.json") or []

    print(f"# Repair Bench Summary: {run_dir}")
    if summary:
        print(f"repo_commit: {summary.get('repo_commit')}")
        print(f"dataset: {summary.get('dataset_path')}")
        print(f"samples: {summary.get('total_samples')} invalid: {summary.get('invalid_samples')}")
        print(
            "localizer_topk: "
            f"{summary.get('localizer_topk_hits')}/{summary.get('total_samples')} "
            f"rate={summary.get('localizer_topk_hit_rate')}"
        )
    else:
        print("summary.json: missing")

    if clean_validation:
        print("\n## Clean Validation")
        for record in clean_validation:
            verdict = "PASS" if all(
                (record.get("case_result") or {}).get(section, {}).get("is_clean") is True
                for section in ("baseline_drc", "traced_drc", "baseline_lvs", "traced_lvs")
            ) else "FAIL"
            print(f"- {verdict} {record.get('case_id')}: {status_bits(record.get('case_result') or {})}")
            if verdict == "FAIL" and args.show_failed_logs:
                print(f"  log: {record.get('log_path')}")

    if rows:
        print("\n## Dataset Rows")
        print(f"rows: {len(rows)}")
        by_case: dict[str, dict[str, int]] = {}
        by_operator: dict[str, dict[str, int]] = {}
        for row in rows:
            hit = bool(row.get("localizer_hit", {}).get("hit"))
            for bucket, key in (
                (by_case, row.get("case_id", "?")),
                (by_operator, row.get("mutation", {}).get("operator", "?")),
            ):
                stats = bucket.setdefault(key, {"total": 0, "hit": 0, "ok": 0})
                stats["total"] += 1
                stats["hit"] += int(hit)
                stats["ok"] += int(row.get("verification", {}).get("returncode") == 0)
        print("by_case:")
        for case_id, stats in sorted(by_case.items()):
            print(f"- {case_id}: total={stats['total']} ok={stats['ok']} hit={stats['hit']}")
        print("by_operator:")
        for operator, stats in sorted(by_operator.items()):
            print(f"- {operator}: total={stats['total']} ok={stats['ok']} hit={stats['hit']}")
    else:
        print("\n## Dataset Rows")
        print("rows: 0")

    if invalid:
        print("\n## Invalid Verification Records")
        print(f"invalid: {len(invalid)}")
        for record in invalid[:10]:
            print(f"- {record.get('sample_id')} returncode={record.get('returncode')} log={record.get('log_path')}")
        if len(invalid) > 10:
            print(f"- ... {len(invalid) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
