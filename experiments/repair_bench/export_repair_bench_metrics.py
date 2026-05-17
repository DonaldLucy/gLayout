from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


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


def percent(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if denominator in (None, 0):
        return None
    if numerator is None:
        return None
    return float(numerator) / float(denominator)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def flatten_status(prefix: str, case_result: dict[str, Any]) -> dict[str, Any]:
    status = case_result.get(prefix) if isinstance(case_result, dict) else None
    if not isinstance(status, dict):
        status = {}
    return {
        f"{prefix}_is_clean": status.get("is_clean"),
        f"{prefix}_status": status.get("status"),
        f"{prefix}_matched": status.get("matched"),
        f"{prefix}_error_count": status.get("error_count"),
        f"{prefix}_mismatch_markers": status.get("mismatch_markers"),
    }


def failure_kind(row: dict[str, Any]) -> str:
    drc = row.get("traced_drc_is_clean")
    lvs = row.get("traced_lvs_is_clean")
    if drc is False and lvs is False:
        return "drc_lvs"
    if drc is False:
        return "drc"
    if lvs is False:
        return "lvs"
    if drc is True and lvs is True:
        return "verification_clean"
    return "unknown"


def hit_mode(localizer_hit: dict[str, Any]) -> str:
    source_text = bool(localizer_hit.get("source_span_text_hit"))
    candidate = bool(localizer_hit.get("candidate_location_hit"))
    if source_text and candidate:
        return "source_text_and_ranked_candidate"
    if source_text:
        return "source_text"
    if candidate:
        return "ranked_candidate"
    if localizer_hit.get("source_span_file_hit"):
        return "source_file_only"
    return "none"


def first_hit_rank(localizer_hit: dict[str, Any]) -> int | None:
    ranks = [
        item.get("rank")
        for item in localizer_hit.get("ranked_hits", [])
        if isinstance(item, dict) and isinstance(item.get("rank"), int)
    ]
    return min(ranks) if ranks else None


def sample_metric_row(run_dir: Path, summary: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    mutation = row.get("mutation") or {}
    target = row.get("target") or {}
    verification = row.get("verification") or {}
    case_result = verification.get("case_result") or {}
    localizer_hit = row.get("localizer_hit") or {}
    metric = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "repo_commit": row.get("repo_commit") or summary.get("repo_commit"),
        "sample_id": row.get("sample_id"),
        "case_id": row.get("case_id"),
        "mutation_id": mutation.get("mutation_id"),
        "operator": mutation.get("operator"),
        "description": mutation.get("description"),
        "target_file": target.get("file") or localizer_hit.get("target_file"),
        "target_line": target.get("focus_line") or localizer_hit.get("target_line"),
        "verification_returncode": verification.get("returncode"),
        "verification_elapsed_sec": verification.get("elapsed_sec"),
        "case_result_available": case_result.get("available"),
        "localizer_hit": bool(localizer_hit.get("hit")),
        "localizer_hit_mode": hit_mode(localizer_hit),
        "localizer_first_hit_rank": first_hit_rank(localizer_hit),
        "source_span_file_hit": bool(localizer_hit.get("source_span_file_hit")),
        "source_span_text_hit": bool(localizer_hit.get("source_span_text_hit")),
        "candidate_location_hit": bool(localizer_hit.get("candidate_location_hit")),
        "ranked_hit_count": len(localizer_hit.get("ranked_hits") or []),
        "top_k": localizer_hit.get("top_k"),
        "line_window": localizer_hit.get("line_window"),
    }
    for section in ("baseline_drc", "traced_drc", "baseline_lvs", "traced_lvs"):
        metric.update(flatten_status(section, case_result))
    metric["failure_kind"] = failure_kind(metric)
    metric["bug_detected"] = metric["failure_kind"] not in {"verification_clean", "unknown"}
    return metric


def clean_metric_row(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    case_result = record.get("case_result") or {}
    row = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "case_id": record.get("case_id"),
        "returncode": record.get("returncode"),
        "elapsed_sec": record.get("elapsed_sec"),
        "log_path": record.get("log_path"),
        "case_result_available": case_result.get("available"),
    }
    for section in ("baseline_drc", "traced_drc", "baseline_lvs", "traced_lvs"):
        row.update(flatten_status(section, case_result))
    row["strict_clean"] = all(
        row.get(f"{section}_is_clean") is True
        for section in ("baseline_drc", "traced_drc", "baseline_lvs", "traced_lvs")
    )
    return row


def group_rows(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)

    aggregates: list[dict[str, Any]] = []
    for key_values, group in sorted(
        buckets.items(),
        key=lambda item: tuple("" if v is None else str(v) for v in item[0]),
    ):
        total = len(group)
        verification_ok = sum(1 for row in group if row.get("verification_returncode") == 0)
        detected = sum(1 for row in group if row.get("bug_detected") is True)
        hits = sum(1 for row in group if row.get("localizer_hit") is True)
        source_text_hits = sum(1 for row in group if row.get("source_span_text_hit") is True)
        candidate_hits = sum(1 for row in group if row.get("candidate_location_hit") is True)
        elapsed = [
            float(row["verification_elapsed_sec"])
            for row in group
            if isinstance(row.get("verification_elapsed_sec"), (int, float))
        ]
        ranks = [
            int(row["localizer_first_hit_rank"])
            for row in group
            if isinstance(row.get("localizer_first_hit_rank"), int)
        ]
        aggregate = {key: value for key, value in zip(keys, key_values)}
        aggregate.update(
            {
                "total": total,
                "verification_ok": verification_ok,
                "verification_ok_rate": percent(verification_ok, total),
                "bug_detected": detected,
                "bug_detected_rate": percent(detected, total),
                "localizer_hits": hits,
                "localizer_hit_rate": percent(hits, total),
                "source_span_text_hits": source_text_hits,
                "source_span_text_hit_rate": percent(source_text_hits, total),
                "candidate_location_hits": candidate_hits,
                "candidate_location_hit_rate": percent(candidate_hits, total),
                "avg_elapsed_sec": (sum(elapsed) / len(elapsed)) if elapsed else None,
                "p50_elapsed_sec": percentile(elapsed, 0.50),
                "p90_elapsed_sec": percentile(elapsed, 0.90),
                "avg_first_hit_rank": (sum(ranks) / len(ranks)) if ranks else None,
            }
        )
        aggregates.append(aggregate)
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def collect_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = load_json(run_dir / "summary.json") or {}
    sample_rows = [
        sample_metric_row(run_dir, summary, row)
        for row in read_jsonl(run_dir / "dataset.jsonl")
    ]
    clean_rows = [
        clean_metric_row(run_dir, record)
        for record in (load_json(run_dir / "clean_validation.json") or [])
        if isinstance(record, dict)
    ]
    invalid_rows = load_json(run_dir / "invalid_records.json") or []
    if not isinstance(invalid_rows, list):
        invalid_rows = []
    run_info = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "repo_commit": summary.get("repo_commit"),
        "dataset_path": summary.get("dataset_path") or str(run_dir / "dataset.jsonl"),
        "summary_total_samples": summary.get("total_samples"),
        "summary_invalid_samples": summary.get("invalid_samples"),
        "summary_localizer_topk_hits": summary.get("localizer_topk_hits"),
        "summary_localizer_topk_hit_rate": summary.get("localizer_topk_hit_rate"),
        "loaded_sample_rows": len(sample_rows),
        "loaded_clean_rows": len(clean_rows),
        "loaded_invalid_rows": len(invalid_rows),
    }
    return run_info, sample_rows, clean_rows, invalid_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export repair-bench plot-ready metrics.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON metrics. Defaults to <run_dir>/metrics for one run.",
    )
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [run_dir.resolve() for run_dir in args.run_dirs]
    if args.output_dir is None:
        if len(run_dirs) != 1:
            raise SystemExit("--output-dir is required when exporting multiple runs")
        output_dir = run_dirs[0] / "metrics"
    else:
        output_dir = args.output_dir.resolve()

    run_infos: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_info, run_samples, run_clean, run_invalid = collect_run(run_dir)
        run_infos.append(run_info)
        sample_rows.extend(run_samples)
        clean_rows.extend(run_clean)
        for row in run_invalid:
            if isinstance(row, dict):
                invalid_rows.append({"run_dir": str(run_dir), "run_name": run_dir.name, **row})

    by_run = group_rows(sample_rows, ("run_name",))
    by_case = group_rows(sample_rows, ("case_id",))
    by_operator = group_rows(sample_rows, ("operator",))
    by_case_operator = group_rows(sample_rows, ("case_id", "operator"))
    by_mutation = group_rows(sample_rows, ("case_id", "mutation_id", "operator"))
    by_failure_kind = group_rows(sample_rows, ("failure_kind",))
    overall = group_rows(sample_rows, tuple())
    localizer_failures = [row for row in sample_rows if row.get("localizer_hit") is not True]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "sample_metrics.csv", sample_rows)
    write_csv(output_dir / "by_run.csv", by_run)
    write_csv(output_dir / "by_case.csv", by_case)
    write_csv(output_dir / "by_operator.csv", by_operator)
    write_csv(output_dir / "by_case_operator.csv", by_case_operator)
    write_csv(output_dir / "by_mutation.csv", by_mutation)
    write_csv(output_dir / "by_failure_kind.csv", by_failure_kind)
    write_csv(output_dir / "clean_validation.csv", clean_rows)
    write_csv(output_dir / "invalid_records.csv", invalid_rows)
    write_csv(output_dir / "localizer_failures.csv", localizer_failures)
    write_json(
        output_dir / "metrics.json",
        {
            "runs": run_infos,
            "overall": overall[0] if overall else {},
            "by_run": by_run,
            "by_case": by_case,
            "by_operator": by_operator,
            "by_case_operator": by_case_operator,
            "by_mutation": by_mutation,
            "by_failure_kind": by_failure_kind,
            "clean_validation": clean_rows,
            "invalid_records": invalid_rows,
            "localizer_failures": localizer_failures,
        },
    )

    print(f"[repair-bench-metrics] wrote {output_dir}")
    print(f"[repair-bench-metrics] sample rows: {len(sample_rows)}")
    print(f"[repair-bench-metrics] localizer failures: {len(localizer_failures)}")
    if args.print_summary:
        summary = overall[0] if overall else {"total": 0}
        print(
            "[repair-bench-metrics] overall "
            f"hit={summary.get('localizer_hits')}/{summary.get('total')} "
            f"rate={summary.get('localizer_hit_rate')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
