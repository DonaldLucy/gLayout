from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("build/repair_bench_validated12_unique287")
DEFAULT_OUTPUT_DIR = Path("experiments/repair_bench/paper_seed287/benchmark")
SCRIPT_DIR = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_representative_ids() -> list[str]:
    path = SCRIPT_DIR / "representative_samples.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [
        item["sample_id"]
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("sample_id"), str)
    ]


def find_dataset(run_root: Path, dataset: Path | None) -> tuple[Path, bool]:
    if dataset is not None:
        return dataset, dataset.name == "dataset_drc_lvs_repair.jsonl"
    preferred = run_root / "dataset_drc_lvs_repair.jsonl"
    if preferred.exists():
        return preferred, True
    fallback = run_root / "dataset.jsonl"
    if fallback.exists():
        return fallback, False
    raise FileNotFoundError(
        f"Could not find dataset_drc_lvs_repair.jsonl or dataset.jsonl under {run_root}"
    )


def looks_like_repair_record(row: dict[str, Any]) -> bool:
    expected = row.get("expected_repair_action")
    verification = row.get("verification")
    mutation = row.get("mutation")
    return isinstance(expected, dict) and isinstance(verification, dict) and isinstance(mutation, dict)


def unique_by_sample_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in seen:
            continue
        seen.add(sample_id)
        unique.append(row)
    return unique


def operator(row: dict[str, Any]) -> str:
    mutation = row.get("mutation") or {}
    return str(mutation.get("operator") or "unknown")


def case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or "unknown")


def balanced_by_operator(rows: list[dict[str, Any]], per_operator: int) -> list[dict[str, Any]]:
    if per_operator <= 0:
        return []
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_operator[operator(row)].append(row)

    selected: list[dict[str, Any]] = []
    for op in sorted(by_operator):
        candidates = sorted(by_operator[op], key=lambda row: (case_id(row), str(row.get("sample_id"))))
        bucket: list[dict[str, Any]] = []
        used_cases: set[str] = set()
        for row in candidates:
            cid = case_id(row)
            if cid in used_cases:
                continue
            bucket.append(row)
            used_cases.add(cid)
            if len(bucket) >= per_operator:
                break
        if len(bucket) < per_operator:
            chosen_ids = {row.get("sample_id") for row in bucket}
            for row in candidates:
                if row.get("sample_id") in chosen_ids:
                    continue
                bucket.append(row)
                if len(bucket) >= per_operator:
                    break
        selected.extend(bucket)
    return selected


def representative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row.get("sample_id"): row for row in rows}
    return [by_id[sample_id] for sample_id in load_representative_ids() if sample_id in by_id]


def write_shards(output_dir: Path, rows: list[dict[str, Any]], shards: int) -> list[str]:
    shard_paths: list[str] = []
    if shards <= 1:
        return shard_paths
    buckets = [[] for _ in range(shards)]
    for index, row in enumerate(rows):
        buckets[index % shards].append(row)
    for shard_index, bucket in enumerate(buckets):
        path = output_dir / f"seed287_full_repair_shard{shard_index:02d}_of{shards:02d}.jsonl"
        write_jsonl(path, bucket)
        shard_paths.append(str(path))
    return shard_paths


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "by_case": dict(sorted(Counter(case_id(row) for row in rows).items())),
        "by_operator": dict(sorted(Counter(operator(row) for row in rows).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create paper-ready seed287 benchmark JSONL splits.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-operator", type=int, default=5)
    parser.add_argument("--smoke-per-operator", type=int, default=3)
    parser.add_argument("--shards", type=int, default=0, help="Optional round-robin shard count for full repair JSONL.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path, already_repair_filtered = find_dataset(args.run_root, args.dataset)
    all_rows = unique_by_sample_id(read_jsonl(dataset_path))
    repair_rows = all_rows if already_repair_filtered else [row for row in all_rows if looks_like_repair_record(row)]

    full_path = output_dir / "seed287_full_repair.jsonl"
    smoke_path = output_dir / "seed287_smoke_operator3.jsonl"
    balanced_path = output_dir / f"seed287_balanced_operator{args.per_operator}.jsonl"
    representative_path = output_dir / "seed287_representative10.jsonl"

    smoke_rows = balanced_by_operator(repair_rows, args.smoke_per_operator)
    balanced_rows = balanced_by_operator(repair_rows, args.per_operator)
    reps = representative_rows(repair_rows)

    write_jsonl(full_path, repair_rows)
    write_jsonl(smoke_path, smoke_rows)
    write_jsonl(balanced_path, balanced_rows)
    write_jsonl(representative_path, reps)
    shard_paths = write_shards(output_dir, repair_rows, args.shards)

    manifest = {
        "source_dataset": str(dataset_path),
        "source_was_dataset_drc_lvs_repair": already_repair_filtered,
        "outputs": {
            "full_repair": str(full_path),
            "smoke_operator3": str(smoke_path),
            "balanced_operator": str(balanced_path),
            "representative10": str(representative_path),
            "full_repair_shards": shard_paths,
        },
        "all_input": counts(all_rows),
        "full_repair": counts(repair_rows),
        "smoke_operator3": counts(smoke_rows),
        "balanced_operator": counts(balanced_rows),
        "representative10": counts(reps),
    }
    write_json(output_dir / "manifest.json", manifest)

    print(f"source_dataset: {dataset_path}")
    print(f"full_repair rows: {len(repair_rows)} -> {full_path}")
    print(f"smoke rows: {len(smoke_rows)} -> {smoke_path}")
    print(f"balanced rows: {len(balanced_rows)} -> {balanced_path}")
    print(f"representative rows: {len(reps)} -> {representative_path}")
    if shard_paths:
        print(f"shards: {len(shard_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
