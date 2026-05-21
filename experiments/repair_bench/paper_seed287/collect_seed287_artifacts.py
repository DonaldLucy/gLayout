from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("build/repair_bench_validated12_unique287")
DEFAULT_OUTPUT_DIR = Path("experiments/repair_bench/paper_seed287/artifacts")
SCRIPT_DIR = Path(__file__).resolve().parent


TOP_LEVEL_FILES = (
    "plan.json",
    "summary.json",
    "clean_validation.json",
    "invalid_records.json",
    "dataset.jsonl",
    "dataset_drc_lvs_repair.jsonl",
)

METRIC_FILES = (
    "metrics/metrics.json",
    "metrics/samples.csv",
    "metrics/by_case.csv",
    "metrics/by_operator.csv",
    "metrics/localizer_failures.csv",
)

ZERO_SHOT_FILES = (
    "zero_shot_summary.json",
    "summary.md",
    "by_operator_zero_shot.csv",
    "by_operator_zero_shot.png",
    "by_operator_zero_shot.svg",
)

ZERO_SHOT_SAMPLE_FILES = (
    "prompt.md",
    "response.txt",
    "repair_action.json",
)


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def copy_file(src: Path, dst: Path, copied: list[str]) -> None:
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def representative_ids(extra_ids: list[str]) -> list[str]:
    ids: list[str] = []
    path = SCRIPT_DIR / "representative_samples.json"
    if path.exists():
        payload = json.loads(path.read_text())
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("sample_id"), str):
                ids.append(item["sample_id"])
    for sample_id in extra_ids:
        if sample_id not in ids:
            ids.append(sample_id)
    return ids


def case_result_files(sample_dir: Path) -> list[Path]:
    if not sample_dir.exists():
        return []
    keep_names = {
        "case_result.json",
        "repair_packet.json",
        "verification_locator.json",
        "verification.log",
    }
    return [path for path in sample_dir.rglob("*") if path.name in keep_names and path.is_file()]


def collect_bench_sample(run_root: Path, output_dir: Path, sample_id: str, copied: list[str]) -> None:
    src_dir = run_root / "samples" / sample_id
    if not src_dir.exists():
        return
    dst_dir = output_dir / "representative_samples" / sample_id
    copy_file(src_dir / "sample.json", dst_dir / "sample.json", copied)
    for path in case_result_files(src_dir):
        rel = path.relative_to(src_dir)
        copy_file(path, dst_dir / rel, copied)


def collect_zero_shot_sample(run_dir: Path, output_dir: Path, sample_id: str, copied: list[str]) -> None:
    src_dir = run_dir / sample_id
    if not src_dir.exists():
        return
    dst_dir = output_dir / "zero_shot_runs" / run_dir.name / "representative_samples" / sample_id
    for name in ZERO_SHOT_SAMPLE_FILES:
        copy_file(src_dir / name, dst_dir / name, copied)
    for path in case_result_files(src_dir):
        rel = path.relative_to(src_dir)
        if rel.parts and rel.parts[0] == "workspace":
            continue
        copy_file(path, dst_dir / rel, copied)


def write_manifest(output_dir: Path, copied: list[str], run_root: Path, sample_ids: list[str]) -> None:
    rel_files = [str(Path(path).relative_to(output_dir)) for path in copied]
    manifest = {
        "run_root": str(run_root),
        "representative_sample_ids": sample_ids,
        "copied_file_count": len(rel_files),
        "files": sorted(rel_files),
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    lines = ["# Seed287 Artifact Manifest", "", f"- run_root: `{run_root}`", f"- files: {len(rel_files)}", ""]
    lines.extend(f"- `{path}`" for path in sorted(rel_files))
    (output_dir / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect paper-safe seed287 benchmark artifacts.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-zero-shot", action="store_true")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root
    output_dir = args.output_dir
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in TOP_LEVEL_FILES:
        copy_file(run_root / name, output_dir / "run_root" / name, copied)
    for name in METRIC_FILES:
        copy_file(run_root / name, output_dir / "run_root" / name, copied)

    sample_ids = representative_ids(args.sample_id)
    copy_file(SCRIPT_DIR / "representative_samples.json", output_dir / "representative_samples.json", copied)
    for sample_id in sample_ids:
        collect_bench_sample(run_root, output_dir, sample_id, copied)

    if args.include_zero_shot:
        for run_dir in sorted(run_root.glob("zero_shot*")):
            if not run_dir.is_dir():
                continue
            for name in ZERO_SHOT_FILES:
                copy_file(run_dir / name, output_dir / "zero_shot_runs" / run_dir.name / name, copied)
            for sample_id in sample_ids:
                collect_zero_shot_sample(run_dir, output_dir, sample_id, copied)

    write_manifest(output_dir, copied, run_root, sample_ids)
    print(f"copied files: {len(copied)}")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
