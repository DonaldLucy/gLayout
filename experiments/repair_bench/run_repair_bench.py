from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from mutation_specs import CASE_PROFILES, DEFAULT_STRICT_CLEAN_CASES, MUTATION_SPECS, MutationSpec
except ImportError:  # pragma: no cover - supports python -m execution.
    from experiments.repair_bench.mutation_specs import (
        CASE_PROFILES,
        DEFAULT_STRICT_CLEAN_CASES,
        MUTATION_SPECS,
        MutationSpec,
    )


IGNORE_COPY_PATTERNS = shutil.ignore_patterns(
    ".git",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.gds",
    "*.oas",
    "*.rpt",
    "*.ext",
    "*.sim",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(data, sort_keys=True, default=_json_default) + "\n")


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def normalize_repo_path(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    for marker in ("/src/", "/tests/", "/scripts/", "/experiments/"):
        if marker in text:
            return text[text.index(marker) + 1 :]
    while text.startswith("./"):
        text = text[2:]
    if text.startswith("workspace/"):
        text = text[len("workspace/") :]
    return text


def repo_sha(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def copy_repo(repo_root: Path, workspace: Path, force: bool) -> None:
    if workspace.exists() and force:
        shutil.rmtree(workspace)
    if workspace.exists():
        return
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root, workspace, ignore=IGNORE_COPY_PATTERNS)


def source_context(text: str, start_line: int, lines_before: int = 18, lines_after: int = 18) -> dict[str, Any]:
    lines = text.splitlines()
    first = max(start_line - lines_before, 1)
    last = min(start_line + lines_after, len(lines))
    excerpt = "\n".join(f"{idx:04d}: {lines[idx - 1]}" for idx in range(first, last + 1))
    return {"start_line": first, "end_line": last, "text": excerpt}


def find_focus_line(source_text: str, needle: str) -> int | None:
    offset = source_text.find(needle)
    if offset < 0:
        return None
    return source_text[:offset].count("\n") + 1


def specs_for_cases(case_ids: set[str], operators: set[str] | None = None) -> list[MutationSpec]:
    specs = [spec for spec in MUTATION_SPECS if spec.case_id in case_ids]
    if operators:
        specs = [spec for spec in specs if spec.operator in operators]
    return specs


def cases_with_specs(case_ids: list[str]) -> tuple[list[str], list[str]]:
    spec_case_ids = {spec.case_id for spec in MUTATION_SPECS}
    active = [case_id for case_id in case_ids if case_id in spec_case_ids]
    skipped = [case_id for case_id in case_ids if case_id not in spec_case_ids]
    return active, skipped


def make_sample_specs(
    specs: list[MutationSpec],
    max_samples: int,
    sample_offset: int = 0,
    sample_stride: int = 1,
) -> list[tuple[int, MutationSpec]]:
    if not specs:
        return []
    if sample_stride < 1:
        raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")
    return [
        (
            sample_offset + idx * sample_stride,
            specs[(sample_offset + idx * sample_stride) % len(specs)],
        )
        for idx in range(max_samples)
    ]


def restore_files(workspace: Path, originals: dict[str, str]) -> None:
    for relpath, text in originals.items():
        (workspace / relpath).write_text(text)


def apply_mutation(workspace: Path, spec: MutationSpec) -> dict[str, Any]:
    path = workspace / spec.file_path
    source = path.read_text()
    focus_line = find_focus_line(source, spec.clean_text)
    if focus_line is None:
        raise ValueError(f"clean_text not found for {spec.mutation_id} in {spec.file_path}")
    mutated = source.replace(spec.clean_text, spec.buggy_text, 1)
    path.write_text(mutated)
    return {
        "focus_line": focus_line,
        "clean_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "buggy_sha256": hashlib.sha256(mutated.encode()).hexdigest(),
        "buggy_context": source_context(mutated, focus_line),
    }


def verification_env(workspace: Path, pdk_root: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workspace / "src")
    env["GLAYOUT_SMGR"] = "1"
    if pdk_root is not None:
        env["PDK_ROOT"] = str(pdk_root)
        env["PDKPATH"] = str(pdk_root)
    for name in (
        "GLAYOUT_SMGR_CAPTURE_POLYGONS",
        "GLAYOUT_SMGR_CAPTURE_PORT_OBJECTS",
        "GLAYOUT_SMGR_CAPTURE_LIVE_REFS",
    ):
        env.pop(name, None)
    return env


def run_locator(
    workspace: Path,
    case_id: str,
    output_dir: Path,
    top_k: int,
    timeout: int,
    pdk_root: Path | None,
    traced_only: bool = False,
) -> dict[str, Any]:
    log_path = output_dir / "verification.log"
    cmd = [
        sys.executable,
        "scripts/run_smgr_verification_locator.py",
        "--case",
        case_id,
        "--output-dir",
        str(output_dir),
        "--top-k",
        str(top_k),
    ]
    if traced_only:
        cmd.append("--traced-only")
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=workspace,
        env=verification_env(workspace, pdk_root=pdk_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.time() - started
    log_path.write_text(completed.stdout)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "elapsed_sec": round(elapsed, 3),
        "log_path": log_path,
        "case_dir": output_dir / case_id,
        "case_result_path": output_dir / case_id / "case_result.json",
        "locator_path": output_dir / case_id / "verification_locator.json",
        "repair_packet_path": output_dir / case_id / "repair_packet.json",
    }


def iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def iter_ranked_dicts(value: Any, top_k: int) -> Iterable[tuple[int | None, dict[str, Any]]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, list) and key in {
                "candidate_calls",
                "top_candidate_calls",
                "source_spans",
                "source_label_candidates",
                "source_netlist_candidates",
                "source_physical_candidates",
                "ranked_source_spans",
                "candidates",
                "suspects",
            }:
                for idx, item in enumerate(child[:top_k], start=1):
                    if isinstance(item, dict):
                        yield idx, item
            yield from iter_ranked_dicts(child, top_k)
    elif isinstance(value, list):
        for child in value:
            yield from iter_ranked_dicts(child, top_k)


def dict_file(dct: dict[str, Any]) -> str | None:
    for key in ("file", "path", "source_file", "filepath", "relative_path"):
        value = dct.get(key)
        if isinstance(value, str):
            return normalize_repo_path(value)
    location = dct.get("location")
    if isinstance(location, dict):
        return dict_file(location)
    return None


def dict_line(dct: dict[str, Any]) -> int | None:
    for key in ("line", "focus_line", "start_line", "lineno", "source_line"):
        value = dct.get(key)
        if isinstance(value, int):
            return value
    location = dct.get("location")
    if isinstance(location, dict):
        return dict_line(location)
    return None


def dict_text(dct: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("source", "source_text", "snippet", "excerpt", "code", "text", "body"):
        value = dct.get(key)
        if isinstance(value, str):
            pieces.append(value)
    return "\n".join(pieces)


def compute_localizer_hit(
    locator: Any,
    repair_packet: Any,
    spec: MutationSpec,
    focus_line: int,
    top_k: int,
    line_window: int,
) -> dict[str, Any]:
    target_file = normalize_repo_path(spec.file_path)
    combined = {"locator": locator, "repair_packet": repair_packet}
    source_span_file_hit = False
    source_span_text_hit = False
    candidate_location_hit = False
    ranked_hits: list[dict[str, Any]] = []
    needle_fragments = [
        line.strip()
        for text in (spec.clean_text, spec.buggy_text)
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("# MUTATION")
    ]

    for dct in iter_dicts(combined):
        found_file = dict_file(dct)
        if found_file == target_file:
            source_span_file_hit = True
            text = dict_text(dct)
            if text and any(fragment in text for fragment in needle_fragments):
                source_span_text_hit = True

    for rank, dct in iter_ranked_dicts(combined, top_k):
        found_file = dict_file(dct)
        line = dict_line(dct)
        if found_file != target_file:
            continue
        text_hit = any(fragment in dict_text(dct) for fragment in needle_fragments)
        line_hit = line is not None and abs(line - focus_line) <= line_window
        if text_hit or line_hit:
            candidate_location_hit = True
            ranked_hits.append(
                {
                    "rank": rank,
                    "file": found_file,
                    "line": line,
                    "text_hit": text_hit,
                    "line_hit": line_hit,
                }
            )

    return {
        "top_k": top_k,
        "line_window": line_window,
        "target_file": target_file,
        "target_line": focus_line,
        "source_span_file_hit": source_span_file_hit,
        "source_span_text_hit": source_span_text_hit,
        "candidate_location_hit": candidate_location_hit,
        "hit": source_span_text_hit or candidate_location_hit,
        "ranked_hits": ranked_hits[:top_k],
    }


def summarize_case_result(case_result: Any) -> dict[str, Any]:
    if not isinstance(case_result, dict):
        return {"available": False}
    def status(section: str) -> dict[str, Any]:
        value = case_result.get(section)
        if not isinstance(value, dict):
            return {}
        return {
            "is_clean": value.get("is_clean"),
            "status": value.get("status"),
            "matched": value.get("matched"),
            "error_count": value.get("error_count"),
            "mismatch_markers": value.get("mismatch_markers"),
        }

    return {
        "available": True,
        "case_id": case_result.get("case_id"),
        "baseline_drc": status("baseline_drc"),
        "traced_drc": status("traced_drc"),
        "baseline_lvs": status("baseline_lvs"),
        "traced_lvs": status("traced_lvs"),
    }


def verification_artifacts_available(verifier: dict[str, Any], case_result: Any) -> bool:
    return verifier.get("returncode") == 0 and isinstance(case_result, dict)


def clean_case_passed(record: dict[str, Any], traced_only: bool = False) -> bool:
    if record.get("returncode") != 0:
        return False
    case_result = record.get("case_result")
    if not isinstance(case_result, dict) or not case_result.get("available"):
        return False
    required_sections = ("traced_drc", "traced_lvs") if traced_only else (
        "baseline_drc",
        "traced_drc",
        "baseline_lvs",
        "traced_lvs",
    )
    for key in required_sections:
        status = case_result.get(key) or {}
        if status.get("is_clean") is not True:
            return False
    return True


def clean_record_line(record: dict[str, Any], traced_only: bool = False) -> str:
    summary = record.get("case_result") or {}
    if not summary.get("available"):
        return (
            f"{record['case_id']}: FAIL no case_result returncode={record.get('returncode')} "
            f"log={record.get('log_path')}"
        )
    bits = []
    for key, label in (
        ("baseline_drc", "base_drc"),
        ("traced_drc", "traced_drc"),
        ("baseline_lvs", "base_lvs"),
        ("traced_lvs", "traced_lvs"),
    ):
        status = summary.get(key) or {}
        clean = status.get("is_clean")
        detail = status.get("status") or status.get("error_count") or status.get("mismatch_markers")
        bits.append(f"{label}={clean if clean is not None else '?'}({detail})")
    verdict = "PASS" if clean_case_passed(record, traced_only=traced_only) else "FAIL"
    return f"{record['case_id']}: {verdict} " + " ".join(bits)


def build_record(
    sample_id: str,
    replica_index: int,
    spec: MutationSpec,
    repo_commit: str | None,
    mutation_result: dict[str, Any],
    verifier: dict[str, Any],
    locator: Any,
    repair_packet: Any,
    top_k: int,
    line_window: int,
) -> dict[str, Any]:
    case_result = load_json(verifier["case_result_path"])
    hit = compute_localizer_hit(
        locator,
        repair_packet,
        spec,
        int(mutation_result["focus_line"]),
        top_k,
        line_window,
    )
    return {
        "sample_id": sample_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": repo_commit,
        "case_id": spec.case_id,
        "mutation": asdict(spec),
        "replica_index": replica_index,
        "target": {
            "file": spec.file_path,
            "focus_line": mutation_result["focus_line"],
            "buggy_sha256": mutation_result["buggy_sha256"],
            "clean_sha256": mutation_result["clean_sha256"],
            "buggy_context": mutation_result["buggy_context"],
        },
        "expected_repair_action": {
            "type": "replace_text",
            "file": spec.file_path,
            "find": spec.buggy_text,
            "replace": spec.clean_text,
        },
        "verification": {
            "returncode": verifier["returncode"],
            "elapsed_sec": verifier["elapsed_sec"],
            "case_result": summarize_case_result(case_result),
            "case_result_path": str(verifier["case_result_path"]),
            "locator_path": str(verifier["locator_path"]),
            "repair_packet_path": str(verifier["repair_packet_path"]),
            "log_path": str(verifier["log_path"]),
        },
        "localizer_hit": hit,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    hit_count = sum(1 for record in records if record.get("localizer_hit", {}).get("hit"))
    by_operator: dict[str, dict[str, int]] = {}
    by_case: dict[str, dict[str, int]] = {}
    for record in records:
        hit = bool(record.get("localizer_hit", {}).get("hit"))
        for bucket, key in (
            (by_operator, record["mutation"]["operator"]),
            (by_case, record["case_id"]),
        ):
            stats = bucket.setdefault(key, {"total": 0, "hit": 0})
            stats["total"] += 1
            stats["hit"] += int(hit)
    return {
        "total_samples": total,
        "localizer_topk_hits": hit_count,
        "localizer_topk_hit_rate": (hit_count / total) if total else None,
        "by_operator": by_operator,
        "by_case": by_case,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mutation-based SMGR repair benchmark data.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Global sample index offset for parallel shards. Example: offsets 0, 50, 100, 150 with --max-samples 50.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help=(
            "Step between global sample indices. Use --sample-offset SHARD --sample-stride NUM_SHARDS "
            "to round-robin specs across parallel workers instead of assigning contiguous cell-heavy ranges."
        ),
    )
    parser.add_argument(
        "--case-profile",
        choices=sorted(CASE_PROFILES),
        default="conservative",
        help="Named candidate case set. Explicit --cases overrides this.",
    )
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--operators", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--line-window", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--pdk-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-clean-validation", action="store_true")
    parser.add_argument("--allow-failed-clean-validation", action="store_true")
    parser.add_argument(
        "--drop-failed-clean-cases",
        action="store_true",
        help="Validate the requested/profile cases, then generate samples only for cases that are strict clean.",
    )
    parser.add_argument(
        "--fast-clean-validation",
        action="store_true",
        help=(
            "For clean validation, run only traced DRC/LVS and require traced sections to be clean. "
            "Use this when full baseline clean validation is unavailable in the current container."
        ),
    )
    parser.add_argument("--include-invalid-verification", action="store_true")
    parser.add_argument(
        "--fast-sample-verification",
        action="store_true",
        help="For mutated samples, skip baseline generation/DRC/LVS and run only traced DRC/LVS plus locator.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    workspace = output_dir / "workspace"
    samples_dir = output_dir / "samples"
    dataset_path = output_dir / "dataset.jsonl"
    records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    requested_cases = args.cases if args.cases is not None else CASE_PROFILES[args.case_profile]
    active_cases, cases_without_specs = cases_with_specs(list(requested_cases))
    specs = specs_for_cases(set(active_cases), set(args.operators) if args.operators else None)
    sample_specs = make_sample_specs(
        specs,
        args.max_samples,
        sample_offset=args.sample_offset,
        sample_stride=args.sample_stride,
    )
    plan = {
        "repo_root": str(repo_root),
        "workspace": str(workspace),
        "case_profile": args.case_profile,
        "requested_cases": requested_cases,
        "cases": active_cases,
        "cases_without_specs": cases_without_specs,
        "operators": args.operators,
        "pdk_root": str(args.pdk_root.resolve()) if args.pdk_root else os.environ.get("PDK_ROOT"),
        "available_specs": len(specs),
        "max_samples": args.max_samples,
        "sample_offset": args.sample_offset,
        "sample_stride": args.sample_stride,
        "fast_clean_validation": args.fast_clean_validation,
        "fast_sample_verification": args.fast_sample_verification,
        "planned_samples": [
            {"sample_index": idx, **asdict(spec)}
            for idx, spec in sample_specs
        ],
    }

    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "plan.json", plan)
    if args.dry_run:
        print(f"[repair-bench] dry-run plan written to {output_dir / 'plan.json'}", flush=True)
        if cases_without_specs:
            print(f"[repair-bench] cases without mutation specs skipped: {', '.join(cases_without_specs)}", flush=True)
        return 0

    copy_repo(repo_root, workspace, force=args.force)
    repo_commit = repo_sha(repo_root)

    if not args.skip_clean_validation:
        clean_records = []
        for case_index, case_id in enumerate(active_cases, start=1):
            print(f"[repair-bench] clean validation {case_index}/{len(active_cases)} {case_id}", flush=True)
            clean_dir = output_dir / "clean_validation" / case_id
            verifier = run_locator(
                workspace,
                case_id,
                clean_dir,
                args.top_k,
                args.timeout,
                args.pdk_root.resolve() if args.pdk_root else None,
                traced_only=args.fast_clean_validation,
            )
            case_result = load_json(verifier["case_result_path"])
            clean_records.append(
                {
                    "case_id": case_id,
                    "returncode": verifier["returncode"],
                    "elapsed_sec": verifier["elapsed_sec"],
                    "case_result": summarize_case_result(case_result),
                    "clean_validation_mode": "traced_only" if args.fast_clean_validation else "full",
                    "log_path": str(verifier["log_path"]),
                }
            )
            print(
                f"[repair-bench] clean {clean_record_line(clean_records[-1], traced_only=args.fast_clean_validation)}",
                flush=True,
            )
        write_json(output_dir / "clean_validation.json", clean_records)
        failed_clean = [
            record
            for record in clean_records
            if not clean_case_passed(record, traced_only=args.fast_clean_validation)
        ]
        if failed_clean and args.drop_failed_clean_cases:
            failed_ids = {record["case_id"] for record in failed_clean}
            active_cases = [case_id for case_id in active_cases if case_id not in failed_ids]
            specs = specs_for_cases(set(active_cases), set(args.operators) if args.operators else None)
            sample_specs = make_sample_specs(
                specs,
                args.max_samples,
                sample_offset=args.sample_offset,
                sample_stride=args.sample_stride,
            )
            print(
                "[repair-bench] dropping failed clean cases: "
                + ", ".join(record["case_id"] for record in failed_clean),
                flush=True,
            )
        elif failed_clean and not args.allow_failed_clean_validation:
            logs = "\n".join(f"  - {record['case_id']}: {record['log_path']}" for record in failed_clean)
            raise RuntimeError(
                "Clean validation failed before mutation generation. "
                "Fix the PDK/environment or pass --allow-failed-clean-validation for debugging only.\n"
                f"Failed clean cases:\n{logs}"
            )
    if not specs:
        raise RuntimeError("No mutation specs remain after case selection/clean filtering.")

    touched_files = sorted({spec.file_path for spec in specs})
    originals = {relpath: (workspace / relpath).read_text() for relpath in touched_files}

    write_json(
        output_dir / "active_plan.json",
        {
            **plan,
            "active_cases": active_cases,
            "active_specs": len(specs),
            "planned_samples": [
                {"sample_index": idx, **asdict(spec)}
                for idx, spec in sample_specs
            ],
        },
    )

    if dataset_path.exists():
        dataset_path.unlink()
    dataset_path.touch()

    try:
        for local_index, (sample_index, spec) in enumerate(sample_specs, start=1):
            sample_id = f"{sample_index:04d}_{spec.case_id}_{spec.mutation_id}"
            sample_dir = samples_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            restore_files(workspace, originals)
            print(f"[repair-bench] sample {local_index}/{len(sample_specs)} {sample_id}", flush=True)
            try:
                mutation_result = apply_mutation(workspace, spec)
                mutated_source = (workspace / spec.file_path).read_text()
                (sample_dir / Path(spec.file_path).name).write_text(mutated_source)
                verifier = run_locator(
                    workspace,
                    spec.case_id,
                    sample_dir / "verification",
                    args.top_k,
                    args.timeout,
                    args.pdk_root.resolve() if args.pdk_root else None,
                    traced_only=args.fast_sample_verification,
                )
                locator = load_json(verifier["locator_path"])
                repair_packet = load_json(verifier["repair_packet_path"])
                case_result = load_json(verifier["case_result_path"])
                if not verification_artifacts_available(verifier, case_result):
                    invalid = {
                        "sample_id": sample_id,
                        "case_id": spec.case_id,
                        "mutation_id": spec.mutation_id,
                        "returncode": verifier["returncode"],
                        "case_result_path": str(verifier["case_result_path"]),
                        "log_path": str(verifier["log_path"]),
                    }
                    invalid_records.append(invalid)
                    write_json(sample_dir / "invalid_verification.json", invalid)
                    if not args.include_invalid_verification:
                        raise RuntimeError(
                            f"Verification failed or did not produce case_result.json for {sample_id}; "
                            f"see {verifier['log_path']}"
                        )
                record = build_record(
                    sample_id=sample_id,
                    replica_index=sample_index // max(len(specs), 1),
                    spec=spec,
                    repo_commit=repo_commit,
                    mutation_result=mutation_result,
                    verifier=verifier,
                    locator=locator,
                    repair_packet=repair_packet,
                    top_k=args.top_k,
                    line_window=args.line_window,
                )
                write_json(sample_dir / "sample.json", record)
                append_jsonl(dataset_path, record)
                records.append(record)
            except Exception as exc:
                error = {
                    "sample_id": sample_id,
                    "case_id": spec.case_id,
                    "mutation_id": spec.mutation_id,
                    "error": repr(exc),
                }
                write_json(sample_dir / "error.json", error)
                if not args.continue_on_error:
                    raise
    finally:
        restore_files(workspace, originals)

    summary = {
        "repo_root": str(repo_root),
        "workspace": str(workspace),
        "dataset_path": str(dataset_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": repo_commit,
        "invalid_samples": len(invalid_records),
        "invalid_records_path": str(output_dir / "invalid_records.json"),
        **aggregate(records),
    }
    write_json(output_dir / "invalid_records.json", invalid_records)
    write_json(output_dir / "summary.json", summary)
    print(f"[repair-bench] wrote {dataset_path}", flush=True)
    print(f"[repair-bench] summary written to {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
