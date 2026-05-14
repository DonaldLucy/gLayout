#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TESTS_DIR))


def _load_regression_runner() -> Any:
    path = TESTS_DIR / "run_smgr_regression.py"
    spec = importlib.util.spec_from_file_location("smgr_regression_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _print_locator_summary(locator: dict[str, Any], top_k: int) -> None:
    case_id = locator["case_id"]
    print(f"[LOCATOR] case={case_id}")
    drc = locator.get("drc") or {}
    lvs = locator.get("lvs") or {}
    print(f"[LOCATOR] DRC issues parsed: {drc.get('issue_count', 0)}")
    print(f"[LOCATOR] LVS issues parsed: {lvs.get('issue_count', 0)} status={lvs.get('status')}")

    for group_name in ("drc", "lvs"):
        group = locator.get(group_name) or {}
        issues = group.get("issues", [])
        if not issues:
            continue
        print(f"\n[{group_name.upper()}] first issues and candidate source locations")
        for issue_index, issue in enumerate(issues[:3], start=1):
            print(f"  issue {issue_index}: {issue.get('kind')} {issue.get('rule') or issue.get('raw')}")
            for candidate in issue.get("candidate_calls", [])[:top_k]:
                definition = candidate.get("definition") or {}
                callsite = candidate.get("callsite") or {}
                location = definition.get("file") or callsite.get("file")
                line = definition.get("line") or callsite.get("line")
                print(
                    "    "
                    f"{candidate.get('call_id')} score={candidate.get('score')} "
                    f"{candidate.get('generator_id')} {location}:{line}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one SMGR verification case and map DRC/LVS failures back to provenance calls."
    )
    parser.add_argument("--case", default="diff_pair_ibias", help="SMGR case id to run.")
    parser.add_argument(
        "--output-dir",
        default="build/smgr_locator_testbench",
        help="Directory for GDS, verification reports, and locator output.",
    )
    parser.add_argument("--no-run", action="store_true", help="Only locate from an existing case_result.json.")
    parser.add_argument("--top-k", type=int, default=8, help="Candidate calls per issue.")
    parser.add_argument("--max-drc-issues", type=int, default=24)
    parser.add_argument("--max-lvs-issues", type=int, default=48)
    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    case_dir = output_root / args.case
    case_dir.mkdir(parents=True, exist_ok=True)
    case_result_path = case_dir / "case_result.json"

    if not args.no_run:
        runner = _load_regression_runner()
        result = runner.run_case(args.case, output_root, run_drc=True, run_lvs=True)
        case_result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        (output_root / "summary.json").write_text(json.dumps({"cases": [result]}, indent=2, sort_keys=True))
        print(f"[LOCATOR] Wrote case result to {case_result_path}")
    elif not case_result_path.is_file():
        raise FileNotFoundError(f"Missing existing case result: {case_result_path}")

    from glayout.verification.locator import locate_case_result

    locator = locate_case_result(
        case_result_path,
        top_k=args.top_k,
        max_drc_issues=args.max_drc_issues,
        max_lvs_issues=args.max_lvs_issues,
    )
    locator_path = case_dir / "verification_locator.json"
    locator_path.write_text(json.dumps(locator, indent=2, sort_keys=True))
    print(f"[LOCATOR] Wrote locator artifact to {locator_path}")
    _print_locator_summary(locator, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
