from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"
for path in (SRC_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_regression_runner() -> Any:
    path = TESTS_DIR / "run_smgr_regression.py"
    spec = importlib.util.spec_from_file_location("smgr_regression_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n")


def _status(result: dict[str, Any], section: str) -> dict[str, Any]:
    value = result.get(section)
    return value if isinstance(value, dict) else {}


def _is_clean(result: dict[str, Any], mode: str) -> bool:
    sections = (
        ("traced_drc", "traced_lvs")
        if mode == "traced"
        else ("baseline_drc", "traced_drc", "baseline_lvs", "traced_lvs")
    )
    return all(_status(result, section).get("is_clean") is True for section in sections)


def _summarize_result(case_id: str, mode: str, result: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "mode": mode,
        "ok": result is not None and error is None,
        "clean": False,
        "error": error or "",
    }
    if result is not None:
        row["clean"] = _is_clean(result, mode)
        for section in ("baseline_drc", "traced_drc", "baseline_lvs", "traced_lvs"):
            status = _status(result, section)
            row[f"{section}_clean"] = status.get("is_clean")
            row[f"{section}_status"] = status.get("status")
            row[f"{section}_matched"] = status.get("matched")
            row[f"{section}_error_count"] = status.get("error_count")
            row[f"{section}_mismatch_markers"] = status.get("mismatch_markers")
    return row


def _run_traced_case(runner: Any, case_id: str, output_root: Path) -> dict[str, Any]:
    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    design_name = runner._component_name(case_id)
    traced_component = runner._build_component(case_id, traced=True)
    traced_gds = case_dir / f"{design_name}.traced.gds"
    traced_component.write_gds(str(traced_gds))
    return {
        "case_id": case_id,
        "description": runner.get_case(case_id).description,
        "traced_gds": str(traced_gds),
        "sidecar": str(traced_gds.with_suffix(".provenance.json")),
        "traced_drc": runner._run_drc(traced_component, f"{design_name}_traced", case_dir),
        "traced_lvs": runner._run_lvs(traced_component, f"{design_name}_traced", case_dir),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "mode",
        "ok",
        "clean",
        "error",
        "baseline_drc_clean",
        "traced_drc_clean",
        "baseline_lvs_clean",
        "traced_lvs_clean",
        "baseline_drc_status",
        "traced_drc_status",
        "baseline_lvs_status",
        "traced_lvs_status",
        "baseline_drc_error_count",
        "traced_drc_error_count",
        "baseline_lvs_mismatch_markers",
        "traced_lvs_mismatch_markers",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover which SMGR cases are clean in the current EDA environment.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", default=None, help="Optional case ids. Defaults to all SMGR_CASES.")
    parser.add_argument(
        "--mode",
        choices=("traced", "full"),
        default="traced",
        help="Use traced-only DRC/LVS or full baseline+traced regression.",
    )
    parser.add_argument("--pdk-root", type=Path, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.force:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.pdk_root is not None:
        pdk_root = args.pdk_root.resolve()
        os.environ["PDK_ROOT"] = str(pdk_root)
        os.environ["PDKPATH"] = str(pdk_root / "sky130A")
        os.environ["MAGIC_PDK_ROOT"] = str(pdk_root)
        os.environ["NETGEN_PDK_ROOT"] = str(pdk_root)
        os.environ["PDK"] = "sky130A"

    runner = _load_regression_runner()
    case_ids = args.cases or [case.case_id for case in runner.SMGR_CASES]

    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids, start=1):
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[clean-census] {index}/{len(case_ids)} {case_id}", flush=True)
        result = None
        error = None
        try:
            if args.mode == "traced":
                result = _run_traced_case(runner, case_id, output_dir)
            else:
                result = runner.run_case(case_id, output_dir, run_drc=True, run_lvs=True)
            _write_json(case_dir / "case_result.json", result)
        except Exception:
            error = traceback.format_exc()
            (case_dir / "case_error.txt").write_text(error)
            print(error, file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise
        row = _summarize_result(case_id, args.mode, result, error)
        rows.append(row)
        print(
            f"[clean-census] {case_id} ok={row['ok']} clean={row['clean']}",
            flush=True,
        )

    clean_cases = [row["case_id"] for row in rows if row["clean"]]
    payload = {
        "mode": args.mode,
        "total": len(rows),
        "clean_count": len(clean_cases),
        "clean_cases": clean_cases,
        "rows": rows,
    }
    _write_json(output_dir / "clean_census.json", payload)
    _write_csv(output_dir / "clean_census.csv", rows)
    (output_dir / "clean_cases.txt").write_text("\n".join(clean_cases) + ("\n" if clean_cases else ""))
    print(f"[clean-census] clean {len(clean_cases)}/{len(rows)}", flush=True)
    print(f"[clean-census] wrote {output_dir / 'clean_census.json'}", flush=True)
    return 0 if clean_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
