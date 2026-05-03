from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from smgr_cases import SMGR_CASES, get_case


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clear_cache() -> None:
    try:
        from gdsfactory.cell import clear_cache

        clear_cache()
    except Exception:
        pass


def _component_name(case_id: str) -> str:
    return case_id.replace("-", "_")


def _strict_magic_report(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise AssertionError(f"Magic DRC report missing: {report_path}")
    content = report_path.read_text()
    match = re.search(r"count:\s*(\d+)", content)
    if not match:
        raise AssertionError(f"Magic DRC report did not contain an error count: {report_path}")
    count = int(match.group(1))
    return {
        "tool": "magic",
        "report": str(report_path),
        "error_count": count,
        "is_clean": count == 0,
    }


def _strict_lvs_report(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise AssertionError(f"Netgen LVS report missing: {report_path}")
    content = report_path.read_text()
    matched = "Netlists match" in content or "Circuits match uniquely" in content
    mismatched = "Netlists do not match" in content or "Netlist mismatch" in content
    if not matched and not mismatched:
        raise AssertionError(f"Netgen LVS report was inconclusive: {report_path}")
    mismatch_count = len(re.findall(r"no matching (?:net|instance)", content))
    return {
        "tool": "netgen",
        "report": str(report_path),
        "matched": matched,
        "mismatch_markers": mismatch_count,
        "is_clean": matched and mismatch_count == 0 and not mismatched,
    }


def _run_drc(component: Any, design_name: str, case_dir: Path) -> dict[str, Any]:
    from glayout import sky130

    if shutil.which("magic") is None:
        raise RuntimeError("magic is not available in PATH")
    output_dir = case_dir / "magic_drc"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    sky130.drc_magic(component, design_name, output_file=output_dir)
    report_path = output_dir / "drc" / design_name / f"{design_name}.rpt"
    return _strict_magic_report(report_path)


def _run_lvs(component: Any, design_name: str, case_dir: Path) -> dict[str, Any]:
    from glayout import sky130

    if shutil.which("magic") is None or shutil.which("netgen") is None:
        raise RuntimeError("magic/netgen are not available in PATH")
    output_dir = case_dir / "netgen_lvs"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    sky130.lvs_netgen(layout=component, design_name=design_name, output_file_path=output_dir)
    report_path = output_dir / "lvs" / design_name / f"{design_name}_lvs.rpt"
    return _strict_lvs_report(report_path)


def _build_component(case_id: str, traced: bool) -> Any:
    from glayout import disable_source_mapping, enable_source_mapping, reset_source_mapping

    _clear_cache()
    reset_source_mapping()
    if traced:
        enable_source_mapping(reset=True, auto_emit_sidecar=True)
    else:
        disable_source_mapping()
    component = get_case(case_id).builder()
    if isinstance(component, tuple):
        component = component[0]
    return component


def _validate_sidecar(case_id: str, sidecar_path: Path) -> dict[str, Any]:
    from glayout import load_provenance

    if not sidecar_path.is_file():
        raise AssertionError(f"Missing provenance sidecar for {case_id}: {sidecar_path}")
    snapshot = load_provenance(sidecar_path)
    if not snapshot.calls:
        raise AssertionError(f"No call records found in {sidecar_path}")
    if not snapshot.objects:
        raise AssertionError(f"No object records found in {sidecar_path}")
    if len(snapshot.objects) > len(snapshot.calls) * 500:
        raise AssertionError(
            f"Sidecar for {case_id} looks over-captured: {len(snapshot.objects)} objects across {len(snapshot.calls)} calls"
        )

    root_call_id = next(iter(snapshot.calls.keys()))
    root_call = snapshot.get_call(root_call_id)
    if root_call is None:
        raise AssertionError(f"Missing root call {root_call_id}")
    if root_call.get("output_bbox") is None:
        raise AssertionError(f"Root call for {case_id} did not record an output bbox")

    sample_object = None
    for obj in snapshot.objects.values():
        if obj.get("bbox") and obj.get("generated_by", {}).get("call_id"):
            sample_object = obj
            break
    if sample_object is None:
        raise AssertionError(f"Could not find a queryable object in {sidecar_path}")

    candidates = snapshot.rank_candidate_calls(sample_object["bbox"])
    if not candidates:
        raise AssertionError(f"No candidate calls found for sample bbox in {sidecar_path}")
    sample_call_id = sample_object["generated_by"]["call_id"]
    candidate_ids = [entry["call_id"] for entry in candidates[:3]]
    if sample_call_id not in candidate_ids:
        raise AssertionError(
            f"Sample bbox query for {case_id} did not return its originating call in the top candidates"
        )

    return {
        "root_call_id": root_call_id,
        "call_count": len(snapshot.calls),
        "object_count": len(snapshot.objects),
        "sidecar_bytes": sidecar_path.stat().st_size,
        "sample_object_id": sample_object["object_id"],
        "sample_call_id": sample_call_id,
        "top_candidate_call_ids": candidate_ids,
    }


def run_case(case_id: str, output_root: Path, run_drc: bool, run_lvs: bool) -> dict[str, Any]:
    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    design_name = _component_name(case_id)

    baseline_component = _build_component(case_id, traced=False)
    baseline_gds = case_dir / f"{design_name}.baseline.gds"
    baseline_component.write_gds(str(baseline_gds))

    traced_component = _build_component(case_id, traced=True)
    traced_gds = case_dir / f"{design_name}.traced.gds"
    traced_component.write_gds(str(traced_gds))
    sidecar_path = traced_gds.with_suffix(".provenance.json")

    baseline_hash = _sha256(baseline_gds)
    traced_hash = _sha256(traced_gds)
    if baseline_hash != traced_hash:
        raise AssertionError(
            f"GDS changed after enabling SMGR for {case_id}: {baseline_hash} != {traced_hash}"
        )

    result = {
        "case_id": case_id,
        "description": get_case(case_id).description,
        "baseline_gds": str(baseline_gds),
        "traced_gds": str(traced_gds),
        "sidecar": str(sidecar_path),
        "gds_sha256": baseline_hash,
        "sidecar_summary": _validate_sidecar(case_id, sidecar_path),
    }

    if run_drc:
        result["baseline_drc"] = _run_drc(baseline_component, f"{design_name}_baseline", case_dir)
        result["traced_drc"] = _run_drc(traced_component, f"{design_name}_traced", case_dir)
        if result["baseline_drc"]["is_clean"] != result["traced_drc"]["is_clean"]:
            raise AssertionError(f"DRC status changed after SMGR for {case_id}")

    if run_lvs:
        result["baseline_lvs"] = _run_lvs(baseline_component, f"{design_name}_baseline", case_dir)
        result["traced_lvs"] = _run_lvs(traced_component, f"{design_name}_traced", case_dir)
        if result["baseline_lvs"]["is_clean"] != result["traced_lvs"]["is_clean"]:
            raise AssertionError(f"LVS status changed after SMGR for {case_id}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SMGR regression cases against gLayout cells.")
    parser.add_argument("--output-dir", default="build/smgr_regression", help="Directory for GDS, reports, and summaries.")
    parser.add_argument("--cases", nargs="*", default=None, help="Optional subset of case ids to run.")
    parser.add_argument("--skip-drc", action="store_true", help="Skip Magic DRC.")
    parser.add_argument("--skip-lvs", action="store_true", help="Skip Netgen LVS.")
    args = parser.parse_args()

    selected = args.cases or [case.case_id for case in SMGR_CASES]
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for case_id in selected:
        result = run_case(case_id, output_root, run_drc=not args.skip_drc, run_lvs=not args.skip_lvs)
        summary.append(result)
        print(f"[PASS] {case_id}")

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
