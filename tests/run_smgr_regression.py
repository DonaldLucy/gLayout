from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from smgr_cases import SMGR_CASES, get_case


def _candidate_pdk_roots() -> list[Path]:
    candidates: list[Path] = []
    raw_values = [
        os.environ.get("PDK_ROOT"),
        os.environ.get("PDKPATH"),
        os.environ.get("MAGIC_PDK_ROOT"),
        os.environ.get("NETGEN_PDK_ROOT"),
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        raw_values.append(str(Path(conda_prefix) / "share" / "pdk"))
    raw_values.extend(
        [
            "/foss/pdks",
            "/usr/bin/miniconda3/share/pdk",
            "/headless/conda-env/miniconda3/share/pdk",
        ]
    )

    seen: set[Path] = set()
    for raw in raw_values:
        if not raw or raw == "None":
            continue
        path = Path(raw).resolve()
        # Accept either the PDK root or the sky130A subdirectory itself.
        variants = [path]
        if path.name == "sky130A":
            variants.append(path.parent)
        else:
            variants.append(path / "sky130A")
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                candidates.append(variant)
    return candidates


def _resolve_pdk_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    lvs_ref = repo_root / "src" / "glayout" / "pdk" / "sky130_mapped" / "sky130_fd_sc_hd.spice"

    for candidate in _candidate_pdk_roots():
        if candidate.name == "sky130A":
            pdk_root = candidate.parent
            sky130_dir = candidate
        else:
            pdk_root = candidate
            sky130_dir = candidate / "sky130A"
        magicrc = sky130_dir / "libs.tech" / "magic" / "sky130A.magicrc"
        lvs_setup = sky130_dir / "libs.tech" / "netgen" / "sky130A_setup.tcl"
        if magicrc.exists() and lvs_setup.exists() and lvs_ref.exists():
            os.environ["PDK_ROOT"] = str(pdk_root)
            os.environ["PDKPATH"] = str(sky130_dir)
            os.environ["PDK"] = "sky130A"
            os.environ["MAGIC_PDK_ROOT"] = str(pdk_root)
            os.environ["NETGEN_PDK_ROOT"] = str(pdk_root)
            return {
                "pdk_root": pdk_root,
                "sky130_dir": sky130_dir,
                "magicrc": magicrc,
                "lvs_setup": lvs_setup,
                "lvs_ref": lvs_ref,
            }

    raise FileNotFoundError(
        "Could not locate a usable SKY130 PDK installation. Checked candidates:\n"
        + "\n".join(str(path) for path in _candidate_pdk_roots())
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonicalize_polygon(points: Any) -> tuple[tuple[float, float], ...]:
    pts = [(round(float(x), 6), round(float(y), 6)) for x, y in points]
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    if not pts:
        return tuple()

    def _rotations(seq: list[tuple[float, float]]) -> list[tuple[tuple[float, float], ...]]:
        return [tuple(seq[i:] + seq[:i]) for i in range(len(seq))]

    forward = min(_rotations(pts))
    backward = min(_rotations(list(reversed(pts))))
    return min(forward, backward)


def _gds_semantic_signature(path: Path) -> str:
    import gdsfactory as gf

    component = gf.import_gds(path)
    component = component.flatten()

    records: list[tuple[Any, ...]] = []
    for spec, polygons in component.get_polygons(by_spec=True).items():
        try:
            layer_spec = tuple(spec)
        except TypeError:
            layer_spec = (spec,)
        for polygon in polygons:
            records.append(("polygon", layer_spec, _canonicalize_polygon(polygon)))

    for label in getattr(component, "labels", []):
        origin = getattr(label, "origin", getattr(label, "position", (0.0, 0.0)))
        records.append(
            (
                "label",
                tuple(label.layer) if isinstance(label.layer, (tuple, list)) else label.layer,
                round(float(origin[0]), 6),
                round(float(origin[1]), 6),
                label.text,
            )
        )

    payload = json.dumps(sorted(records), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    rule_counts: dict[str, int] = {}
    current_rule: str | None = None
    total_errors = 0

    for line in lines:
        if line == "----------------------------------------":
            continue
        if line.startswith("Error while reading cell"):
            continue
        if line and line[0].isalpha():
            current_rule = line
            rule_counts.setdefault(current_rule, 0)
            continue
        if line.endswith("um") and current_rule:
            rule_counts[current_rule] += 1
            total_errors += 1

    count_match = re.search(r"count:\s*(\d+)\s*$", content, re.IGNORECASE | re.MULTILINE)
    count_value = int(count_match.group(1)) if count_match else None
    clean = total_errors == 0 and (
        (count_value == 0)
        or ("No errors found." in content)
        or ("count:" in content)
    )

    return {
        "tool": "magic",
        "report": str(report_path),
        "error_count": total_errors if count_value is None else max(total_errors, count_value),
        "rule_counts": rule_counts,
        "is_clean": clean,
    }


def _strict_lvs_report(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise AssertionError(f"Netgen LVS report missing: {report_path}")
    content = report_path.read_text()
    matched = (
        "Final result: Circuits match uniquely." in content
        or "Final result:\nCircuits match uniquely." in content
        or "Circuits match uniquely" in content
    )
    property_error = "Property errors were found." in content
    topology_mismatch = (
        "Top level cell failed pin matching." in content
        or "Netlists do not match." in content
        or ("Mismatch" in content and not matched)
    )
    mismatch_count = len(re.findall(r"no matching (?:net|instance)", content, re.IGNORECASE))
    status = (
        "property_error" if matched and property_error
        else "clean" if matched
        else "topology_mismatch" if topology_mismatch
        else "fail"
    )
    return {
        "tool": "netgen",
        "report": str(report_path),
        "matched": matched,
        "mismatch_markers": mismatch_count,
        "status": status,
        "property_error": property_error,
        "is_clean": matched and not property_error,
    }


def _run_drc(component: Any, design_name: str, case_dir: Path) -> dict[str, Any]:
    paths = _resolve_pdk_paths()
    from glayout import sky130

    if shutil.which("magic") is None:
        raise RuntimeError("magic is not available in PATH")
    output_dir = case_dir / "magic_drc"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    print(f"[SMGR] Using PDK_ROOT={paths['pdk_root']}")
    print(f"[SMGR] Using MAGICRC={paths['magicrc']}")
    sky130.drc_magic(
        component,
        design_name,
        pdk_root=paths["pdk_root"],
        magic_drc_file=paths["magicrc"],
        output_file=output_dir,
    )
    report_path = output_dir / "drc" / design_name / f"{design_name}.rpt"
    return _strict_magic_report(report_path)


def _run_lvs(component: Any, design_name: str, case_dir: Path) -> dict[str, Any]:
    paths = _resolve_pdk_paths()
    from glayout import sky130

    if shutil.which("magic") is None or shutil.which("netgen") is None:
        raise RuntimeError("magic/netgen are not available in PATH")
    output_dir = case_dir / "netgen_lvs"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    print(f"[SMGR] Using PDK_ROOT={paths['pdk_root']}")
    print(f"[SMGR] Using LVS setup={paths['lvs_setup']}")
    sky130.lvs_netgen(
        layout=component,
        design_name=design_name,
        pdk_root=paths["pdk_root"],
        magic_drc_file=paths["magicrc"],
        lvs_setup_tcl_file=paths["lvs_setup"],
        lvs_schematic_ref_file=paths["lvs_ref"],
        output_file_path=output_dir,
        copy_intermediate_files=True,
    )
    report_path = output_dir / "lvs" / design_name / f"{design_name}_lvs.rpt"
    return _strict_lvs_report(report_path)


def _build_component(case_id: str, traced: bool) -> Any:
    _resolve_pdk_paths()
    from glayout import disable_source_mapping, enable_source_mapping, reset_source_mapping
    from gdsfactory.component import Component

    _clear_cache()
    reset_source_mapping()
    if traced:
        enable_source_mapping(reset=True, auto_emit_sidecar=True)
    else:
        disable_source_mapping()
    component = get_case(case_id).builder()
    if isinstance(component, tuple):
        component = component[0]
    if hasattr(component, "write_gds"):
        return component
    # Some legacy helpers return a ComponentReference instead of a top-level
    # Component. Wrap those in a lightweight container so regression can still
    # emit GDS and provenance without changing the helper contract.
    if hasattr(component, "parent") and hasattr(component, "ports"):
        wrapper = Component(f"{case_id}_wrapped")
        ref = wrapper.add_ref(component.parent)
        try:
            ref.move(component.center)
        except Exception:
            pass
        try:
            wrapper.add_ports(ref.get_ports_list())
        except Exception:
            pass
        info = getattr(component, "info", None)
        if info:
            wrapper.info.update(info)
        return wrapper
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
    candidate_ids = [entry["call_id"] for entry in candidates[:10]]
    if sample_call_id not in candidate_ids:
        sample_call = snapshot.get_call(sample_call_id) or {}
        parent_id = sample_call.get("parent_call_id")
        if parent_id not in candidate_ids:
            # Fall back to the root component object for deeply hierarchical
            # designs where the first sampled object is too low-level to rank
            # its exact originating helper call near the top.
            root_object = None
            for obj in snapshot.objects.values():
                if obj.get("generated_by", {}).get("call_id") == root_call_id and obj.get("bbox"):
                    root_object = obj
                    break
            if root_object is None:
                raise AssertionError(
                    f"Sample bbox query for {case_id} did not return its originating call in the top candidates"
                )
            root_candidates = snapshot.rank_candidate_calls(root_object["bbox"])
            root_candidate_ids = [entry["call_id"] for entry in root_candidates[:10]]
            if root_call_id not in root_candidate_ids:
                raise AssertionError(
                    f"Sample bbox query for {case_id} did not return either the sampled call or the root call in top candidates"
                )
            candidate_ids = root_candidate_ids
            sample_call_id = root_call_id
            sample_object = root_object

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
    baseline_semantic_hash = _gds_semantic_signature(baseline_gds)
    traced_semantic_hash = _gds_semantic_signature(traced_gds)
    if baseline_semantic_hash != traced_semantic_hash:
        raise AssertionError(
            f"GDS geometry changed after enabling SMGR for {case_id}: {baseline_semantic_hash} != {traced_semantic_hash}"
        )

    result = {
        "case_id": case_id,
        "description": get_case(case_id).description,
        "baseline_gds": str(baseline_gds),
        "traced_gds": str(traced_gds),
        "sidecar": str(sidecar_path),
        "baseline_gds_sha256": baseline_hash,
        "traced_gds_sha256": traced_hash,
        "gds_semantic_sha256": baseline_semantic_hash,
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
    parser.add_argument("--resume", action="store_true", help="Skip cases that already have a case_result.json in the output directory.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running remaining cases after a case fails.")
    args = parser.parse_args()

    selected = args.cases or [case.case_id for case in SMGR_CASES]
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case_id in selected:
        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_result_path = case_dir / "case_result.json"
        if args.resume and case_result_path.exists():
            result = json.loads(case_result_path.read_text())
            summary.append(result)
            print(f"[SKIP] {case_id} (resume)")
            continue
        try:
            result = run_case(case_id, output_root, run_drc=not args.skip_drc, run_lvs=not args.skip_lvs)
        except Exception as exc:
            failure = {"case_id": case_id, "error": str(exc)}
            failures.append(failure)
            (case_dir / "case_error.txt").write_text(str(exc))
            print(f"[FAIL] {case_id}: {exc}")
            if not args.continue_on_error:
                raise
            continue
        summary.append(result)
        case_result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"[PASS] {case_id}")

    summary_path = output_root / "summary.json"
    payload: dict[str, Any] = {"cases": summary}
    if failures:
        payload["failures"] = failures
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Summary written to {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
