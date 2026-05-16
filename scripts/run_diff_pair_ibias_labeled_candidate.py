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
    spec = importlib.util.spec_from_file_location("smgr_regression_runner_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_component(candidate: Any, name: str):
    from gdsfactory.component import Component

    if hasattr(candidate, "write_gds"):
        candidate.name = name
        return candidate
    if not (hasattr(candidate, "parent") and hasattr(candidate, "ports")):
        raise TypeError(f"Expected Component or ComponentReference, got {type(candidate)!r}")

    wrapper = Component(name)
    ref = wrapper.add_ref(candidate.parent)
    try:
        ref.move(candidate.center)
    except Exception:
        pass
    wrapper.add_ports(ref.get_ports_list())
    info = getattr(candidate, "info", None)
    if info:
        wrapper.info.update(info)
    return wrapper


def _add_label(component, pdk, text: str, port_name: str, glayer: str, size: float) -> None:
    from gdsfactory.components.rectangle import rectangle
    from glayout.util.comp_utils import align_comp_to_port

    if port_name not in component.ports:
        available = "\n".join(sorted(component.ports.keys())[:250])
        raise KeyError(
            f"Missing port {port_name!r} for label {text!r}. "
            f"First available ports:\n{available}"
        )
    pin = rectangle(layer=pdk.get_glayer(f"{glayer}_pin"), size=(size, size), centered=True).copy()
    pin.add_label(text=text, layer=pdk.get_glayer(f"{glayer}_label"))
    component.add(align_comp_to_port(pin, component.ports[port_name], alignment=("c", "b")))


def add_diff_pair_ibias_candidate_labels(component, pdk):
    """Add only the top-level LVS pins suggested by the locator trace."""

    component.unlock()
    label_ports = {
        "VP": ("br_multiplier_0_gate_S", "met2", 0.27),
        "VN": ("bl_multiplier_0_gate_S", "met2", 0.27),
        "VDD1": ("tl_multiplier_0_drain_N", "met2", 0.27),
        "VDD2": ("tr_multiplier_0_drain_N", "met2", 0.27),
        "B": ("tap_N_top_met_S", "met1", 0.50),
        "IBIAS": ("ibias_A_drain_E", "met3", 0.50),
        "VSS": ("ibias_purposegndport", "met2", 0.50),
    }
    for label, (port_name, glayer, size) in label_ports.items():
        _add_label(component, pdk, label, port_name, glayer, size)
    return component.flatten()


def build_candidate(pdk):
    from glayout.cells.composite.diffpair_cmirror_bias import diff_pair_ibias

    raw = diff_pair_ibias(
        pdk,
        half_diffpair_params=(6.0, 1.0, 4),
        diffpair_bias=(6.0, 2.0, 4),
        rmult=2,
        with_antenna_diode_on_diffinputs=0,
    )
    component = _as_component(raw, "diff_pair_ibias_candidate")
    netlist_obj = component.info.get("netlist")
    component = add_diff_pair_ibias_candidate_labels(component, pdk)
    if hasattr(netlist_obj, "generate_netlist"):
        component.info["netlist"] = netlist_obj.generate_netlist()
        component.info["netlist_obj"] = netlist_obj
        component.info["netlist_data"] = {
            "circuit_name": netlist_obj.circuit_name,
            "nodes": netlist_obj.nodes,
            "source_netlist": netlist_obj.source_netlist,
        }
    return component


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a labeled diff_pair_ibias LVS candidate and optionally run DRC/LVS plus locator."
    )
    parser.add_argument("--output-dir", default="build/diff_pair_ibias_labeled_candidate")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    runner = _load_regression_runner()
    runner._resolve_pdk_paths()

    from glayout import disable_source_mapping, enable_source_mapping, reset_source_mapping, sky130
    from glayout.verification.locator import locate_case_result

    output_root = Path(args.output_dir).resolve()
    case_id = "diff_pair_ibias_labeled_candidate"
    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    reset_source_mapping()
    enable_source_mapping(reset=True, auto_emit_sidecar=True)
    component = build_candidate(sky130)
    gds_path = case_dir / f"{case_id}.gds"
    component.write_gds(str(gds_path))
    sidecar_path = gds_path.with_suffix(".provenance.json")
    print(f"[CANDIDATE] wrote {gds_path}")
    print(f"[CANDIDATE] sidecar {sidecar_path}")

    result = {
        "case_id": case_id,
        "description": "Labeled candidate for diff_pair_ibias LVS debug",
        "traced_gds": str(gds_path),
        "sidecar": str(sidecar_path),
    }

    if not args.skip_verification:
        result["traced_drc"] = runner._run_drc(component, f"{case_id}_traced", case_dir)
        result["traced_lvs"] = runner._run_lvs(component, f"{case_id}_traced", case_dir)

    case_result_path = case_dir / "case_result.json"
    case_result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[CANDIDATE] wrote {case_result_path}")

    if not args.skip_verification:
        locator = locate_case_result(case_result_path, top_k=args.top_k)
        locator_path = case_dir / "verification_locator.json"
        locator_path.write_text(json.dumps(locator, indent=2, sort_keys=True))
        lvs = locator.get("lvs") or {}
        print(
            "[CANDIDATE] LVS "
            f"status={lvs.get('status')} issues={lvs.get('issue_count')} "
            f"locator={locator_path}"
        )

    disable_source_mapping()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
