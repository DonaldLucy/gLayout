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


def _pick_port(component, names: list[str], purpose: str):
    for name in names:
        if name in component.ports:
            return component.ports[name], name
    available = "\n".join(sorted(component.ports.keys())[:300])
    raise KeyError(
        f"Could not find a port for {purpose}. Tried {names!r}. "
        f"First available ports:\n{available}"
    )


def _copy_port(port):
    try:
        return port.copy()
    except AttributeError:
        return port


def _candidate_ports(component, names: list[str]):
    return [(name, component.ports[name]) for name in names if name in component.ports]


def _build_route_between(pdk, edge1, edge2, purpose: str):
    from glayout.routing import L_route, c_route, straight_route
    from glayout.routing.smart_route import smart_route

    errors: list[str] = []

    def fresh_edges():
        return _copy_port(edge1), _copy_port(edge2)

    attempts = [("smart_route", lambda: smart_route(pdk, *fresh_edges()))]
    orientation1 = round(edge1.orientation) % 360
    orientation2 = round(edge2.orientation) % 360
    if orientation1 == orientation2:
        attempts.append(
            (
                "c_route",
                lambda: c_route(
                    pdk,
                    *fresh_edges(),
                    extension=3 * pdk.util_max_metal_seperation(),
                    viaoffset=False,
                ),
            )
        )
    if orientation1 % 180 != orientation2 % 180:
        attempts.append(("L_route", lambda: L_route(pdk, *fresh_edges(), viaoffset=False)))
    attempts.append(("straight_route", lambda: straight_route(pdk, *fresh_edges())))

    for method, builder in attempts:
        try:
            return builder(), method
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    raise ValueError(f"Could not route {purpose}: " + " | ".join(errors))


def _add_first_successful_route(component, pdk, start_names: list[str], end_names: list[str], purpose: str, prefix: str):
    errors: list[str] = []
    for start_name, start_port in _candidate_ports(component, start_names):
        for end_name, end_port in _candidate_ports(component, end_names):
            try:
                route, method = _build_route_between(pdk, start_port, end_port, purpose)
                route_ref = component << route
                component.add_ports(route_ref.get_ports_list(), prefix=prefix)
                return {
                    "start_port": start_name,
                    "end_port": end_name,
                    "method": method,
                }
            except Exception as exc:
                errors.append(f"{start_name}->{end_name}: {exc}")
    raise ValueError(f"Could not add {purpose}. Tried routes: " + " | ".join(errors[:12]))


def add_diff_pair_ibias_candidate_routes(component, pdk):
    """Patch the topology called out by the locator packet before labeling.

    The schematic netlist connects DIFF_PAIR.VTAIL to CMIRROR.VOUT.  The
    original layout exposes both sides as ports but does not physically route
    them together, so this candidate adds that missing tail-current route.
    It also ties the current-mirror source rail to the local pwell tie when
    those ports are present, matching the schematic B->VSS mapping.
    """

    component.unlock()
    tail_route_info = _add_first_successful_route(
        component,
        pdk,
        [
            "source_routeE_con_S",
            "source_routeW_con_S",
            "source_routeE_con_N",
            "source_routeW_con_N",
            "bl_multiplier_0_source_S",
            "br_multiplier_0_source_S",
            "tl_multiplier_0_source_S",
            "tr_multiplier_0_source_S",
        ],
        [
            "ibias_B_drain_N",
            "ibias_B_drain_E",
            "ibias_B_drain_W",
            "ibias_B_drain_S",
        ],
        "diff-pair VTAIL to current-mirror VOUT",
        "repair_tail_",
    )

    bulk_route_info = _add_first_successful_route(
        component,
        pdk,
        [
            "ibias_purposegndport",
            "ibias_purposegndportscon_S",
            "ibias_purposegndportscon_N",
            "ibias_A_source_E",
            "ibias_B_source_E",
            "ibias_A_source_W",
            "ibias_B_source_W",
        ],
        [
            "ibias_welltie_S_top_met_N",
            "ibias_welltie_S_top_met_S",
            "ibias_welltie_W_top_met_W",
            "ibias_welltie_E_top_met_E",
            "ibias_welltie_N_top_met_N",
        ],
        "current-mirror source rail to local pwell tie",
        "repair_bulk_",
    )
    component.info["candidate_repair_routes"] = {
        "tail": tail_route_info,
        "bulk": bulk_route_info,
    }
    return component


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
    component = add_diff_pair_ibias_candidate_routes(component, pdk)
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
        locator = locate_case_result(case_result_path, top_k=args.top_k, include_repair_packet=True)
        repair_packet = locator.pop("repair_packet", None)
        if repair_packet is not None:
            repair_packet_path = case_dir / "repair_packet.json"
            repair_packet_path.write_text(json.dumps(repair_packet, indent=2, sort_keys=True))
            locator["repair_packet_path"] = str(repair_packet_path)
            print(f"[CANDIDATE] wrote {repair_packet_path}")
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
