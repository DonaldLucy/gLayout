from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_netlist_summary_is_compact_and_queryable():
    summary_mod = _load_module(
        "smgr_netlist_summary_test",
        "src/glayout/provenance/netlist_summary.py",
    )

    class ChildNetlist:
        circuit_name = "sky130_fd_pr__nfet_01v8"
        nodes = ["D", "G", "S", "B"]
        parameters = {"width": 2.0, "length": 0.5}
        source_netlist = ".subckt sky130_fd_pr__nfet_01v8 D G S B"

    class TopNetlist:
        circuit_name = "DEMO"
        nodes = ["VIN", "VOUT", "VSS"]
        sub_netlists = [ChildNetlist()]
        netlist_connections = [["VOUT", "VIN", "VSS", "VSS"]]
        parameters = {}
        source_netlist = ""

    summary = summary_mod.summarize_netlist(TopNetlist())
    assert summary["circuit_name"] == "DEMO"
    assert summary["instance_count"] == 1
    assert summary["instances"][0]["pin_connections"][1] == {"pin": "G", "net": "VIN"}
    assert any(row["net"] == "VSS" for row in summary["net_fanout"])


def test_spice_summary_recovers_instance_pin_map():
    summary_mod = _load_module(
        "smgr_netlist_summary_spice_test",
        "src/glayout/provenance/netlist_summary.py",
    )
    spice = """
.subckt CHILD D G S B
.ends CHILD
.subckt TOP IN OUT VSS
X0 OUT IN VSS VSS CHILD l=0.5 w=2
.ends TOP
"""
    summary = summary_mod.parse_spice_netlist_summary(spice, circuit_name="TOP")
    assert summary["circuit_name"] == "TOP"
    assert summary["instances"][0]["circuit_name"] == "CHILD"
    assert {"pin": "G", "net": "IN"} in summary["instances"][0]["pin_connections"]


def test_spice_summary_assigns_sky130_mos_pin_roles_without_subckt():
    summary_mod = _load_module(
        "smgr_netlist_summary_sky130_pin_test",
        "src/glayout/provenance/netlist_summary.py",
    )
    spice = """
.subckt TOP D G S B
X0 D G S B sky130_fd_pr__nfet_01v8
.ends TOP
"""
    summary = summary_mod.parse_spice_netlist_summary(spice, circuit_name="TOP")
    assert summary["instances"][0]["pin_connections"] == [
        {"pin": "D", "net": "D"},
        {"pin": "G", "net": "G"},
        {"pin": "S", "net": "S"},
        {"pin": "B", "net": "B"},
    ]


def test_lvs_parser_and_locator_rank_matching_call(tmp_path):
    from glayout.provenance.runtime import ProvenanceSnapshot
    from glayout.verification.locator import parse_netgen_lvs_report, rank_lvs_candidate_calls

    report = tmp_path / "demo_lvs.rpt"
    report.write_text(
        """
NET mismatches: Class fragments follow
Net: wire0 |Net: IBIAS
(no matching net) |Net: B
DEVICE mismatches: Class fragments follow
Instance: sky130_fd_pr__nfet_01v8:0 |Instance: sky130_fd_pr__nfet_01v8:DUMMY1
Final result:
Netlists do not match.
"""
    )
    parsed = parse_netgen_lvs_report(report)
    assert parsed["issue_count"] >= 2

    snapshot = ProvenanceSnapshot(
        {
            "calls": {
                "call_000001": {
                    "call_id": "call_000001",
                    "generator_id": "diff_pair_ibias",
                    "function_name": "diff_pair_ibias",
                    "module": "demo",
                    "parent_call_id": None,
                    "callsite": {"file": "demo.py", "line": 10, "function": "build"},
                    "definition": {"file": "cell.py", "line": 20},
                    "params": {},
                    "netlist_summary": {
                        "circuit_name": "DIFF_PAIR_IBIAS",
                        "nodes": ["IBIAS", "B", "VSS"],
                        "instances": [
                            {
                                "name": "0",
                                "circuit_name": "sky130_fd_pr__nfet_01v8",
                                "pin_connections": [{"pin": "G", "net": "IBIAS"}],
                            }
                        ],
                        "net_fanout": [{"net": "IBIAS", "pins": []}],
                    },
                }
            },
            "objects": {},
            "artifacts": {},
            "pdk": {},
            "source_hashes": {},
        }
    )
    candidates = rank_lvs_candidate_calls(snapshot, parsed["issues"][0])
    assert candidates
    assert candidates[0]["call_id"] == "call_000001"
    assert candidates[0]["matched_terms"]
    assert candidates[0]["netlist_excerpt"]["matched_fanout"][0]["net"] == "IBIAS"


def test_lvs_repair_packet_highlights_missing_route_and_floating_label():
    from glayout.provenance.runtime import ProvenanceSnapshot
    from glayout.verification.locator import build_lvs_repair_packet, summarize_repair_packet

    snapshot = ProvenanceSnapshot(
        {
            "calls": {
                "call_000001": {
                    "call_id": "call_000001",
                    "generator_id": "diff_pair_ibias",
                    "definition": {"file": "missing.py", "line": 10},
                    "callsite": {"file": "caller.py", "line": 20},
                    "ports": [{"name": "ibias_B_drain_N"}, {"name": "source_routeE_con_S"}],
                    "port_count_total": 2,
                }
            },
            "objects": {},
            "artifacts": {},
            "pdk": {},
            "source_hashes": {},
        }
    )
    lvs = {
        "status": "mismatch",
        "matched": False,
        "netlists_matched": False,
        "issue_count": 3,
        "issues": [
            {
                "kind": "lvs_net_mismatch",
                "raw": "Net: layout_tail |Net: wire0",
                "left_net": "layout_tail",
                "right_net": "wire0",
                "candidate_calls": [{"call_id": "call_000001", "score": 5}],
            },
            {
                "kind": "lvs_net_mismatch",
                "raw": "Net: VSS |Net: VSS",
                "left_net": "VSS",
                "right_net": "VSS",
                "candidate_calls": [{"call_id": "call_000001", "score": 1}],
            },
        ],
    }
    layout_summary = {
        "nodes": ["VSS"],
        "net_fanout": [
            {
                "net": "layout_tail",
                "pin_count": 2,
                "pins": [
                    {"instance": "X0", "circuit_name": "nfet", "pin": "D"},
                    {"instance": "X1", "circuit_name": "nfet", "pin": "S"},
                ],
            }
        ],
    }
    schematic_summary = {
        "nodes": ["VSS"],
        "net_fanout": [
            {
                "net": "wire0",
                "pin_count": 2,
                "pins": [
                    {"instance": "X0", "circuit_name": "DIFF_PAIR", "pin": "VTAIL"},
                    {"instance": "X1", "circuit_name": "CMIRROR", "pin": "VOUT"},
                ],
            },
            {
                "net": "VSS",
                "pin_count": 1,
                "pins": [{"instance": "X1", "circuit_name": "CMIRROR", "pin": "VSS"}],
            },
        ],
    }
    drc = {
        "issue_count": 1,
        "issues": [
            {
                "rule": "Metal2 spacing < 0.14um (met2.2)",
                "layer_hint": "met2",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "candidate_calls": [{"call_id": "call_000001", "score": 3}],
            }
        ],
    }
    packet = build_lvs_repair_packet(snapshot, lvs, layout_summary, schematic_summary, drc=drc)
    assert "missing_route_for_schematic_internal_net" in packet["primary_hint_types"]
    assert "floating_or_misplaced_top_label" in packet["primary_hint_types"]
    assert "drc_marker_cluster" in packet["primary_hint_types"]
    assert packet["component_port_manifest"][0]["ports"][0]["name"] == "ibias_B_drain_N"
    assert packet["component_port_manifest"][0]["ports_included"] == 2
    assert packet["drc_repair_hints"][0]["layer_hint"] == "met2"
    summary = summarize_repair_packet(packet)
    assert summary["repair_hint_count"] == 2
    assert summary["drc_repair_hint_count"] == 1
    assert summary["unmatched_net_count"] == 2
    assert "source_spans" not in summary


def test_lvs_repair_packet_points_label_text_mismatch_to_source(tmp_path):
    from glayout.provenance.runtime import ProvenanceSnapshot
    from glayout.verification.locator import build_lvs_repair_packet, summarize_repair_packet

    source = tmp_path / "label_cell.py"
    source.write_text(
        "\n".join(
            [
                "def add_labels(component, pdk):",
                "    pin = rectangle(layer=pdk.get_glayer('met2_pin'), size=(0.27, 0.27), centered=True).copy()",
                '    pin.add_label(text="VOUT_BAD",layer=pdk.get_glayer("met2_label"))',
                "    component.add(pin)",
            ]
        )
    )
    snapshot = ProvenanceSnapshot(
        {
            "calls": {
                "call_000001": {
                    "call_id": "call_000001",
                    "generator_id": "current_mirror",
                    "definition": {"file": str(source), "line": 1},
                    "callsite": {"file": str(source), "line": 1},
                    "ports": [],
                    "port_count_total": 0,
                }
            },
            "objects": {},
            "artifacts": {},
            "pdk": {},
            "source_hashes": {},
        }
    )
    lvs = {
        "status": "top_level_pin_mismatch",
        "matched": False,
        "netlists_matched": True,
        "issue_count": 1,
        "issues": [
            {
                "kind": "lvs_pin_mismatch",
                "raw": "VOUT_BAD |VOUT **Mismatch**",
                "left": "VOUT_BAD",
                "right": "VOUT",
                "candidate_calls": [],
            },
        ],
    }
    layout_summary = {"nodes": ["VOUT_BAD"], "net_fanout": []}
    schematic_summary = {"nodes": ["VOUT"], "net_fanout": []}

    packet = build_lvs_repair_packet(snapshot, lvs, layout_summary, schematic_summary)

    assert packet["source_label_candidates"][0]["label"] == "VOUT_BAD"
    assert packet["source_label_candidates"][0]["line"] == 3
    assert "possible_top_label_text_mismatch" in packet["primary_hint_types"]
    assert any("VOUT_BAD" in span["text"] for span in packet["source_spans"])
    summary = summarize_repair_packet(packet)
    assert summary["source_label_candidate_count"] == 1
