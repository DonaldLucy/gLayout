from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.routing.route_quad import route_quad

from glayout.cells.elementary.diff_pair.diff_pair import diff_pair_netlist
from glayout.pdk.sky130_mapped import sky130_mapped_pdk
from glayout.primitives.fet import nmos, pmos
from glayout.primitives.via_gen import via_stack
from glayout.routing.c_route import c_route
from glayout.util.comp_utils import evaluate_bbox, movey
from glayout.util.port_utils import (
    get_orientation,
    rename_ports_by_orientation,
    set_port_orientation,
)
from glayout.util.snap_to_grid import component_snap_to_grid


def _bbox_to_list(ref) -> list[list[float]]:
    bbox = ref.bbox
    return [
        [float(bbox[0][0]), float(bbox[0][1])],
        [float(bbox[1][0]), float(bbox[1][1])],
    ]


def build_intentional_drc_fail_diff_pair(
    width: float = 3.0,
    length: float = 0.15,
    fingers: int = 4,
) -> tuple[Component, dict[str, object]]:
    pdk = sky130_mapped_pdk
    if pdk is None:
        raise RuntimeError(
            "sky130_mapped_pdk is unavailable. Set PDK_ROOT and PYTHONPATH before running."
        )
    pdk.activate()

    diffpair = Component("intentional_drc_fail_diff_pair")
    fetL = nmos(
        pdk,
        width=width,
        fingers=fingers,
        length=length,
        multipliers=1,
        with_tie=False,
        with_dummy=(True, False),
        with_dnwell=False,
        with_substrate_tap=False,
        rmult=1,
    )
    fetR = nmos(
        pdk,
        width=width,
        fingers=fingers,
        length=length,
        multipliers=1,
        with_tie=False,
        with_dummy=(False, True),
        with_dnwell=False,
        with_substrate_tap=False,
        rmult=1,
    )
    min_spacing_x = pdk.get_grule("n+s/d")["min_separation"] - 2 * (
        fetL.xmax - fetL.ports["multiplier_0_plusdoped_E"].center[0]
    )

    viam2m3 = via_stack(pdk, "met2", "met3", centered=True)
    metal_min_dim = max(
        pdk.get_grule("met2")["min_width"],
        pdk.get_grule("met3")["min_width"],
    )
    metal_space = max(
        pdk.get_grule("met2")["min_separation"],
        pdk.get_grule("met3")["min_separation"],
        metal_min_dim,
    )
    gate_route_os = (
        evaluate_bbox(viam2m3)[0] - fetL.ports["multiplier_0_gate_W"].width + metal_space
    )
    min_spacing_y = metal_space + 2 * gate_route_os
    min_spacing_y = min_spacing_y - 2 * abs(
        fetL.ports["well_S"].center[1] - fetL.ports["multiplier_0_gate_S"].center[1]
    )

    a_topl = (diffpair << fetL).movey(fetL.ymax + min_spacing_y / 2 + 0.5).movex(
        0 - fetL.xmax - min_spacing_x / 2
    )
    b_topr = (diffpair << fetR).movey(fetR.ymax + min_spacing_y / 2 + 0.5).movex(
        fetL.xmax + min_spacing_x / 2
    )
    a_botr = (diffpair << fetR)
    a_botr = a_botr.mirror_y()
    a_botr.movey(0 - 0.5 - fetL.ymax - min_spacing_y / 2).movex(
        fetL.xmax + min_spacing_x / 2
    )
    b_botl = (diffpair << fetL)
    b_botl = b_botl.mirror_y()
    b_botl.movey(0 - 0.5 - fetR.ymax - min_spacing_y / 2).movex(
        0 - fetL.xmax - min_spacing_x / 2
    )

    diffpair << route_quad(
        a_topl.ports["multiplier_0_source_E"],
        b_topr.ports["multiplier_0_source_W"],
        layer=pdk.get_glayer("met2"),
    )
    diffpair << route_quad(
        b_botl.ports["multiplier_0_source_E"],
        a_botr.ports["multiplier_0_source_W"],
        layer=pdk.get_glayer("met2"),
    )
    sextension = (
        b_topr.ports["well_E"].center[0]
        - b_topr.ports["multiplier_0_source_E"].center[0]
    )
    source_routeE = diffpair << c_route(
        pdk,
        b_topr.ports["multiplier_0_source_E"],
        a_botr.ports["multiplier_0_source_E"],
        extension=sextension,
        viaoffset=False,
    )
    source_routeW = diffpair << c_route(
        pdk,
        a_topl.ports["multiplier_0_source_W"],
        b_botl.ports["multiplier_0_source_W"],
        extension=sextension,
        viaoffset=False,
    )

    drain_br_via = diffpair << viam2m3
    drain_bl_via = diffpair << viam2m3
    drain_br_via.move(a_botr.ports["multiplier_0_drain_N"].center).movey(viam2m3.ymin)
    drain_bl_via.move(b_botl.ports["multiplier_0_drain_N"].center).movey(viam2m3.ymin)
    drain_br_viatm = diffpair << viam2m3
    drain_bl_viatm = diffpair << viam2m3
    drain_br_viatm.move(a_botr.ports["multiplier_0_drain_N"].center).movey(viam2m3.ymin)
    drain_bl_viatm.move(b_botl.ports["multiplier_0_drain_N"].center).movey(
        -1.5 * evaluate_bbox(viam2m3)[1] - metal_space
    )
    width_drain_route = b_topr.ports["multiplier_0_drain_E"].width
    dextension = (
        source_routeE.xmax - b_topr.ports["multiplier_0_drain_E"].center[0] + metal_space
    )
    bottom_extension = viam2m3.ymax + width_drain_route / 2 + 2 * metal_space
    drain_br_viatm.movey(
        0 - bottom_extension - metal_space - width_drain_route / 2 - viam2m3.ymax
    )
    diffpair << route_quad(
        drain_br_viatm.ports["top_met_N"],
        drain_br_via.ports["top_met_S"],
        layer=pdk.get_glayer("met3"),
    )
    diffpair << route_quad(
        drain_bl_viatm.ports["top_met_N"],
        drain_bl_via.ports["top_met_S"],
        layer=pdk.get_glayer("met3"),
    )
    floating_port_drain_bottom_L = set_port_orientation(
        movey(drain_bl_via.ports["bottom_met_W"], 0 - bottom_extension),
        get_orientation("E"),
    )
    floating_port_drain_bottom_R = set_port_orientation(
        movey(
            drain_br_via.ports["bottom_met_E"],
            0 - bottom_extension - metal_space - width_drain_route,
        ),
        get_orientation("W"),
    )
    drain_routeTR_BL = diffpair << c_route(
        pdk,
        floating_port_drain_bottom_L,
        b_topr.ports["multiplier_0_drain_E"],
        extension=dextension,
        width1=width_drain_route,
        width2=width_drain_route,
    )
    drain_routeTL_BR = diffpair << c_route(
        pdk,
        floating_port_drain_bottom_R,
        a_topl.ports["multiplier_0_drain_W"],
        extension=dextension,
        width1=width_drain_route,
        width2=width_drain_route,
    )

    get_left_extension = lambda bar: (
        abs(diffpair.xmin - min(a_topl.ports["multiplier_0_gate_W"].center[0], bar.ports["e1"].center[0]))
        + pdk.get_grule("met2")["min_separation"]
    )
    get_right_extension = lambda bar: (
        abs(diffpair.xmax - max(b_topr.ports["multiplier_0_gate_E"].center[0], bar.ports["e3"].center[0]))
        + pdk.get_grule("met2")["min_separation"]
    )
    bar_comp = rectangle(
        centered=True,
        size=(abs(b_topr.xmax - a_topl.xmin), b_topr.ports["multiplier_0_gate_E"].width),
        layer=pdk.get_glayer("met2"),
    )
    bar_plus = (diffpair << bar_comp).movey(
        diffpair.ymax + bar_comp.ymax + pdk.get_grule("met2")["min_separation"]
    )
    PLUSgate_routeW = diffpair << c_route(
        pdk,
        a_topl.ports["multiplier_0_gate_W"],
        bar_plus.ports["e1"],
        extension=get_left_extension(bar_plus),
    )

    # Intentional DRC fail hotspot:
    # Original source geometry comes from src/glayout/cells/elementary/diff_pair/diff_pair.py
    # around lines 214-219. Instead of using the legal minimum spacing between the two
    # MET2 gate bars, we place `bar_minus` only a quarter of the minimum spacing above
    # `bar_plus`, forcing a real spacing/overlap violation on an existing route family.
    met2_min_sep = float(pdk.get_grule("met2")["min_separation"])
    bad_gate_bar_shift = 0.25 * met2_min_sep
    bar_minus = (diffpair << bar_comp).movey(bar_plus.center[1] + bad_gate_bar_shift)

    MINUSgate_routeE = diffpair << c_route(
        pdk,
        b_topr.ports["multiplier_0_gate_E"],
        bar_minus.ports["e3"],
        extension=get_right_extension(bar_minus),
    )
    MINUSgate_routeW = diffpair << c_route(
        pdk,
        set_port_orientation(b_botl.ports["multiplier_0_gate_E"], "W"),
        bar_minus.ports["e1"],
        extension=get_left_extension(bar_minus),
    )
    PLUSgate_routeE = diffpair << c_route(
        pdk,
        set_port_orientation(a_botr.ports["multiplier_0_gate_W"], "E"),
        bar_plus.ports["e3"],
        extension=get_right_extension(bar_plus),
    )

    diffpair.add_ports(a_topl.get_ports_list(), prefix="tl_")
    diffpair.add_ports(b_topr.get_ports_list(), prefix="tr_")
    diffpair.add_ports(b_botl.get_ports_list(), prefix="bl_")
    diffpair.add_ports(a_botr.get_ports_list(), prefix="br_")
    diffpair.add_ports(source_routeE.get_ports_list(), prefix="source_routeE_")
    diffpair.add_ports(source_routeW.get_ports_list(), prefix="source_routeW_")
    diffpair.add_ports(drain_routeTR_BL.get_ports_list(), prefix="drain_routeTR_BL_")
    diffpair.add_ports(drain_routeTL_BR.get_ports_list(), prefix="drain_routeTL_BR_")
    diffpair.add_ports(MINUSgate_routeW.get_ports_list(), prefix="MINUSgateroute_W_")
    diffpair.add_ports(MINUSgate_routeE.get_ports_list(), prefix="MINUSgateroute_E_")
    diffpair.add_ports(PLUSgate_routeW.get_ports_list(), prefix="PLUSgateroute_W_")
    diffpair.add_ports(PLUSgate_routeE.get_ports_list(), prefix="PLUSgateroute_E_")
    diffpair.add_padding(layers=(pdk.get_glayer("pwell"),), default=0)

    top = component_snap_to_grid(rename_ports_by_orientation(diffpair))
    top.info["netlist"] = diff_pair_netlist(fetL, fetR)

    metadata = {
        "rule": "met2 minimum separation",
        "met2_min_separation": met2_min_sep,
        "actual_gate_bar_shift": bad_gate_bar_shift,
        "description": (
            "The existing top MET2 gate bars inside the diff_pair generator were "
            "intentionally moved too close together to trigger a real spacing/overlap DRC violation."
        ),
        "hotspot": {
            "bar_plus_bbox": _bbox_to_list(bar_plus),
            "bar_minus_bbox": _bbox_to_list(bar_minus),
            "center_y": float((bar_plus.center[1] + bar_minus.center[1]) / 2.0),
        },
        "code_hint": {
            "file": "scripts/generate_intentional_diff_pair_drc_fail.py",
            "note": (
                "The intentional violation is created where `bar_minus` is positioned relative "
                "to `bar_plus`. This mirrors the original diff_pair gate-bar construction and "
                "modifies the legal spacing logic from the source generator."
            ),
            "source_reference": {
                "file": "src/glayout/cells/elementary/diff_pair/diff_pair.py",
                "original_lines": [214, 219],
                "modified_lines": [145, 168],
            },
        },
    }
    return top, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an intentionally DRC-failing diff pair layout."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/manual_intentional_diff_pair_drc_fail"),
    )
    parser.add_argument("--width", type=float, default=3.0)
    parser.add_argument("--length", type=float, default=0.15)
    parser.add_argument("--fingers", type=int, default=4)
    parser.add_argument("--skip-drc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layout, metadata = build_intentional_drc_fail_diff_pair(
        width=args.width,
        length=args.length,
        fingers=args.fingers,
    )

    gds_path = output_dir / f"{layout.name}.gds"
    layout.write_gds(gds_path)

    metadata_path = output_dir / f"{layout.name}.hotspot.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"GDS: {gds_path}")
    print(f"Hotspot metadata: {metadata_path}")

    if not args.skip_drc:
        drc_output_dir = output_dir / "magic_drc"
        explicit_magicrc = Path(os.environ["PDKPATH"]) / "libs.tech" / "magic" / "sky130A.magicrc"
        result = sky130_mapped_pdk.drc_magic(
            layout,
            layout.name,
            pdk_root=Path(os.environ["PDK_ROOT"]),
            magic_drc_file=explicit_magicrc,
            output_file=drc_output_dir,
        )
        report_path = (
            drc_output_dir
            / "drc"
            / layout.name
            / f"{layout.name}.rpt"
        )
        print(f"Magic DRC report: {report_path}")
        print(f"Magic DRC result: {result}")


if __name__ == "__main__":
    main()
