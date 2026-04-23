from __future__ import annotations

import argparse
from pathlib import Path

from gdsfactory.component import Component
from glayout.pdk.sky130_mapped import sky130_mapped_pdk
from glayout.primitives.fet import nmos
from glayout.primitives.guardring import tapring
from glayout.spice import Netlist
from glayout.util.comp_utils import evaluate_bbox
from glayout.util.port_utils import add_ports_perimeter, rename_ports_by_orientation


def two_fet_separate_netlist(width: float, length: float) -> Netlist:
    return Netlist(
        circuit_name="TWO_FET_SEPARATE",
        nodes=["LD", "LG", "LS", "RD", "RG", "RS", "B"],
        source_netlist=(
            ".subckt {circuit_name} {nodes} l={length} w={width}\n"
            "XLEFT LD LG LS B sky130_fd_pr__nfet_01v8 l={length} w={width}\n"
            "XRIGHT RD RG RS B sky130_fd_pr__nfet_01v8 l={length} w={width}\n"
            ".ends {circuit_name}"
        ),
        instance_format="X{name} {nodes} {circuit_name} l={length} w={width}",
        parameters={"length": length, "width": width},
    )


def build_two_fet_separate(
    width: float = 2.0,
    length: float = 0.15,
    fingers: int = 1,
) -> Component:
    pdk = sky130_mapped_pdk
    if pdk is None:
        raise RuntimeError(
            "sky130_mapped_pdk is unavailable. Set PDK_ROOT and PYTHONPATH before running."
        )
    pdk.activate()

    top_level = Component("nfet_two_fet_separate")
    left = top_level << nmos(
        pdk=pdk,
        width=width,
        length=length,
        fingers=fingers,
        multipliers=1,
        with_tie=False,
        with_dummy=(True, False),
        with_dnwell=False,
        with_substrate_tap=False,
    )
    right = top_level << nmos(
        pdk=pdk,
        width=width,
        length=length,
        fingers=fingers,
        multipliers=1,
        with_tie=False,
        with_dummy=(False, True),
        with_dnwell=False,
        with_substrate_tap=False,
    )
    spacing = evaluate_bbox(left)[0] + 4 * pdk.util_max_metal_seperation()
    left.movex(-spacing / 2)
    right.movex(spacing / 2)

    top_level.add_ports(left.get_ports_list(), prefix="left_")
    top_level.add_ports(right.get_ports_list(), prefix="right_")

    tap_separation = max(
        float(pdk.util_max_metal_seperation()),
        float(pdk.get_grule("active_diff", "active_tap")["min_separation"]),
    )
    tap_separation += float(pdk.get_grule("p+s/d", "active_tap")["min_enclosure"])
    tap_ref = top_level << tapring(
        pdk,
        enclosed_rectangle=(
            2 * (tap_separation + top_level.xmax),
            2 * (tap_separation + top_level.ymax),
        ),
        sdlayer="p+s/d",
        horizontal_glayer="met2",
        vertical_glayer="met1",
    )
    top_level.add_ports(tap_ref.get_ports_list(), prefix="welltie_")
    top_level.add_padding(
        layers=(pdk.get_glayer("pwell"),),
        default=pdk.get_grule("pwell", "active_tap")["min_enclosure"],
    )
    top_level = add_ports_perimeter(
        top_level,
        layer=pdk.get_glayer("pwell"),
        prefix="well_",
    )
    top_level = rename_ports_by_orientation(top_level)
    top_level.info["netlist"] = two_fet_separate_netlist(width=width, length=length)
    return top_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--length", type=float, default=0.15)
    parser.add_argument("--fingers", type=int, default=1)
    parser.add_argument("--output-gds", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    component = build_two_fet_separate(
        width=args.width,
        length=args.length,
        fingers=args.fingers,
    )
    output_path = args.output_gds.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    component.write_gds(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
