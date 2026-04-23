from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from gdsfactory.component import Component
from glayout.pdk.sky130_mapped import sky130_mapped_pdk
from glayout.primitives.fet import multiplier
from glayout.primitives.guardring import tapring
from glayout.primitives.via_gen import via_stack
from glayout.spice import Netlist
from glayout.util.comp_utils import align_comp_to_port
from glayout.util.port_utils import add_ports_perimeter, rename_ports_by_orientation


def two_fet_shared_diffusion_netlist(
    width: float,
    length: float,
    with_dummy: bool,
    device: Literal["nfet", "pfet"] = "nfet",
) -> Netlist:
    model = (
        "sky130_fd_pr__nfet_01v8"
        if device == "nfet"
        else "sky130_fd_pr__pfet_01v8"
    )
    source_netlist = (
        ".subckt {circuit_name} {nodes} l={length} w={width}\n"
        "XLEFT MID GLEFT LEFT B {model} l={length} w={width}\n"
        "XRIGHT RIGHT GRIGHT MID B {model} l={length} w={width}"
    )
    if with_dummy:
        source_netlist += (
            "\nXDUMMYL B B B B {model} l={length} w={width}"
            "\nXDUMMYR B B B B {model} l={length} w={width}"
        )
    source_netlist += "\n.ends {circuit_name}"
    return Netlist(
        circuit_name="TWO_FET_SHARED_DIFFUSION",
        nodes=["LEFT", "GLEFT", "MID", "GRIGHT", "RIGHT", "B"],
        source_netlist=source_netlist,
        instance_format="X{name} {nodes} {circuit_name} l={length} w={width}",
        parameters={
            "model": model,
            "width": width,
            "length": length,
        },
    )


def build_two_fet_shared_diffusion(
    width: float = 2.0,
    length: float = 0.15,
    fingers: int = 1,
    device: Literal["nfet", "pfet"] = "nfet",
    with_dummy: bool = True,
) -> Component:
    if fingers != 1:
        raise ValueError(
            "Explicit shared-diffusion hello world models two devices with one finger each."
        )

    pdk = sky130_mapped_pdk
    if pdk is None:
        raise RuntimeError(
            "sky130_mapped_pdk is unavailable. Set PDK_ROOT and PYTHONPATH before running."
        )
    pdk.activate()

    sdlayer = "n+s/d" if device == "nfet" else "p+s/d"
    well_layer = "pwell" if device == "nfet" else "nwell"
    tie_sdlayer = "p+s/d" if device == "nfet" else "n+s/d"

    top_level = Component(f"{device}_two_fet_shared_diffusion")
    core = multiplier(
        pdk=pdk,
        sdlayer=sdlayer,
        width=width,
        length=length,
        fingers=2,
        routing=False,
        dummy=with_dummy,
        inter_finger_topmet="met1",
        dummy_routes=True,
    )
    core_ref = top_level << core
    top_level.add_ports(core_ref.get_ports_list(), prefix="core_")

    gate_via_template = via_stack(pdk, "poly", "met1", fullbottom=True)
    gate0_via = top_level << gate_via_template
    align_comp_to_port(
        gate0_via,
        core_ref.ports["row0_col0_gate_S"],
        layer=pdk.get_glayer("poly"),
    )
    gate1_via = top_level << gate_via_template
    align_comp_to_port(
        gate1_via,
        core_ref.ports["row0_col1_gate_S"],
        layer=pdk.get_glayer("poly"),
    )

    top_level.add_ports(gate0_via.get_ports_list(), prefix="gate_left_")
    top_level.add_ports(gate1_via.get_ports_list(), prefix="gate_right_")
    top_level.add_ports(core_ref.get_ports_list(), prefix="raw_")

    # Export the three explicit diffusion terminals so the shared node is visible.
    for port in core_ref.get_ports_list():
        if port.name.startswith("leftsd_top_met_"):
            top_level.add_port(name=f"left_diff_{port.name.rsplit('_', 1)[-1]}", port=port)
        elif port.name.startswith("row0_col0_rightsd_top_met_"):
            top_level.add_port(
                name=f"shared_diff_{port.name.rsplit('_', 1)[-1]}",
                port=port,
            )
        elif port.name.startswith("row0_col1_rightsd_top_met_"):
            top_level.add_port(
                name=f"right_diff_{port.name.rsplit('_', 1)[-1]}",
                port=port,
            )

    tap_separation = max(
        float(pdk.util_max_metal_seperation()),
        float(pdk.get_grule("active_diff", "active_tap")["min_separation"]),
    )
    tap_separation += float(pdk.get_grule(tie_sdlayer, "active_tap")["min_enclosure"])
    tap_encloses = (
        2 * (tap_separation + core_ref.xmax),
        2 * (tap_separation + core_ref.ymax),
    )
    tie_ref = top_level << tapring(
        pdk,
        enclosed_rectangle=tap_encloses,
        sdlayer=tie_sdlayer,
        horizontal_glayer="met2",
        vertical_glayer="met1",
    )
    top_level.add_ports(tie_ref.get_ports_list(), prefix="welltie_")

    for source_port, tie_port in [
        ("core_dummy_L_gsdcon_top_met_W", "welltie_W_top_met_W"),
        ("core_dummy_R_gsdcon_top_met_E", "welltie_E_top_met_E"),
    ]:
        try:
            from glayout.routing.straight_route import straight_route

            top_level << straight_route(
                pdk,
                top_level.ports[source_port],
                top_level.ports[tie_port],
                glayer2="met1",
            )
        except KeyError:
            pass

    top_level.add_padding(
        layers=(pdk.get_glayer(well_layer),),
        default=pdk.get_grule(well_layer, "active_tap")["min_enclosure"],
    )
    top_level = add_ports_perimeter(
        top_level,
        layer=pdk.get_glayer(well_layer),
        prefix="well_",
    )
    top_level = rename_ports_by_orientation(top_level)
    top_level.info["netlist"] = two_fet_shared_diffusion_netlist(
        width=width,
        length=length,
        with_dummy=with_dummy,
        device=device,
    )
    return top_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--length", type=float, default=0.15)
    parser.add_argument("--fingers", type=int, default=1)
    parser.add_argument("--device", choices=["nfet", "pfet"], default="nfet")
    parser.add_argument("--output-gds", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    component = build_two_fet_shared_diffusion(
        width=args.width,
        length=args.length,
        fingers=args.fingers,
        device=args.device,
    )
    output_path = args.output_gds.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    component.write_gds(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
