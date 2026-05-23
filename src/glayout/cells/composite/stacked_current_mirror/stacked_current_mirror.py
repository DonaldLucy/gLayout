from glayout import MappedPDK, sky130,gf180
from gdsfactory.cell import cell, clear_cache
from gdsfactory.component import Component, copy
from gdsfactory.component_reference import ComponentReference
from gdsfactory.components.rectangle import rectangle
from typing import Optional, Union
from glayout.primitives.fet import fet_netlist, nmos, pmos, multiplier
from glayout.cells.elementary.diff_pair import diff_pair
from glayout.primitives.guardring import tapring
from glayout.primitives.mimcap import mimcap_array, mimcap
from glayout.routing import c_route,L_route,straight_route
from glayout.primitives.via_gen import via_stack, via_array
from gdsfactory.routing.route_quad import route_quad
from glayout.util.comp_utils import evaluate_bbox, prec_ref_center, movex, movey, to_decimal, to_float, move, align_comp_to_port, get_padding_points_cc
from glayout.util.port_utils import rename_ports_by_orientation, rename_ports_by_list, add_ports_perimeter, print_ports, set_port_orientation, rename_component_ports
from glayout.util.snap_to_grid import component_snap_to_grid
from pydantic import validate_arguments
from glayout.placement.two_transistor_interdigitized import two_nfet_interdigitized
from glayout.spice import Netlist
from glayout.provenance import tracked_generator


def _private_fet_mapping(prefix: str) -> list[tuple[str, str]]:
    return [(node, f"{prefix}_{node}") for node in ("D", "G", "S", "B")]


def stacked_nfet_current_mirror_netlist(
    pdk: MappedPDK,
    half_common_source_nbias: tuple[float, float, int, int],
    rmult: int,
    *,
    circuit_name: str = "STACKED_NFET_CURRENT_MIRROR",
) -> Netlist:
    netlist = Netlist(
        circuit_name=circuit_name,
        nodes=["REF_D", "REF_G", "REF_S", "REF_B", "OUT_D", "OUT_G", "OUT_S"],
    )
    ref_netlist = fet_netlist(
        pdk,
        circuit_name="NMOS",
        model=pdk.models["nfet"],
        width=half_common_source_nbias[0],
        length=half_common_source_nbias[1],
        fingers=half_common_source_nbias[2],
        multipliers=1,
        with_dummy=True,
    )
    output_netlist = fet_netlist(
        pdk,
        circuit_name="NMOS",
        model=pdk.models["nfet"],
        width=half_common_source_nbias[0],
        length=half_common_source_nbias[1],
        fingers=half_common_source_nbias[2],
        multipliers=half_common_source_nbias[3],
        with_dummy=True,
    )
    netlist.connect_netlist(ref_netlist, _private_fet_mapping("REF"))
    netlist.connect_netlist(
        output_netlist,
        [("D", "OUT_D"), ("G", "OUT_G"), ("S", "OUT_S"), ("B", "REF_B")],
    )
    return netlist


@validate_arguments
@tracked_generator("stacked_nfet_current_mirror")
def stacked_nfet_current_mirror(pdk: MappedPDK, half_common_source_nbias: tuple[float, float, int, int], rmult: int, sd_route_left: bool) -> Component:
    cmirror_output = nmos(
        pdk,
        width=half_common_source_nbias[0],
        length=half_common_source_nbias[1],
        fingers=half_common_source_nbias[2],
        multipliers=half_common_source_nbias[3],
        with_tie=True,
        with_dnwell=False,
        with_substrate_tap=False,
        with_dummy=True,
        sd_route_left = sd_route_left,
        rmult=rmult,
        tie_layers=("met2","met2")
    )
    cmirrorref = nmos(
        pdk,
        width=half_common_source_nbias[0],
        length=half_common_source_nbias[1],
        fingers=half_common_source_nbias[2],
        multipliers=1,
        with_tie=True,
        with_dnwell=False,
        with_substrate_tap=False,
        with_dummy=True,
        sd_route_left = sd_route_left,
        rmult=rmult,
        tie_layers=("met2","met2")
    )
    cmirrorref_ref = prec_ref_center(cmirrorref)
    cmirrorout_ref = prec_ref_center(cmirror_output)
    return cmirrorref_ref, cmirrorout_ref

# Create and evaluate a current mirror instance
if __name__ == "__main__":
    cm = stacked_nfet_current_mirror(
        pdk=sky130,
        half_common_source_nbias=(0.5, 0.15, 4, 4),
        rmult=2,
        sd_route_left=True
    )
    print(cm)
    cm.show()
    cm_gds = cm.write_gds("stacked_nfet_current_mirror.gds")
