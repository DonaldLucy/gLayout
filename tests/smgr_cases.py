from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SmgrCase:
    case_id: str
    description: str
    builder: Callable[[], object]


def _sky130():
    from glayout import sky130

    if sky130 is None:
        raise RuntimeError("glayout.sky130 is unavailable in this environment")
    return sky130


def build_diff_pair_default():
    from glayout.cells.elementary.diff_pair import diff_pair
    from glayout.cells.elementary.diff_pair.diff_pair import add_df_labels

    return add_df_labels(diff_pair(_sky130()), _sky130())


def build_diff_pair_pmos():
    from glayout.cells.elementary.diff_pair import diff_pair
    from glayout.cells.elementary.diff_pair.diff_pair import add_df_labels

    return add_df_labels(diff_pair(_sky130(), width=2.0, fingers=2, n_or_p_fet=False), _sky130())


def build_diff_pair_generic():
    from glayout.cells.elementary.diff_pair import diff_pair_generic

    return diff_pair_generic(_sky130(), width=4.0, fingers=4, rmult=2)


def build_current_mirror_nfet():
    from glayout.cells.elementary.current_mirror import current_mirror

    return current_mirror(_sky130(), device="nfet", numcols=2, width=3.0, length=0.5, fingers=2)


def build_current_mirror_pfet():
    from glayout.cells.elementary.current_mirror import current_mirror

    return current_mirror(_sky130(), device="pfet", numcols=2, width=3.0, length=0.5, fingers=2)


def build_transmission_gate():
    from glayout.cells.elementary.transmission_gate import transmission_gate

    return transmission_gate(_sky130(), width=(2.0, 2.0), length=(0.5, 0.5), multipliers=(1, 1))


def build_flipped_voltage_follower():
    from glayout.cells.elementary.FVF import flipped_voltage_follower

    return flipped_voltage_follower(_sky130(), width=(2.0, 1.0), length=(1.0, 1.0), sd_rmult=2)


def build_low_voltage_cmirror():
    from glayout.cells.composite.low_voltage_cmirror import low_voltage_cmirror

    return low_voltage_cmirror(_sky130())


def build_differential_to_single_ended_converter():
    from glayout.cells.composite.differential_to_single_ended_converter import differential_to_single_ended_converter

    return differential_to_single_ended_converter(_sky130(), rmult=2, half_pload=(6.0, 1.0, 4), via_xlocation=0.0)


def build_diff_pair_ibias():
    from glayout.cells.composite.diffpair_cmirror_bias import diff_pair_ibias

    return diff_pair_ibias(
        _sky130(),
        half_diffpair_params=(6.0, 1.0, 4),
        diffpair_bias=(6.0, 2.0, 4),
        rmult=2,
        with_antenna_diode_on_diffinputs=0,
    )


def build_stacked_nfet_current_mirror():
    from glayout.cells.composite.stacked_current_mirror import stacked_nfet_current_mirror
    from gdsfactory.component import Component

    left_ref, right_ref = stacked_nfet_current_mirror(_sky130(), half_common_source_nbias=(6.0, 1.0, 4, 3), rmult=2, sd_route_left=True)
    top = Component("stacked_nfet_current_mirror_case")
    top.add(left_ref)
    top.add(right_ref)
    top.add_ports(left_ref.get_ports_list(), prefix="left_")
    top.add_ports(right_ref.get_ports_list(), prefix="right_")
    if getattr(left_ref.parent, "info", None) and "netlist" in left_ref.parent.info:
        top.info["netlist"] = left_ref.parent.info["netlist"]
    return top


def build_diff_pair_stackedcmirror_component():
    from glayout.cells.composite.opamp.diff_pair_stackedcmirror import diff_pair_stackedcmirror

    return diff_pair_stackedcmirror(
        _sky130(),
        half_diffpair_params=(6.0, 1.0, 4),
        diffpair_bias=(6.0, 2.0, 4),
        half_common_source_nbias=(6.0, 1.0, 4, 3),
        rmult=2,
        with_antenna_diode_on_diffinputs=0,
    )[0]


def build_opamp_twostage():
    from glayout.cells.composite.opamp.opamp_twostage import opamp_twostage

    return opamp_twostage(_sky130())


def build_opamp():
    from glayout.cells.composite.opamp import opamp

    return opamp(_sky130())


def build_row_csamplifier_diff_to_single_ended_converter():
    from glayout.cells.composite.differential_to_single_ended_converter import differential_to_single_ended_converter
    from glayout.cells.composite.opamp.row_csamplifier_diff_to_single_ended_converter import row_csamplifier_diff_to_single_ended_converter

    diff_to_single = differential_to_single_ended_converter(_sky130(), rmult=2, half_pload=(6.0, 1.0, 4), via_xlocation=0.0)
    return row_csamplifier_diff_to_single_ended_converter(_sky130(), diff_to_single, pamp_hparams=(7.0, 1.0, 8, 2), rmult=2)


def build_p_block():
    from glayout.cells.composite.fvf_based_ota.p_block import p_block

    return p_block(_sky130())


def build_n_block():
    from glayout.cells.composite.fvf_based_ota.n_block import n_block

    return n_block(_sky130())


def build_fvf_based_ota_low_voltage_cmirror():
    from glayout.cells.composite.fvf_based_ota.low_voltage_cmirror import low_voltage_cmirror

    return low_voltage_cmirror(_sky130())


def build_super_class_ab_ota():
    from glayout.cells.composite.fvf_based_ota.ota import super_class_AB_OTA

    return super_class_AB_OTA(_sky130())


SMGR_CASES: list[SmgrCase] = [
    SmgrCase("diff_pair_default", "Elementary NMOS diff pair", build_diff_pair_default),
    SmgrCase("diff_pair_pmos", "Elementary PMOS diff pair variant", build_diff_pair_pmos),
    SmgrCase("diff_pair_generic", "Generic diff pair with smart routing", build_diff_pair_generic),
    SmgrCase("current_mirror_nfet", "Elementary NFET current mirror", build_current_mirror_nfet),
    SmgrCase("current_mirror_pfet", "Elementary PFET current mirror", build_current_mirror_pfet),
    SmgrCase("transmission_gate", "Elementary transmission gate", build_transmission_gate),
    SmgrCase("flipped_voltage_follower", "Elementary flipped voltage follower", build_flipped_voltage_follower),
    SmgrCase("low_voltage_cmirror", "Composite low-voltage current mirror", build_low_voltage_cmirror),
    SmgrCase("differential_to_single_ended_converter", "Composite diff-to-single-ended converter", build_differential_to_single_ended_converter),
    SmgrCase("diff_pair_ibias", "Composite diff pair with mirror bias", build_diff_pair_ibias),
    SmgrCase("stacked_nfet_current_mirror", "Composite stacked current mirror", build_stacked_nfet_current_mirror),
    SmgrCase("diff_pair_stackedcmirror_component", "Opamp front-stage stack", build_diff_pair_stackedcmirror_component),
    SmgrCase("row_csamplifier_diff_to_single_ended_converter", "Row common-source amplifier wrapper", build_row_csamplifier_diff_to_single_ended_converter),
    SmgrCase("opamp_twostage", "Two-stage opamp", build_opamp_twostage),
    SmgrCase("opamp", "Top-level opamp", build_opamp),
    SmgrCase("p_block", "FVF OTA p-block", build_p_block),
    SmgrCase("n_block", "FVF OTA n-block", build_n_block),
    SmgrCase("fvf_based_ota_low_voltage_cmirror", "FVF OTA local low-voltage current mirror", build_fvf_based_ota_low_voltage_cmirror),
    SmgrCase("super_class_ab_ota", "FVF-based OTA top-level", build_super_class_ab_ota),
]


def get_case(case_id: str) -> SmgrCase:
    for case in SMGR_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(f"Unknown SMGR case: {case_id}")
