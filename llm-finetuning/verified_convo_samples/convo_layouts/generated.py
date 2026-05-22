from __future__ import annotations

from glayout.pdk.mappedpdk import MappedPDK
from glayout.provenance import tracked_generator

from .common import DeviceSpec, MoveSpec, RouteSpec, build_netlisted_layout


@tracked_generator("syntax_ctat_vgen")
def ctat_vgen(
    pdk: MappedPDK,
    src_width: float = 3.0,
    load_width: float = 3.0,
    src_length: float = 0.5,
    load_length: float = 0.5,
    src_multiplier: int = 1,
    load_multiplier: int = 1,
    src_fingers: int = 2,
    load_fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "ctat_vgen",
        [
            DeviceSpec("src", "nmos", {"width": src_width, "length": src_length, "fingers": src_fingers, "multipliers": src_multiplier}),
            DeviceSpec("load", "nmos", {"width": load_width, "length": load_length, "fingers": load_fingers, "multipliers": load_multiplier}),
        ],
        [MoveSpec("load", "below", "src")],
        [RouteSpec("src_drain_W", "load_source_W")],
    )


@tracked_generator("syntax_cascode_common_gate")
def cascode_common_gate(
    pdk: MappedPDK,
    input_width: float = 3.0,
    output_width: float = 3.0,
    input_length: float = 0.5,
    output_length: float = 0.5,
    input_multipliers: int = 1,
    output_multipliers: int = 1,
    input_fingers: int = 2,
    output_fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "cascode_common_gate",
        [
            DeviceSpec("input", "nmos", {"width": input_width, "length": input_length, "fingers": input_fingers, "multipliers": input_multipliers}),
            DeviceSpec("output", "nmos", {"width": output_width, "length": output_length, "fingers": output_fingers, "multipliers": output_multipliers}),
        ],
        [MoveSpec("output", "above", "input")],
        [RouteSpec("input_source_W", "output_drain_W")],
    )


@tracked_generator("syntax_class_b_push_pull")
def class_b_push_pull(
    pdk: MappedPDK,
    supply_width: float = 3.0,
    absorb_width: float = 3.0,
    supply_length: float = 0.5,
    absorb_length: float = 0.5,
    supply_multiplier: int = 1,
    absorb_multiplier: int = 1,
    supply_fingers: int = 2,
    absorb_fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "class_b_push_pull",
        [
            DeviceSpec("supply", "nmos", {"width": supply_width, "length": supply_length, "fingers": supply_fingers, "multipliers": supply_multiplier}),
            DeviceSpec("absorb", "pmos", {"width": absorb_width, "length": absorb_length, "fingers": absorb_fingers, "multipliers": absorb_multiplier}),
        ],
        [MoveSpec("absorb", "below", "supply")],
        [
            RouteSpec("supply_source_W", "absorb_source_W"),
            RouteSpec("supply_gate_E", "absorb_gate_E"),
        ],
    )


@tracked_generator("syntax_common_source_amplifier")
def common_source_amplifier(
    pdk: MappedPDK,
    input_width: float = 3.0,
    bias_width: float = 3.0,
    input_length: float = 0.5,
    bias_length: float = 0.5,
    input_multiplier: int = 1,
    bias_multiplier: int = 1,
    input_fingers: int = 2,
    bias_fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "common_source_amplifier",
        [
            DeviceSpec("input", "nmos", {"width": input_width, "length": input_length, "fingers": input_fingers, "multipliers": input_multiplier}),
            DeviceSpec("bias", "pmos", {"width": bias_width, "length": bias_length, "fingers": bias_fingers, "multipliers": bias_multiplier}),
        ],
        [MoveSpec("input", "below", "bias")],
        [RouteSpec("input_drain_W", "bias_source_W")],
    )


@tracked_generator("syntax_common_source_amplifier_w_diode_load")
def common_source_amplifier_w_diode_load(
    pdk: MappedPDK,
    input_width: float = 3.0,
    diode_width: float = 3.0,
    input_length: float = 0.5,
    diode_length: float = 0.5,
    input_multiplier: int = 1,
    diode_multiplier: int = 1,
    input_fingers: int = 2,
    diode_fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "common_source_amplifier_w_diode_load",
        [
            DeviceSpec("input", "nmos", {"width": input_width, "length": input_length, "fingers": input_fingers, "multipliers": input_multiplier}),
            DeviceSpec("diode", "nmos", {"width": diode_width, "length": diode_length, "fingers": diode_fingers, "multipliers": diode_multiplier}),
        ],
        [MoveSpec("input", "below", "diode")],
        [
            RouteSpec("diode_source_W", "input_drain_W"),
            RouteSpec("diode_gate_E", "diode_drain_E"),
        ],
    )


@tracked_generator("syntax_current_mirror_ptype")
def current_mirror_ptype(
    pdk: MappedPDK,
    reference_width: float = 3.0,
    mirror_width: float = 3.0,
    reference_length: float = 0.5,
    mirror_length: float = 0.5,
    reference_multiplier: int = 1,
    mirror_multiplier: int = 1,
    reference_fingers: int = 2,
    mirror_fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "current_mirror_ptype",
        [
            DeviceSpec("reference", "pmos", {"width": reference_width, "length": reference_length, "fingers": reference_fingers, "multipliers": reference_multiplier}),
            DeviceSpec("mirror", "pmos", {"width": mirror_width, "length": mirror_length, "fingers": mirror_fingers, "multipliers": mirror_multiplier}),
        ],
        [MoveSpec("mirror", "right", "reference")],
        [
            RouteSpec("reference_gate_E", "mirror_gate_E"),
            RouteSpec("mirror_drain_E", "mirror_gate_E"),
            RouteSpec("reference_source_E", "mirror_source_E"),
        ],
    )


@tracked_generator("syntax_inverter")
def inverter(
    pdk: MappedPDK,
    pullup_width: float = 3.0,
    pulldown_width: float = 3.0,
    pullup_length: float = 0.5,
    pulldown_length: float = 0.5,
    pullup_multiplier: int = 1,
    pulldown_multiplier: int = 1,
    pullup_fingers: int = 2,
    pulldown_fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "inverter",
        [
            DeviceSpec("pullup", "pmos", {"width": pullup_width, "length": pullup_length, "fingers": pullup_fingers, "multipliers": pullup_multiplier}),
            DeviceSpec("pulldown", "nmos", {"width": pulldown_width, "length": pulldown_length, "fingers": pulldown_fingers, "multipliers": pulldown_multiplier}),
        ],
        [MoveSpec("pullup", "above", "pulldown")],
        [
            RouteSpec("pullup_source_E", "pulldown_drain_E"),
            RouteSpec("pullup_gate_W", "pulldown_gate_W"),
        ],
    )


@tracked_generator("syntax_low_noise_amp")
def low_noise_amp(
    pdk: MappedPDK,
    input_width: float = 3.0,
    gain_width: float = 3.0,
    input_length: float = 0.5,
    gain_length: float = 0.5,
    input_multiplier: int = 1,
    gain_multiplier: int = 1,
    input_fingers: int = 2,
    gain_fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "low_noise_amp",
        [
            DeviceSpec("input", "nmos", {"width": input_width, "length": input_length, "fingers": input_fingers, "multipliers": input_multiplier}),
            DeviceSpec("gain", "nmos", {"width": gain_width, "length": gain_length, "fingers": gain_fingers, "multipliers": gain_multiplier}),
        ],
        [MoveSpec("gain", "below", "input")],
        [
            RouteSpec("input_source_W", "gain_drain_W"),
            RouteSpec("input_gate_E", "gain_source_E"),
        ],
    )


@tracked_generator("syntax_noise_x_diff_conv")
def noise_x_diff_conv(
    pdk: MappedPDK,
    width_m1: float = 3.0,
    width_m2: float = 3.0,
    length_m1: float = 0.5,
    length_m2: float = 0.5,
    m1_multiplier: int = 1,
    m2_multiplier: int = 1,
    m1_fingers: int = 2,
    m2_fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "noise_x_diff_conv",
        [
            DeviceSpec("m1", "nmos", {"width": width_m1, "length": length_m1, "fingers": m1_fingers, "multipliers": m1_multiplier}),
            DeviceSpec("m2", "nmos", {"width": width_m2, "length": length_m2, "fingers": m2_fingers, "multipliers": m2_multiplier}),
        ],
        [MoveSpec("m1", "below", "m2")],
        [RouteSpec("m1_gate_E", "m2_source_E")],
    )


@tracked_generator("syntax_pmos_array_2x5")
def pmos_array_2x5(
    pdk: MappedPDK,
    length: float = 0.5,
    fingers: int = 2,
):
    devices = [
        DeviceSpec(f"element{index}", "pmos", {"length": length, "fingers": fingers})
        for index in range(1, 11)
    ]
    moves = [
        MoveSpec("element2", "right", "element1"),
        MoveSpec("element3", "right", "element2"),
        MoveSpec("element4", "right", "element3"),
        MoveSpec("element5", "right", "element4"),
        MoveSpec("element6", "above", "element1"),
        MoveSpec("element7", "above", "element1"),
        MoveSpec("element8", "above", "element1"),
        MoveSpec("element9", "above", "element1"),
        MoveSpec("element10", "above", "element1"),
        MoveSpec("element7", "right", "element6"),
        MoveSpec("element8", "right", "element7"),
        MoveSpec("element9", "right", "element8"),
        MoveSpec("element10", "right", "element9"),
    ]
    return build_netlisted_layout(pdk, "pmos_array_2x5", devices, moves, [])


@tracked_generator("syntax_pmos_array_4x3")
def pmos_array_4x3(
    pdk: MappedPDK,
    width: float = 3.0,
    length: float = 0.5,
    multipliers: int = 1,
    fingers: int = 2,
):
    devices = [
        DeviceSpec(f"element{index}", "pmos", {"width": width, "length": length, "fingers": fingers, "multipliers": multipliers})
        for index in range(1, 13)
    ]
    moves = [
        MoveSpec("element2", "right", "element1"),
        MoveSpec("element3", "right", "element2"),
        MoveSpec("element4", "above", "element1"),
        MoveSpec("element5", "above", "element1"),
        MoveSpec("element6", "above", "element1"),
        MoveSpec("element5", "right", "element4"),
        MoveSpec("element6", "right", "element5"),
        MoveSpec("element7", "above", "element4"),
        MoveSpec("element8", "above", "element4"),
        MoveSpec("element9", "above", "element4"),
        MoveSpec("element8", "right", "element7"),
        MoveSpec("element9", "right", "element8"),
        MoveSpec("element10", "above", "element7"),
        MoveSpec("element11", "above", "element7"),
        MoveSpec("element12", "above", "element7"),
        MoveSpec("element11", "right", "element10"),
        MoveSpec("element12", "right", "element11"),
    ]
    return build_netlisted_layout(pdk, "pmos_array_4x3", devices, moves, [])


@tracked_generator("syntax_ulpd")
def ulpd(
    pdk: MappedPDK,
    forward_width: float = 3.0,
    leakred_width: float = 3.0,
    forward_length: float = 0.5,
    leakred_length: float = 0.5,
    forward_multiplier: int = 1,
    leakred_multiplier: int = 1,
    forward_fingers: int = 2,
    leakred_fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "ulpd",
        [
            DeviceSpec("forward", "nmos", {"width": forward_width, "length": forward_length, "fingers": forward_fingers, "multipliers": forward_multiplier}),
            DeviceSpec("leakred", "nmos", {"width": leakred_width, "length": leakred_length, "fingers": leakred_fingers, "multipliers": leakred_multiplier}),
        ],
        [MoveSpec("forward", "above", "leakred")],
        [
            RouteSpec("forward_gate_E", "leakred_source_E"),
            RouteSpec("forward_source_W", "leakred_gate_W"),
        ],
    )


@tracked_generator("syntax_varactor")
def varactor(
    pdk: MappedPDK,
    control_width: float = 3.0,
    accumulation_width: float = 3.0,
    control_length: float = 0.5,
    accumulation_length: float = 0.5,
    control_multiplier: int = 1,
    accumulation_multiplier: int = 1,
    control_fingers: int = 1,
    accumulation_fingers: int = 1,
):
    return build_netlisted_layout(
        pdk,
        "varactor",
        [
            DeviceSpec("control", "nmos", {"width": control_width, "length": control_length, "fingers": control_fingers, "multipliers": control_multiplier}),
            DeviceSpec("accumulation", "nmos", {"width": accumulation_width, "length": accumulation_length, "fingers": accumulation_fingers, "multipliers": accumulation_multiplier}),
        ],
        [MoveSpec("accumulation", "right", "control")],
        [
            RouteSpec("control_drain_E", "control_source_E"),
            RouteSpec("accumulation_drain_W", "accumulation_source_W"),
            RouteSpec("control_drain_E", "accumulation_drain_W", kind="straight_route"),
        ],
    )


@tracked_generator("syntax_cascode_common_gate_common_centroid")
def cascode_common_gate_common_centroid(
    pdk: MappedPDK,
    width: float = 3.0,
    length: float = 0.5,
    fingers: int = 1,
):
    return build_netlisted_layout(
        pdk,
        "cascode_common_gate_common_centroid",
        [
            DeviceSpec("ccg_A", "nmos", {"width": width, "length": length, "fingers": fingers, "with_dummy": False, "with_tie": False}),
            DeviceSpec("ccg_B", "nmos", {"width": width, "length": length, "fingers": fingers, "with_dummy": False, "with_tie": False}),
        ],
        [MoveSpec("ccg_B", "right", "ccg_A")],
        [
            RouteSpec("ccg_A_source_E", "ccg_B_drain_E", kind="highway_route", params={"glayer": "met4", "track_y": 5.0}),
            RouteSpec("ccg_A_drain_E", "ccg_A_source_E"),
        ],
    )


@tracked_generator("syntax_cascode_common_source_interdigitated")
def cascode_common_source_interdigitated(
    pdk: MappedPDK,
    width: float = 3.0,
    length: float = 0.5,
    fingers: int = 1,
):
    return build_netlisted_layout(
        pdk,
        "cascode_common_source_interdigitated",
        [
            DeviceSpec("CascodeCommonSource_A", "nmos", {"width": width, "length": length, "fingers": fingers, "with_dummy": False, "with_tie": False}),
            DeviceSpec("CascodeCommonSource_B", "nmos", {"width": width, "length": length, "fingers": fingers, "with_dummy": False, "with_tie": False}),
        ],
        [MoveSpec("CascodeCommonSource_B", "right", "CascodeCommonSource_A")],
        [
            RouteSpec("CascodeCommonSource_A_drain_E", "CascodeCommonSource_B_source_E", kind="highway_route", params={"glayer": "met4", "track_y": 5.0}),
            RouteSpec("CascodeCommonSource_A_source_E", "CascodeCommonSource_A_drain_E"),
        ],
    )


@tracked_generator("syntax_class_b_push_pull_interdigitated")
def class_b_push_pull_interdigitated(
    pdk: MappedPDK,
    width: float = 3.0,
    length: float = 0.5,
    fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "class_b_push_pull_interdigitated",
        [
            DeviceSpec("ClassBPushPull_A", "nmos", {"width": width, "length": length, "fingers": fingers}),
            DeviceSpec("ClassBPushPull_B", "nmos", {"width": width, "length": length, "fingers": fingers}),
        ],
        [MoveSpec("ClassBPushPull_B", "right", "ClassBPushPull_A")],
        [
            RouteSpec("ClassBPushPull_A_source_W", "ClassBPushPull_B_source_W"),
            RouteSpec("ClassBPushPull_A_gate_E", "ClassBPushPull_B_gate_E"),
        ],
    )


@tracked_generator("syntax_current_mirror_ntype_interdigitated")
def current_mirror_ntype_interdigitated(
    pdk: MappedPDK,
    width: float = 3.0,
    length: float = 0.5,
    fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "current_mirror_ntype_interdigitated",
        [
            DeviceSpec("cm_A", "nmos", {"width": width, "length": length, "fingers": fingers}),
            DeviceSpec("cm_B", "nmos", {"width": width, "length": length, "fingers": fingers}),
        ],
        [MoveSpec("cm_B", "right", "cm_A")],
        [
            RouteSpec("cm_A_gate_E", "cm_B_gate_E"),
            RouteSpec("cm_A_drain_E", "cm_A_gate_E"),
            RouteSpec("cm_A_source_E", "cm_B_source_E"),
        ],
    )


@tracked_generator("syntax_current_mirror_ptype_interdigitated")
def current_mirror_ptype_interdigitated(
    pdk: MappedPDK,
    width: float = 3.0,
    length: float = 0.5,
    fingers: int = 2,
):
    return build_netlisted_layout(
        pdk,
        "current_mirror_ptype_interdigitated",
        [
            DeviceSpec("cm_A", "pmos", {"width": width, "length": length, "fingers": fingers}),
            DeviceSpec("cm_B", "pmos", {"width": width, "length": length, "fingers": fingers}),
        ],
        [MoveSpec("cm_B", "right", "cm_A")],
        [
            RouteSpec("cm_A_gate_E", "cm_B_gate_E"),
            RouteSpec("cm_A_drain_E", "cm_A_gate_E"),
            RouteSpec("cm_A_source_E", "cm_B_source_E"),
        ],
    )


@tracked_generator("syntax_diff_pair")
def diff_pair_sample(
    pdk: MappedPDK,
    vin1_width: float = 3.0,
    vin2_width: float = 3.0,
    vin1_length: float = 0.5,
    vin2_length: float = 0.5,
    vin1_multiplier: int = 1,
    vin2_multiplier: int = 1,
    vin1_fingers: int = 1,
    vin2_fingers: int = 1,
):
    return build_netlisted_layout(
        pdk,
        "diff_pair",
        [
            DeviceSpec("vin1", "nmos", {"width": vin1_width, "length": vin1_length, "fingers": vin1_fingers, "multipliers": vin1_multiplier}),
            DeviceSpec("vin2", "nmos", {"width": vin2_width, "length": vin2_length, "fingers": vin2_fingers, "multipliers": vin2_multiplier}),
        ],
        [MoveSpec("vin2", "right", "vin1")],
        [RouteSpec("vin1_source_E", "vin2_source_W")],
    )


@tracked_generator("syntax_mimcap_array")
def mimcap_array_sample(
    pdk: MappedPDK,
    mimcap_size_x: float = 1.0,
    mimcap_size_y: float = 1.0,
):
    devices = [
        DeviceSpec(f"element{index}", "mimcap", {"size": (mimcap_size_x, mimcap_size_y)})
        for index in range(1, 7)
    ]
    moves = [
        MoveSpec("element2", "right", "element1"),
        MoveSpec("element3", "right", "element2"),
        MoveSpec("element4", "above", "element1"),
        MoveSpec("element5", "above", "element1"),
        MoveSpec("element6", "above", "element1"),
        MoveSpec("element5", "right", "element4"),
        MoveSpec("element6", "right", "element5"),
    ]
    routes = [
        RouteSpec("element1_top_met_E", "element2_top_met_W"),
        RouteSpec("element2_top_met_E", "element3_top_met_W"),
        RouteSpec("element4_top_met_E", "element5_top_met_W"),
        RouteSpec("element5_top_met_E", "element6_top_met_W"),
        RouteSpec("element1_top_met_N", "element4_top_met_S"),
        RouteSpec("element2_top_met_N", "element5_top_met_S"),
        RouteSpec("element3_top_met_N", "element6_top_met_S"),
        RouteSpec("element1_bottom_met_E", "element2_bottom_met_W"),
        RouteSpec("element2_bottom_met_E", "element3_bottom_met_W"),
        RouteSpec("element4_bottom_met_E", "element5_bottom_met_W"),
        RouteSpec("element5_bottom_met_E", "element6_bottom_met_W"),
        RouteSpec("element1_bottom_met_N", "element4_bottom_met_S"),
        RouteSpec("element2_bottom_met_N", "element5_bottom_met_S"),
        RouteSpec("element3_bottom_met_N", "element6_bottom_met_S"),
    ]
    return build_netlisted_layout(pdk, "mimcap_array", devices, moves, routes)


@tracked_generator("syntax_cross_coupled_inverters")
def cross_coupled_inverters(
    pdk: MappedPDK,
    nfet_width: float = 3.0,
    pfet_width: float = 3.0,
    ccinvs_length: float = 0.5,
    ccinvs_fingers: int = 1,
):
    return build_netlisted_layout(
        pdk,
        "cross_coupled_inverters",
        [
            DeviceSpec("ccinvs_top_A", "pmos", {"width": pfet_width, "length": ccinvs_length, "fingers": ccinvs_fingers, "with_dummy": False, "with_tie": False}),
            DeviceSpec("ccinvs_top_B", "pmos", {"width": pfet_width, "length": ccinvs_length, "fingers": ccinvs_fingers, "with_dummy": False, "with_tie": False}),
            DeviceSpec("ccinvs_bottom_A", "nmos", {"width": nfet_width, "length": ccinvs_length, "fingers": ccinvs_fingers, "with_dummy": False, "with_tie": False}),
            DeviceSpec("ccinvs_bottom_B", "nmos", {"width": nfet_width, "length": ccinvs_length, "fingers": ccinvs_fingers, "with_dummy": False, "with_tie": False}),
        ],
        [
            MoveSpec("ccinvs_top_B", "right", "ccinvs_top_A"),
            MoveSpec("ccinvs_bottom_A", "below", "ccinvs_top_A"),
            MoveSpec("ccinvs_bottom_B", "right", "ccinvs_bottom_A"),
        ],
        [],
    )


@tracked_generator("syntax_class_ab_stage")
def class_ab_stage(
    pdk: MappedPDK,
    source_numcols: int = 1,
    gm31_numcols: int = 1,
    source_length: float = 0.5,
    gm31_length: float = 0.5,
    source_width: float = 3.0,
    gm31_width: float = 3.0,
):
    devices = [
        DeviceSpec("gm31_A", "nmos", {"width": gm31_width, "length": gm31_length, "fingers": gm31_numcols, "with_dummy": False, "with_tie": False}),
        DeviceSpec("gm31_B", "nmos", {"width": gm31_width, "length": gm31_length, "fingers": gm31_numcols, "with_dummy": False, "with_tie": False}),
        DeviceSpec("source_A", "pmos", {"width": source_width, "length": source_length, "fingers": source_numcols, "with_dummy": False, "with_tie": False}),
        DeviceSpec("source_B", "pmos", {"width": source_width, "length": source_length, "fingers": source_numcols, "with_dummy": False, "with_tie": False}),
        DeviceSpec("sink_A", "nmos", {"fingers": 2, "with_dummy": False, "with_tie": False}),
        DeviceSpec("sink_B", "nmos", {"fingers": 2, "with_dummy": False, "with_tie": False}),
        DeviceSpec("pp1_c3", "mimcap", {"size": (1.0, 1.0)}),
        DeviceSpec("pp2_c3", "mimcap", {"size": (1.0, 1.0)}),
    ]
    moves = [
        MoveSpec("gm31_B", "right", "gm31_A"),
        MoveSpec("source_A", "above", "gm31_A"),
        MoveSpec("source_B", "right", "source_A"),
        MoveSpec("sink_A", "below", "gm31_A"),
        MoveSpec("sink_B", "right", "sink_A"),
        MoveSpec("pp1_c3", "right", "gm31_B"),
        MoveSpec("pp2_c3", "right", "pp1_c3"),
    ]
    routes: list[RouteSpec] = []
    return build_netlisted_layout(pdk, "class_ab_stage", devices, moves, routes)


@tracked_generator("syntax_four_stage_integrator")
def four_stage_integrator(pdk: MappedPDK):
    devices = [
        DeviceSpec("finteg_gm1_A", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("finteg_gm1_B", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("finteg_sink_cm_A", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("finteg_sink_cm_B", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("sinteg_gm2_A", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("sinteg_gm2_B", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("sinteg_gm4_A", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("sinteg_gm4_B", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("sinteg_sink_cm_A", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("sinteg_sink_cm_B", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("tinteg_gm32_A", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("tinteg_gm32_B", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("tinteg_gm5_A", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("tinteg_gm5_B", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("abstage_gm31_A", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("abstage_gm31_B", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("abstage_pp1_n1", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
        DeviceSpec("abstage_pp2_n1", "nmos", {"fingers": 1, "with_dummy": False, "with_tie": False}),
    ]
    moves = [
        MoveSpec("finteg_gm1_B", "below", "finteg_gm1_A"),
        MoveSpec("finteg_sink_cm_A", "below", "finteg_gm1_B"),
        MoveSpec("finteg_sink_cm_B", "below", "finteg_sink_cm_A"),
        MoveSpec("sinteg_gm2_A", "right", "finteg_gm1_A"),
        MoveSpec("sinteg_gm2_B", "below", "sinteg_gm2_A"),
        MoveSpec("sinteg_gm4_A", "below", "sinteg_gm2_B"),
        MoveSpec("sinteg_gm4_B", "below", "sinteg_gm4_A"),
        MoveSpec("sinteg_sink_cm_A", "below", "sinteg_gm4_B"),
        MoveSpec("sinteg_sink_cm_B", "below", "sinteg_sink_cm_A"),
        MoveSpec("tinteg_gm32_A", "right", "sinteg_gm2_A"),
        MoveSpec("tinteg_gm32_B", "below", "tinteg_gm32_A"),
        MoveSpec("tinteg_gm5_A", "below", "tinteg_gm32_B"),
        MoveSpec("tinteg_gm5_B", "below", "tinteg_gm5_A"),
        MoveSpec("abstage_gm31_A", "right", "tinteg_gm32_A"),
        MoveSpec("abstage_gm31_B", "below", "abstage_gm31_A"),
        MoveSpec("abstage_pp1_n1", "below", "abstage_gm31_B"),
        MoveSpec("abstage_pp2_n1", "below", "abstage_pp1_n1"),
    ]
    routes: list[RouteSpec] = []
    return build_netlisted_layout(pdk, "four_stage_integrator", devices, moves, routes)


SAMPLE_BUILDERS = {
    "CTATVGen": ctat_vgen,
    "CascodeCommonGate": cascode_common_gate,
    "CascodeCommonGateCommonCentroid": cascode_common_gate_common_centroid,
    "CascodeCommonSourceInterdigitated": cascode_common_source_interdigitated,
    "ClassABStage": class_ab_stage,
    "ClassBPushPull": class_b_push_pull,
    "ClassBPushPullInterdigitated": class_b_push_pull_interdigitated,
    "CommonSourceAmplifier": common_source_amplifier,
    "CommonSourceAmplifierWDiodeLoad": common_source_amplifier_w_diode_load,
    "CrossCoupledInverters": cross_coupled_inverters,
    "CurrentMirrorNtypeInterdigitated": current_mirror_ntype_interdigitated,
    "CurrentMirrorPtype": current_mirror_ptype,
    "CurrentMirrorPtypeInterdigitated": current_mirror_ptype_interdigitated,
    "DiffPair": diff_pair_sample,
    "FourStageIntegrator": four_stage_integrator,
    "Inverter": inverter,
    "LowNoiseAmp": low_noise_amp,
    "MimcapArray": mimcap_array_sample,
    "NoiseXDiffConv": noise_x_diff_conv,
    "PMOSArray2x5": pmos_array_2x5,
    "PMOSArray4x3": pmos_array_4x3,
    "ULPD": ulpd,
    "Varactor": varactor,
}
