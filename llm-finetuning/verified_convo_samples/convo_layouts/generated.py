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


SAMPLE_BUILDERS = {
    "CTATVGen": ctat_vgen,
    "CascodeCommonGate": cascode_common_gate,
    "ClassBPushPull": class_b_push_pull,
    "CommonSourceAmplifier": common_source_amplifier,
    "CommonSourceAmplifierWDiodeLoad": common_source_amplifier_w_diode_load,
    "CurrentMirrorPtype": current_mirror_ptype,
    "Inverter": inverter,
    "LowNoiseAmp": low_noise_amp,
    "NoiseXDiffConv": noise_x_diff_conv,
    "PMOSArray2x5": pmos_array_2x5,
    "PMOSArray4x3": pmos_array_4x3,
    "ULPD": ulpd,
    "Varactor": varactor,
}
