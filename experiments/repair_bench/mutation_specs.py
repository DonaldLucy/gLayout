from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MutationSpec:
    """A reversible source-level bug used to build supervised repair samples."""

    mutation_id: str
    case_id: str
    operator: str
    file_path: str
    clean_text: str
    buggy_text: str
    description: str


CURRENT_MIRROR = "src/glayout/cells/elementary/current_mirror/current_mirror.py"
TRANSMISSION_GATE = "src/glayout/cells/elementary/transmission_gate/transmission_gate.py"
LVCM = "src/glayout/cells/composite/low_voltage_cmirror/low_voltage_cmirror.py"
FVF_LVCM = "src/glayout/cells/composite/fvf_based_ota/low_voltage_cmirror.py"
DIFF_PAIR_IBIAS = "src/glayout/cells/composite/diffpair_cmirror_bias/diff_pair_cmirrorbias.py"


def _cmirror_specs(case_id: str, prefix: str) -> list[MutationSpec]:
    return [
        MutationSpec(
            mutation_id=f"{prefix}_label_text_vout",
            case_id=case_id,
            operator="label_text_typo",
            file_path=CURRENT_MIRROR,
            clean_text='vcopylabel.add_label(text="VOUT",layer=pdk.get_glayer("met2_label"))',
            buggy_text='vcopylabel.add_label(text="VOUT_BAD",layer=pdk.get_glayer("met2_label"))',
            description="Rename the top-level current mirror output label so LVS sees the wrong pin.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_label_layer_vref",
            case_id=case_id,
            operator="label_layer_wrong",
            file_path=CURRENT_MIRROR,
            clean_text='vreflabel.add_label(text="VREF",layer=pdk.get_glayer("met2_label"))',
            buggy_text='vreflabel.add_label(text="VREF",layer=pdk.get_glayer("met1_label"))',
            description="Move the VREF label to the wrong label layer.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_netlist_pin_vdd2",
            case_id=case_id,
            operator="netlist_pin_swap",
            file_path=CURRENT_MIRROR,
            clean_text='("VDD2", "VOUT"),   # mirror drain',
            buggy_text='("VDD2", "VREF"),   # MUTATION: mirror drain connected to reference net',
            description="Connect the mirror drain to VREF in the schematic netlist instead of VOUT.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_top_node_vout",
            case_id=case_id,
            operator="top_node_rename",
            file_path=CURRENT_MIRROR,
            clean_text='current_mirror_netlist = Netlist(circuit_name="CMIRROR", nodes=["VREF", "VOUT", "VSS", "B"])',
            buggy_text='current_mirror_netlist = Netlist(circuit_name="CMIRROR", nodes=["VREF", "VOUT_BAD", "VSS", "B"])',
            description="Rename the top-level schematic node VOUT.",
        ),
    ]


def _lvcm_specs(case_id: str, file_path: str, prefix: str, uses_get_component_netlist: bool) -> list[MutationSpec]:
    fet_1a_clean = (
        "fet_1A_ref=netlist.connect_netlist(get_component_netlist(fet_2_ref), "
        "[('D', 'IOUT1'),('G','IBIAS1'),('B','GND')])"
        if uses_get_component_netlist
        else "fet_1A_ref=netlist.connect_netlist(fet_2_ref.info['netlist'], "
        "[('D', 'IOUT1'),('G','IBIAS1'),('B','GND')])"
    )
    fet_1a_buggy = (
        "fet_1A_ref=netlist.connect_netlist(get_component_netlist(fet_2_ref), "
        "[('D', 'IBIAS1'),('G','IBIAS1'),('B','GND')])"
        if uses_get_component_netlist
        else "fet_1A_ref=netlist.connect_netlist(fet_2_ref.info['netlist'], "
        "[('D', 'IBIAS1'),('G','IBIAS1'),('B','GND')])"
    )
    subnet_clean = """        netlist.connect_subnets(
                fet_1A_ref,
                fet_1B_ref,
                [('S', 'D')]
                )"""
    subnet_buggy = "        # MUTATION: removed left-branch source/drain internal connection"
    return [
        MutationSpec(
            mutation_id=f"{prefix}_label_text_iout1",
            case_id=case_id,
            operator="label_text_typo",
            file_path=file_path,
            clean_text='output1label.add_label(text="IOUT1",layer=pdk.get_glayer("met2_label"))',
            buggy_text='output1label.add_label(text="IOUT1_BAD",layer=pdk.get_glayer("met2_label"))',
            description="Rename the first output label so LVS pin matching fails.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_label_layer_ibias1",
            case_id=case_id,
            operator="label_layer_wrong",
            file_path=file_path,
            clean_text='ibias1label.add_label(text="IBIAS1",layer=pdk.get_glayer("met3_label"))',
            buggy_text='ibias1label.add_label(text="IBIAS1",layer=pdk.get_glayer("met2_label"))',
            description="Move the IBIAS1 label to the wrong metal label layer.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_netlist_pin_iout1",
            case_id=case_id,
            operator="netlist_pin_swap",
            file_path=file_path,
            clean_text=fet_1a_clean,
            buggy_text=fet_1a_buggy,
            description="Connect the first output drain to IBIAS1 in the schematic netlist.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_missing_subnet_left",
            case_id=case_id,
            operator="missing_connect_subnet",
            file_path=file_path,
            clean_text=subnet_clean,
            buggy_text=subnet_buggy,
            description="Remove the source/drain subnet connection for the left output branch.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_top_node_iout2",
            case_id=case_id,
            operator="top_node_rename",
            file_path=file_path,
            clean_text="netlist = Netlist(circuit_name='Low_voltage_current_mirror', nodes=['IBIAS1', 'IBIAS2', 'GND', 'IOUT1', 'IOUT2'])",
            buggy_text="netlist = Netlist(circuit_name='Low_voltage_current_mirror', nodes=['IBIAS1', 'IBIAS2', 'GND', 'IOUT1', 'IOUT2_BAD'])",
            description="Rename the second output node in the schematic netlist.",
        ),
    ]


MUTATION_SPECS: list[MutationSpec] = [
    *_cmirror_specs("current_mirror_nfet", "cmn"),
    *_cmirror_specs("current_mirror_pfet", "cmp"),
    MutationSpec(
        mutation_id="tg_label_text_vin",
        case_id="transmission_gate",
        operator="label_text_typo",
        file_path=TRANSMISSION_GATE,
        clean_text='vinlabel.add_label(text="VIN",layer=pdk.get_glayer("met2_label"))',
        buggy_text='vinlabel.add_label(text="VIN_BAD",layer=pdk.get_glayer("met2_label"))',
        description="Rename the transmission-gate input label.",
    ),
    MutationSpec(
        mutation_id="tg_label_layer_vss",
        case_id="transmission_gate",
        operator="label_layer_wrong",
        file_path=TRANSMISSION_GATE,
        clean_text='vsslabel.add_label(text="VSS",layer=pdk.get_glayer("met2_label"))',
        buggy_text='vsslabel.add_label(text="VSS",layer=pdk.get_glayer("met1_label"))',
        description="Move the VSS label to the wrong label layer.",
    ),
    MutationSpec(
        mutation_id="tg_netlist_nfet_drain",
        case_id="transmission_gate",
        operator="netlist_pin_swap",
        file_path=TRANSMISSION_GATE,
        clean_text="netlist.connect_netlist(nfet_netlist, [('D', 'VOUT'), ('G', 'VGN'), ('S', 'VIN'), ('B', 'VSS')])",
        buggy_text="netlist.connect_netlist(nfet_netlist, [('D', 'VIN'), ('G', 'VGN'), ('S', 'VIN'), ('B', 'VSS')])",
        description="Short the NFET schematic drain to VIN instead of VOUT.",
    ),
    MutationSpec(
        mutation_id="tg_netlist_pfet_gate",
        case_id="transmission_gate",
        operator="netlist_pin_swap",
        file_path=TRANSMISSION_GATE,
        clean_text="netlist.connect_netlist(pfet_netlist, [('D', 'VOUT'), ('G', 'VGP'), ('S', 'VIN'), ('B', 'VCC')])",
        buggy_text="netlist.connect_netlist(pfet_netlist, [('D', 'VOUT'), ('G', 'VGN'), ('S', 'VIN'), ('B', 'VCC')])",
        description="Connect the PFET schematic gate to VGN instead of VGP.",
    ),
    MutationSpec(
        mutation_id="tg_top_node_vgn",
        case_id="transmission_gate",
        operator="top_node_rename",
        file_path=TRANSMISSION_GATE,
        clean_text="netlist = Netlist(circuit_name='Transmission_Gate', nodes=['VIN', 'VSS', 'VOUT', 'VCC', 'VGP', 'VGN'])",
        buggy_text="netlist = Netlist(circuit_name='Transmission_Gate', nodes=['VIN', 'VSS', 'VOUT', 'VCC', 'VGP', 'VGN_BAD'])",
        description="Rename the top-level VGN schematic pin.",
    ),
    *_lvcm_specs("low_voltage_cmirror", LVCM, "lvcm", uses_get_component_netlist=False),
    *_lvcm_specs(
        "fvf_based_ota_low_voltage_cmirror",
        FVF_LVCM,
        "fvf_lvcm",
        uses_get_component_netlist=True,
    ),
    MutationSpec(
        mutation_id="dpi_missing_tail_subnet",
        case_id="diff_pair_ibias",
        operator="missing_connect_subnet",
        file_path=DIFF_PAIR_IBIAS,
        clean_text="""    netlist.connect_subnets(
        cmirror_ref,
        diffpair_ref,
        [('VOUT', 'VTAIL')]
    )""",
        buggy_text="    # MUTATION: removed mirror output to differential-pair tail connection",
        description="Remove the current mirror output to differential-pair tail schematic connection.",
    ),
    MutationSpec(
        mutation_id="dpi_top_node_ibias",
        case_id="diff_pair_ibias",
        operator="top_node_rename",
        file_path=DIFF_PAIR_IBIAS,
        clean_text="nodes=['VP', 'VN', 'VDD1', 'VDD2', 'IBIAS', 'VSS', 'B']",
        buggy_text="nodes=['VP', 'VN', 'VDD1', 'VDD2', 'IBIAS_BAD', 'VSS', 'B']",
        description="Rename the top-level IBIAS schematic pin.",
    ),
    MutationSpec(
        mutation_id="dpi_netlist_bias_pin",
        case_id="diff_pair_ibias",
        operator="netlist_pin_swap",
        file_path=DIFF_PAIR_IBIAS,
        clean_text="[('VREF', 'IBIAS'), ('B', 'VSS')]",
        buggy_text="[('VREF', 'VSS'), ('B', 'VSS')]",
        description="Connect the current mirror reference to VSS instead of IBIAS.",
    ),
]


DEFAULT_STRICT_CLEAN_CASES = [
    "current_mirror_nfet",
    "current_mirror_pfet",
    "transmission_gate",
    "low_voltage_cmirror",
    "fvf_based_ota_low_voltage_cmirror",
]
