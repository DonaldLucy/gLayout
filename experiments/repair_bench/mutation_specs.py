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
DIFF_PAIR = "src/glayout/cells/elementary/diff_pair/diff_pair.py"
TRANSMISSION_GATE = "src/glayout/cells/elementary/transmission_gate/transmission_gate.py"
FVF = "src/glayout/cells/elementary/FVF/fvf.py"
LVCM = "src/glayout/cells/composite/low_voltage_cmirror/low_voltage_cmirror.py"
FVF_LVCM = "src/glayout/cells/composite/fvf_based_ota/low_voltage_cmirror.py"
DIFF_PAIR_IBIAS = "src/glayout/cells/composite/diffpair_cmirror_bias/diff_pair_cmirrorbias.py"
DIFF_PAIR_IBIAS_CANDIDATE = "scripts/run_diff_pair_ibias_labeled_candidate.py"


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
        MutationSpec(
            mutation_id=f"{prefix}_label_moved_vout_to_vref",
            case_id=case_id,
            operator="label_moved_to_wrong_port",
            file_path=CURRENT_MIRROR,
            clean_text='move_info.append((vcopylabel,cm_in.ports["fet_B_drain_N"],None))',
            buggy_text='move_info.append((vcopylabel,cm_in.ports["fet_A_drain_N"],None))',
            description="Move the VOUT label onto the reference-drain conductor while keeping the label text unchanged.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_physical_route_removed_diode",
            case_id=case_id,
            operator="physical_route_removed",
            file_path=CURRENT_MIRROR,
            clean_text="    interdigitized_fets << L_route(pdk, interdigitized_fets.ports['A_drain_W'], gate_short.ports['con_N'], viaoffset=False, fullbottom=False)",
            buggy_text="    # MUTATION: removed physical diode-connection route between reference drain and mirror gates",
            description="Remove the physical diode-connection route while leaving the schematic current mirror netlist unchanged.",
        ),
    ]


def _diff_pair_specs(case_id: str, prefix: str) -> list[MutationSpec]:
    return [
        MutationSpec(
            mutation_id=f"{prefix}_label_text_vtail",
            case_id=case_id,
            operator="label_text_typo",
            file_path=DIFF_PAIR,
            clean_text='vtaillabel.add_label(text="VTAIL",layer=pdk.get_glayer("met2_label"))',
            buggy_text='vtaillabel.add_label(text="VTAIL_BAD",layer=pdk.get_glayer("met2_label"))',
            description="Rename the differential-pair tail label so LVS sees the wrong top pin.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_label_layer_vp",
            case_id=case_id,
            operator="label_layer_wrong",
            file_path=DIFF_PAIR,
            clean_text='vplabel.add_label(text="VP",layer=pdk.get_glayer("met2_label"))',
            buggy_text='vplabel.add_label(text="VP",layer=pdk.get_glayer("met1_label"))',
            description="Move the VP input label to the wrong label layer.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_netlist_pin_left_gate",
            case_id=case_id,
            operator="netlist_pin_swap",
            file_path=DIFF_PAIR,
            clean_text="[('D', 'VDD1'), ('G', 'VP'), ('S', 'VTAIL'), ('B', 'B')]",
            buggy_text="[('D', 'VDD1'), ('G', 'VN'), ('S', 'VTAIL'), ('B', 'B')]",
            description="Connect the left differential-pair gate to VN instead of VP in the schematic.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_top_node_vtail",
            case_id=case_id,
            operator="top_node_rename",
            file_path=DIFF_PAIR,
            clean_text="diff_pair_netlist = Netlist(circuit_name='DIFF_PAIR', nodes=['VP', 'VN', 'VDD1', 'VDD2', 'VTAIL', 'B'])",
            buggy_text="diff_pair_netlist = Netlist(circuit_name='DIFF_PAIR', nodes=['VP', 'VN', 'VDD1', 'VDD2', 'VTAIL_BAD', 'B'])",
            description="Rename the differential-pair tail node in the schematic.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_label_moved_vp_to_vn",
            case_id=case_id,
            operator="label_moved_to_wrong_port",
            file_path=DIFF_PAIR,
            clean_text='move_info.append((vplabel,df_in.ports["br_multiplier_0_gate_S"], None))',
            buggy_text='move_info.append((vplabel,df_in.ports["bl_multiplier_0_gate_S"], None))',
            description="Move the VP label onto the VN gate conductor while keeping the label text unchanged.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_route_spacing_violation_gates",
            case_id=case_id,
            operator="route_spacing_violation",
            file_path=DIFF_PAIR,
            clean_text='\tplus_minus_seperation = max(pdk.get_grule("met2")["min_separation"], plus_minus_seperation)',
            buggy_text="\tplus_minus_seperation = 0.0  # MUTATION: force plus/minus gate bars too close",
            description="Force the differential input gate bars too close, creating a routed-metal DRC stress case.",
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


def _fvf_specs(case_id: str, prefix: str) -> list[MutationSpec]:
    return [
        MutationSpec(
            mutation_id=f"{prefix}_label_text_ib",
            case_id=case_id,
            operator="label_text_typo",
            file_path=FVF,
            clean_text='ibiaslabel.add_label(text="Ib",layer=pdk.get_glayer("met2_label"))',
            buggy_text='ibiaslabel.add_label(text="Ib_BAD",layer=pdk.get_glayer("met2_label"))',
            description="Rename the flipped-voltage-follower bias label so LVS sees the wrong pin.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_label_layer_vin",
            case_id=case_id,
            operator="label_layer_wrong",
            file_path=FVF,
            clean_text='inputlabel.add_label(text="VIN",layer=pdk.get_glayer("met1_label"))',
            buggy_text='inputlabel.add_label(text="VIN",layer=pdk.get_glayer("met2_label"))',
            description="Move the FVF input label to the wrong label layer.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_netlist_feedback_gate",
            case_id=case_id,
            operator="netlist_pin_swap",
            file_path=FVF,
            clean_text="netlist.connect_netlist(fet_2_netlist, [('D', 'VOUT'), ('G', 'Ib'), ('S', 'VBULK'), ('B', 'VBULK')])",
            buggy_text="netlist.connect_netlist(fet_2_netlist, [('D', 'VOUT'), ('G', 'VIN'), ('S', 'VBULK'), ('B', 'VBULK')])",
            description="Connect the feedback transistor gate to VIN instead of Ib in the schematic.",
        ),
        MutationSpec(
            mutation_id=f"{prefix}_top_node_ib",
            case_id=case_id,
            operator="top_node_rename",
            file_path=FVF,
            clean_text="netlist = Netlist(circuit_name='FLIPPED_VOLTAGE_FOLLOWER', nodes=['VIN', 'VBULK', 'VOUT', 'Ib'])",
            buggy_text="netlist = Netlist(circuit_name='FLIPPED_VOLTAGE_FOLLOWER', nodes=['VIN', 'VBULK', 'VOUT', 'Ib_BAD'])",
            description="Rename the FVF bias node in the schematic.",
        ),
    ]


MUTATION_SPECS: list[MutationSpec] = [
    *_diff_pair_specs("diff_pair_default", "dpn"),
    *_diff_pair_specs("diff_pair_pmos", "dpp"),
    *_diff_pair_specs("diff_pair_generic", "dpg"),
    *_cmirror_specs("current_mirror_nfet", "cmn"),
    *_cmirror_specs("current_mirror_pfet", "cmp"),
    *_fvf_specs("flipped_voltage_follower", "fvf"),
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
    MutationSpec(
        mutation_id="tg_label_moved_vout_to_vin",
        case_id="transmission_gate",
        operator="label_moved_to_wrong_port",
        file_path=TRANSMISSION_GATE,
        clean_text='move_info.append((voutlabel,tg_in.ports["P_multiplier_0_drain_W"],None))',
        buggy_text='move_info.append((voutlabel,tg_in.ports["N_multiplier_0_source_E"],None))',
        description="Move the VOUT label onto the VIN conductor while keeping the label text unchanged.",
    ),
    MutationSpec(
        mutation_id="tg_physical_route_removed_vin",
        case_id="transmission_gate",
        operator="physical_route_removed",
        file_path=TRANSMISSION_GATE,
        clean_text='    top_level << c_route(pdk, nfet_ref.ports["multiplier_0_source_E"], pfet_ref.ports["multiplier_0_source_E"])',
        buggy_text="    # MUTATION: removed physical VIN source short between NFET and PFET",
        description="Remove the physical VIN route between the NFET and PFET while leaving the schematic transmission-gate netlist unchanged.",
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
    MutationSpec(
        mutation_id="dpil_label_text_ibias",
        case_id="diff_pair_ibias_labeled_candidate",
        operator="label_text_typo",
        file_path=DIFF_PAIR_IBIAS_CANDIDATE,
        clean_text='"IBIAS": ("ibias_A_drain_E", None, 0.50),',
        buggy_text='"IBIAS_BAD": ("ibias_A_drain_E", None, 0.50),',
        description="Rename the labeled candidate IBIAS pin in the source label map.",
    ),
    MutationSpec(
        mutation_id="dpil_missing_tail_subnet",
        case_id="diff_pair_ibias_labeled_candidate",
        operator="missing_connect_subnet",
        file_path=DIFF_PAIR_IBIAS,
        clean_text="""    netlist.connect_subnets(
        cmirror_ref,
        diffpair_ref,
        [('VOUT', 'VTAIL')]
    )""",
        buggy_text="    # MUTATION: removed mirror output to differential-pair tail connection",
        description="Remove the candidate's source schematic tail-current connection.",
    ),
    MutationSpec(
        mutation_id="dpil_top_node_ibias",
        case_id="diff_pair_ibias_labeled_candidate",
        operator="top_node_rename",
        file_path=DIFF_PAIR_IBIAS,
        clean_text="nodes=['VP', 'VN', 'VDD1', 'VDD2', 'IBIAS', 'VSS', 'B']",
        buggy_text="nodes=['VP', 'VN', 'VDD1', 'VDD2', 'IBIAS_BAD', 'VSS', 'B']",
        description="Rename the candidate's top-level IBIAS schematic pin.",
    ),
    MutationSpec(
        mutation_id="dpil_netlist_bias_pin",
        case_id="diff_pair_ibias_labeled_candidate",
        operator="netlist_pin_swap",
        file_path=DIFF_PAIR_IBIAS,
        clean_text="[('VREF', 'IBIAS'), ('B', 'VSS')]",
        buggy_text="[('VREF', 'VSS'), ('B', 'VSS')]",
        description="Connect the candidate current mirror reference to VSS instead of IBIAS.",
    ),
]


CASE_PROFILES = {
    "minimal": [
        "current_mirror_nfet",
        "current_mirror_pfet",
        "transmission_gate",
    ],
    "conservative": [
        "diff_pair_default",
        "current_mirror_nfet",
        "current_mirror_pfet",
        "transmission_gate",
        "diff_pair_ibias_labeled_candidate",
    ],
    # Historical candidates from the 9/19 -> 10 validated-cell discussion.
    # The bench still validates them on the current branch/machine before use.
    "validated10": [
        "diff_pair_default",
        "diff_pair_pmos",
        "current_mirror_nfet",
        "current_mirror_pfet",
        "transmission_gate",
        "low_voltage_cmirror",
        "fvf_based_ota_low_voltage_cmirror",
        "diff_pair_ibias_labeled_candidate",
        "flipped_voltage_follower",
        "diff_pair_generic",
    ],
    # Strict-clean subset observed on the current SKY130 regression setup.
    # Use this for faster sharded data generation after a full validated10 smoke run.
    "validated6": [
        "diff_pair_default",
        "diff_pair_pmos",
        "current_mirror_nfet",
        "current_mirror_pfet",
        "transmission_gate",
        "diff_pair_ibias_labeled_candidate",
    ],
}


DEFAULT_STRICT_CLEAN_CASES = CASE_PROFILES["conservative"]
