from __future__ import annotations

from collections import Counter
from pathlib import Path

from experiments.repair_bench.mutation_specs import CASE_PROFILES, MUTATION_SPECS


def test_validated6_mutation_specs_are_unique_and_source_backed():
    repo_root = Path(__file__).resolve().parents[1]
    validated6 = set(CASE_PROFILES["validated6"])
    specs = [spec for spec in MUTATION_SPECS if spec.case_id in validated6]

    assert len(specs) >= 150
    assert not [key for key, count in Counter((spec.case_id, spec.mutation_id) for spec in specs).items() if count > 1]

    missing = []
    for spec in specs:
        source = (repo_root / spec.file_path).read_text()
        if spec.clean_text not in source:
            missing.append((spec.case_id, spec.mutation_id, spec.file_path))

    assert missing == []


def test_validated6_mutation_specs_cover_repair_agent_fault_families():
    validated6 = set(CASE_PROFILES["validated6"])
    operators = {
        spec.operator
        for spec in MUTATION_SPECS
        if spec.case_id in validated6
    }

    assert {
        "label_text_typo",
        "label_layer_wrong",
        "label_moved_to_wrong_port",
        "netlist_pin_swap",
        "missing_connect_subnet",
        "physical_route_removed",
        "placement_spacing_violation",
        "route_spacing_violation",
        "top_node_rename",
    } <= operators
