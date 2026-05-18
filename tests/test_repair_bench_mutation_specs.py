from __future__ import annotations

from collections import Counter
from pathlib import Path

from experiments.repair_bench.mutation_specs import CASE_PROFILES, INACTIVE_FAULT_MUTATIONS, MUTATION_SPECS
from experiments.repair_bench.run_repair_bench import make_sample_specs


def test_validated6_mutation_specs_are_unique_and_source_backed():
    repo_root = Path(__file__).resolve().parents[1]
    validated6 = set(CASE_PROFILES["validated6"])
    specs = [spec for spec in MUTATION_SPECS if spec.case_id in validated6]

    assert len(specs) >= 140
    assert not [key for key, count in Counter((spec.case_id, spec.mutation_id) for spec in specs).items() if count > 1]
    assert not [
        (spec.case_id, spec.mutation_id)
        for spec in specs
        if (spec.case_id, spec.mutation_id) in INACTIVE_FAULT_MUTATIONS
    ]

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
        "route_spacing_violation",
        "top_node_rename",
    } <= operators


def test_make_sample_specs_supports_round_robin_shards():
    specs = ["m0", "m1", "m2", "m3", "m4", "m5"]

    shard0 = make_sample_specs(specs, 3, sample_offset=0, sample_stride=2)
    shard1 = make_sample_specs(specs, 3, sample_offset=1, sample_stride=2)

    assert shard0 == [(0, "m0"), (2, "m2"), (4, "m4")]
    assert shard1 == [(1, "m1"), (3, "m3"), (5, "m5")]
