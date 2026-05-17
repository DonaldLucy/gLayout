from __future__ import annotations

import json

from experiments.repair_bench.run_zero_shot_baseline import (
    aggregate_results,
    compact_repair_packet,
    is_strict_clean,
    normalize_action_file,
    select_records,
)


def test_select_records_unique_mutations_keeps_first_replica():
    records = [
        {"case_id": "a", "sample_id": "0", "mutation": {"mutation_id": "m0"}},
        {"case_id": "a", "sample_id": "1", "mutation": {"mutation_id": "m0"}},
        {"case_id": "a", "sample_id": "2", "mutation": {"mutation_id": "m1"}},
        {"case_id": "b", "sample_id": "3", "mutation": {"mutation_id": "m0"}},
    ]

    selected = select_records(records, unique_mutations=True, limit=0)

    assert [record["sample_id"] for record in selected] == ["0", "2", "3"]


def test_normalize_action_file_accepts_workspace_absolute_source_path(tmp_path):
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "glayout" / "cell.py"
    source.parent.mkdir(parents=True)
    source.write_text("pass\n")

    normalized = normalize_action_file(
        workspace,
        "/tmp/some_run/workspace/src/glayout/cell.py",
    )

    assert normalized == source.resolve()


def test_normalize_action_file_rejects_path_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert normalize_action_file(workspace, "../outside.py") is None


def test_zero_shot_aggregate_counts_verification_clean():
    results = [
        {"parse_success": True, "apply_success": True, "verification_strict_clean": True, "verification": {}},
        {"parse_success": True, "apply_success": False},
    ]

    aggregate = aggregate_results(results)

    assert aggregate["total"] == 2
    assert aggregate["parse_success"] == 2
    assert aggregate["apply_success"] == 1
    assert aggregate["verification_runs"] == 1
    assert aggregate["verification_strict_clean"] == 1


def test_is_strict_clean_requires_all_drc_lvs_sections():
    clean = {
        "baseline_drc": {"is_clean": True},
        "traced_drc": {"is_clean": True},
        "baseline_lvs": {"is_clean": True},
        "traced_lvs": {"is_clean": True},
    }
    dirty = {**clean, "traced_lvs": {"is_clean": False}}

    assert is_strict_clean(clean) is True
    assert is_strict_clean(dirty) is False


def test_compact_repair_packet_prioritizes_source_evidence(tmp_path):
    packet = {
        "purpose": "test packet",
        "status": "top_level_pin_mismatch",
        "matched": False,
        "netlists_matched": True,
        "issue_count": 1,
        "drc_status": "clean",
        "drc_issue_count": 0,
        "primary_hint_types": ["possible_top_label_text_mismatch"],
        "repair_hints": [
            {
                "type": "possible_top_label_text_mismatch",
                "confidence": "high",
                "message": "Inspect label literals.",
            }
        ],
        "source_label_candidates": [
            {
                "type": "source_label_text_candidate",
                "label": "VOUT_BAD",
                "matched_terms": ["VOUT"],
                "file": "/tmp/run/workspace/src/glayout/cells/demo.py",
                "line": 42,
                "text": 'vcopylabel.add_label(text="VOUT_BAD",layer=pdk.get_glayer("met2_label"))',
            }
        ],
        "source_spans": [
            {
                "location_kind": "source_label_text_candidate",
                "file": "/tmp/run/workspace/src/glayout/cells/demo.py",
                "focus_line": 42,
                "text": "\n".join(f"{line_no:04d}: line_{line_no}" for line_no in range(1, 90)),
            }
        ],
        "component_port_manifest": [
            {
                "call_id": "call_000001",
                "generator_id": "demo",
                "ports": [{"name": f"port_{idx}"} for idx in range(1000)],
                "port_count_total": 1000,
            }
        ],
    }
    path = tmp_path / "repair_packet.json"
    path.write_text(json.dumps(packet))

    compact = compact_repair_packet(str(path), 220, prompt_style="compact")

    assert "source_label_candidates" in compact
    assert "src/glayout/cells/demo.py" in compact
    assert "VOUT_BAD" in compact
    assert "port_0" in compact
    assert "port_999" not in compact


def test_raw_repair_packet_preserves_first_json_lines(tmp_path):
    path = tmp_path / "repair_packet.json"
    path.write_text("{\n  \"first\": 1,\n  \"second\": 2\n}\n")

    raw = compact_repair_packet(str(path), 2, prompt_style="raw")

    assert raw == "{\n  \"first\": 1,"
