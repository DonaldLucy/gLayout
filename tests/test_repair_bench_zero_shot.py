from __future__ import annotations

from experiments.repair_bench.run_zero_shot_baseline import (
    aggregate_results,
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
