from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_runtime_module():
    runtime_path = Path(__file__).resolve().parents[1] / "src" / "glayout" / "provenance" / "runtime.py"
    spec = importlib.util.spec_from_file_location("glayout_smgr_runtime_test", runtime_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_snapshot_query_and_ranking():
    runtime = _load_runtime_module()
    snapshot = runtime.ProvenanceSnapshot(
        {
            "calls": {
                "call_000001": {
                    "call_id": "call_000001",
                    "generator_id": "diff_pair",
                    "callsite": {"file": "demo.py", "line": 10, "function": "build"},
                    "parent_call_id": None,
                },
                "call_000002": {
                    "call_id": "call_000002",
                    "generator_id": "c_route",
                    "callsite": {"file": "demo.py", "line": 20, "function": "build"},
                    "parent_call_id": "call_000001",
                },
            },
            "objects": {
                "comp_000001": {
                    "object_id": "comp_000001",
                    "object_type": "component",
                    "bbox": [0.0, 0.0, 20.0, 10.0],
                    "layer": None,
                    "generated_by": {"call_id": "call_000001"},
                },
                "route_000001": {
                    "object_id": "route_000001",
                    "object_type": "route",
                    "bbox": [4.0, 4.0, 8.0, 5.0],
                    "layer": "met2",
                    "generated_by": {"call_id": "call_000002"},
                },
            },
            "artifacts": {},
            "pdk": {},
            "source_hashes": {},
        }
    )

    matches = snapshot.query_objects_by_bbox([4.5, 4.2, 7.5, 5.1], layer_hint="met2")
    assert matches
    assert matches[0]["object_id"] == "route_000001"

    candidates = snapshot.rank_candidate_calls([4.5, 4.2, 7.5, 5.1], rule_name="m2.min_space")
    assert candidates
    assert candidates[0]["call_id"] == "call_000002"
