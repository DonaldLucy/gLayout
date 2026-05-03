from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_cases_module():
    cases_path = Path(__file__).resolve().parent / "smgr_cases.py"
    spec = importlib.util.spec_from_file_location("glayout_smgr_cases_test", cases_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_case_catalog_contains_public_cells_and_subblocks():
    cases = _load_cases_module()
    case_ids = {case.case_id for case in cases.SMGR_CASES}

    expected = {
        "diff_pair_default",
        "diff_pair_generic",
        "current_mirror_nfet",
        "current_mirror_pfet",
        "transmission_gate",
        "flipped_voltage_follower",
        "low_voltage_cmirror",
        "differential_to_single_ended_converter",
        "diff_pair_ibias",
        "stacked_nfet_current_mirror",
        "opamp_twostage",
        "opamp",
        "p_block",
        "n_block",
        "super_class_ab_ota",
    }

    assert expected.issubset(case_ids)
