from __future__ import annotations

import json
from typing import Any


def flatten_entry(entry: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "sample_id": entry["sample_id"],
        "generator_id": entry["generator_id"],
        "pdk": entry["pdk"],
        "build_status": entry["build_status"],
    }

    for key, value in entry.get("params", {}).items():
        flat[f"param_{key}"] = json.dumps(value) if isinstance(value, (list, tuple, dict)) else value

    flat["drc_status"] = entry.get("drc", {}).get("status")
    flat["drc_clean"] = entry.get("drc", {}).get("clean")
    flat["drc_total_errors"] = entry.get("drc", {}).get("total_errors")

    flat["lvs_status"] = entry.get("lvs", {}).get("status")
    flat["lvs_clean"] = entry.get("lvs", {}).get("clean")
    flat["lvs_property_error"] = entry.get("lvs", {}).get("property_error")
    flat["lvs_topology_match"] = entry.get("lvs", {}).get("topology_match")

    flat["pex_status"] = entry.get("pex", {}).get("status")
    flat["pex_total_resistance_ohms"] = entry.get("pex", {}).get("total_resistance_ohms")
    flat["pex_total_capacitance_farads"] = entry.get("pex", {}).get("total_capacitance_farads")

    flat["geom_area_um2"] = entry.get("geom", {}).get("area_um2")
    flat["geom_sym_h"] = entry.get("geom", {}).get("sym_h")
    flat["geom_sym_v"] = entry.get("geom", {}).get("sym_v")

    runtimes = entry.get("runtime_s", {})
    for key, value in runtimes.items():
        flat[f"runtime_{key}_s"] = value

    flat["failure_tags"] = json.dumps(entry.get("failure_tags", []))
    flat["structure_features"] = json.dumps(entry.get("structure_features", {}))
    return flat
