from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from glayout.provenance import ProvenanceSnapshot, load_provenance
from glayout.provenance.netlist_summary import parse_spice_netlist_summary


_UM_COORD_RE = re.compile(r"(-?\d+(?:\.\d+)?)um")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.$#-]*")


def _read_text(path: Optional[Path]) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(errors="replace")


def _resolve_report_path(raw_path: Optional[str], case_dir: Path, fallback_glob: str) -> Optional[Path]:
    if raw_path:
        path = Path(raw_path)
        if path.is_file():
            return path
    matches = sorted(case_dir.glob(fallback_glob))
    return matches[0] if matches else None


def _extract_layer_hint(rule: str) -> Optional[str]:
    lower = rule.lower()
    if "metal1" in lower or "met1" in lower:
        return "met1"
    if "metal2" in lower or "met2" in lower:
        return "met2"
    if "metal3" in lower or "met3" in lower:
        return "met3"
    if "metal4" in lower or "met4" in lower:
        return "met4"
    if "metal5" in lower or "met5" in lower:
        return "met5"
    if "via1" in lower:
        return "via1"
    if "via2" in lower:
        return "via2"
    return None


def parse_magic_drc_report(report_path: Path, *, max_issues: int = 32) -> dict[str, Any]:
    text = _read_text(report_path)
    issues: list[dict[str, Any]] = []
    current_rule = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("-"):
            continue
        coords = [float(value) for value in _UM_COORD_RE.findall(line)]
        if len(coords) >= 4:
            issues.append(
                {
                    "kind": "drc_marker",
                    "rule": current_rule,
                    "bbox": [coords[0], coords[1], coords[2], coords[3]],
                    "layer_hint": _extract_layer_hint(current_rule),
                    "raw": line,
                }
            )
            if len(issues) >= max_issues:
                break
            continue
        if not line.lower().endswith("count:") and "count:" not in line:
            current_rule = line
    return {
        "tool": "magic",
        "report": str(report_path),
        "issue_count": len(issues),
        "issues": issues,
        "truncated": len(issues) >= max_issues,
    }


def _split_netgen_row(line: str) -> tuple[str, str]:
    if "|" not in line:
        return line.strip(), ""
    left, right = line.split("|", 1)
    return left.strip(), right.strip()


def _extract_named_value(text: str, prefix: str) -> Optional[str]:
    match = re.search(rf"{re.escape(prefix)}\s*:\s*([^\s].*?)\s*(?:\*\*Mismatch\*\*)?$", text)
    if match:
        return match.group(1).strip()
    return None


def parse_netgen_lvs_report(report_path: Path, *, max_issues: int = 64) -> dict[str, Any]:
    text = _read_text(report_path)
    netlists_matched = "Netlists match uniquely" in text or "Netlists match with" in text
    strict_matched = "Circuits match uniquely" in text
    property_error = "Property errors were found." in text
    hard_fail = "Top level cell failed pin matching." in text or "Netlists do not match." in text
    issues: list[dict[str, Any]] = []
    section: Optional[str] = None
    in_pin_table = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            in_pin_table = False
            continue
        if "NET mismatches" in stripped:
            section = "net"
            continue
        if "DEVICE mismatches" in stripped:
            section = "device"
            continue
        if stripped.startswith("Subcircuit pins:"):
            section = "pins"
            in_pin_table = True
            continue
        if stripped.startswith("Final result:"):
            section = None
            in_pin_table = False
            continue

        left, right = _split_netgen_row(line)
        if "Net:" in stripped or "(no matching net)" in stripped:
            issue = {
                "kind": "lvs_net_mismatch",
                "left": left,
                "right": right,
                "left_net": _extract_named_value(left, "Net"),
                "right_net": _extract_named_value(right, "Net"),
                "present_in": None,
                "raw": stripped,
                "section": section,
            }
            if "(no matching net)" in left:
                issue["present_in"] = "schematic"
            elif "(no matching net)" in right:
                issue["present_in"] = "layout"
            issues.append(issue)
        elif "Instance:" in stripped or "(no matching instance)" in stripped:
            issue = {
                "kind": "lvs_device_mismatch",
                "left": left,
                "right": right,
                "left_instance": _extract_named_value(left, "Instance"),
                "right_instance": _extract_named_value(right, "Instance"),
                "raw": stripped,
                "section": section,
            }
            issues.append(issue)
        elif in_pin_table and "|" in line and ("**Mismatch**" in line or "(no pin" in line):
            issues.append(
                {
                    "kind": "lvs_pin_mismatch",
                    "left": left,
                    "right": right,
                    "raw": stripped,
                    "section": section,
                }
            )

        if len(issues) >= max_issues:
            break

    return {
        "tool": "netgen",
        "report": str(report_path),
        "matched": strict_matched,
        "netlists_matched": netlists_matched,
        "property_error": property_error,
        "status": (
            "clean"
            if strict_matched and not property_error and not hard_fail
            else "property_error"
            if property_error
            else "top_level_pin_mismatch"
            if netlists_matched and hard_fail
            else "mismatch"
        ),
        "issue_count": len(issues),
        "issues": issues,
        "truncated": len(issues) >= max_issues,
    }


def _names_from_issue(issue: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("left_net", "right_net", "left_instance", "right_instance", "left", "right", "raw"):
        value = issue.get(key)
        if not value:
            continue
        for match in _NAME_RE.findall(str(value)):
            if match.lower() in {"net", "instance", "no", "matching", "mismatch", "circuit"}:
                continue
            names.add(match)
    return names


def _call_depth(calls: dict[str, dict[str, Any]], call_id: str) -> int:
    depth = 0
    current = calls.get(call_id)
    while current and current.get("parent_call_id"):
        depth += 1
        current = calls.get(current["parent_call_id"])
    return depth


def _source_payload(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "callsite": call.get("callsite"),
        "definition": call.get("definition"),
        "generator_id": call.get("generator_id"),
        "params": call.get("params", {}),
    }


def rank_lvs_candidate_calls(
    snapshot: ProvenanceSnapshot,
    issue: dict[str, Any],
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    names = _names_from_issue(issue)
    ranked: list[dict[str, Any]] = []
    for call_id, call in snapshot.calls.items():
        score = 0.0
        reasons: list[str] = []
        summary = call.get("netlist_summary") or {}
        text_blob = " ".join(
            str(value)
            for value in [
                call.get("generator_id"),
                call.get("function_name"),
                call.get("module"),
                call.get("output_component_name"),
                summary.get("circuit_name"),
            ]
            if value
        )
        for name in names:
            if not name:
                continue
            if name in summary.get("nodes", []):
                score += 4.0
                reasons.append(f"net {name} appears in call nodes")
            if name.lower() in text_blob.lower():
                score += 1.5
                reasons.append(f"name {name} appears in generator context")
            for instance in summary.get("instances", []):
                if name == instance.get("name") or name == instance.get("circuit_name"):
                    score += 3.0
                    reasons.append(f"instance/circuit {name} appears in schematic summary")
                for pin in instance.get("pin_connections", []):
                    if name == pin.get("net") or name == pin.get("pin"):
                        score += 2.0
                        reasons.append(f"{name} appears in instance pin map")
            for fanout in summary.get("net_fanout", []):
                if name == fanout.get("net"):
                    score += 3.0
                    reasons.append(f"net {name} appears in fanout")

        if issue.get("kind") == "lvs_pin_mismatch" and summary.get("nodes"):
            score += 0.5
            reasons.append("pin mismatch and call has top-level nodes")
        if score <= 0.0:
            continue
        depth = _call_depth(snapshot.calls, call_id)
        score += min(depth, 6) * 0.08
        ranked.append(
            {
                "call_id": call_id,
                "score": round(score, 6),
                "reasons": sorted(set(reasons))[:8],
                **_source_payload(call),
                "netlist_excerpt": {
                    "circuit_name": summary.get("circuit_name"),
                    "nodes": summary.get("nodes", [])[:16],
                    "instance_count": summary.get("instance_count"),
                    "net_count": summary.get("net_count"),
                },
            }
        )

    if not ranked:
        roots = [call for call in snapshot.calls.values() if not call.get("parent_call_id")]
        for call in roots[:top_k]:
            ranked.append(
                {
                    "call_id": call.get("call_id"),
                    "score": 0.1,
                    "reasons": ["fallback root generator"],
                    **_source_payload(call),
                    "netlist_excerpt": {},
                }
            )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def _rank_drc_issue(snapshot: ProvenanceSnapshot, issue: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    bbox = issue.get("bbox")
    if not bbox:
        return []
    candidates = snapshot.rank_candidate_calls(
        bbox,
        rule_name=issue.get("rule"),
        layer_hint=issue.get("layer_hint"),
    )
    enriched: list[dict[str, Any]] = []
    for candidate in candidates[:top_k]:
        call = snapshot.get_call(candidate["call_id"]) or {}
        enriched.append(
            {
                **candidate,
                **_source_payload(call),
            }
        )
    return enriched


def locate_case_result(
    case_result_path: Path,
    *,
    top_k: int = 8,
    max_drc_issues: int = 24,
    max_lvs_issues: int = 48,
) -> dict[str, Any]:
    case_result = json.loads(case_result_path.read_text())
    case_dir = case_result_path.parent
    case_id = case_result.get("case_id", case_dir.name)
    sidecar_path = _resolve_report_path(
        case_result.get("sidecar"),
        case_dir,
        f"{case_id}.traced.provenance.json",
    )
    if sidecar_path is None:
        raise FileNotFoundError(f"Could not resolve provenance sidecar for {case_id}")
    snapshot = load_provenance(sidecar_path)

    drc_path = _resolve_report_path(
        (case_result.get("traced_drc") or {}).get("report"),
        case_dir,
        f"magic_drc/drc/{case_id}_traced/{case_id}_traced.rpt",
    )
    lvs_path = _resolve_report_path(
        (case_result.get("traced_lvs") or {}).get("report"),
        case_dir,
        f"netgen_lvs/lvs/{case_id}_traced/{case_id}_traced_lvs.rpt",
    )

    drc = parse_magic_drc_report(drc_path, max_issues=max_drc_issues) if drc_path else None
    if drc is not None:
        for issue in drc["issues"]:
            issue["candidate_calls"] = _rank_drc_issue(snapshot, issue, top_k=top_k)

    lvs = parse_netgen_lvs_report(lvs_path, max_issues=max_lvs_issues) if lvs_path else None
    if lvs is not None:
        for issue in lvs["issues"]:
            issue["candidate_calls"] = rank_lvs_candidate_calls(snapshot, issue, top_k=top_k)

    lvs_dir = lvs_path.parent if lvs_path else case_dir
    layout_spice = lvs_dir / f"{case_id}_traced_lvsmag.spice"
    schematic_spice = lvs_dir / f"{case_id}_traced.spice"
    layout_summary = parse_spice_netlist_summary(_read_text(layout_spice), circuit_name=f"{case_id}_traced") if layout_spice.is_file() else None
    schematic_summary = parse_spice_netlist_summary(_read_text(schematic_spice), circuit_name=f"{case_id}_traced") if schematic_spice.is_file() else None

    return {
        "case_id": case_id,
        "sidecar": str(sidecar_path),
        "drc": drc,
        "lvs": lvs,
        "netlists": {
            "layout_spice": str(layout_spice) if layout_spice.is_file() else None,
            "layout_summary": layout_summary,
            "schematic_spice": str(schematic_spice) if schematic_spice.is_file() else None,
            "schematic_summary": schematic_summary,
        },
    }
