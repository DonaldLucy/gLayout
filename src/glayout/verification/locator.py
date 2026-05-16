from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from glayout.provenance import ProvenanceSnapshot, load_provenance
from glayout.provenance.netlist_summary import parse_spice_netlist_summary


_UM_COORD_RE = re.compile(r"(-?\d+(?:\.\d+)?)um")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.$#-]*")
_LVS_STOPWORDS = {
    "cell",
    "circuit",
    "class",
    "device",
    "instance",
    "is",
    "matching",
    "mismatch",
    "net",
    "no",
    "node",
    "pin",
}
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_SPAN_BEFORE = 20
_SOURCE_SPAN_AFTER = 150
_MAX_SOURCE_SPANS = 4
_MAX_PORTS_PER_CALL = 24
_MAX_NET_PINS = 24
_MAX_DRC_REPAIR_HINTS = 12


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
            if match.lower() in _LVS_STOPWORDS:
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


def _fanout_map(summary: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not summary:
        return {}
    return {
        str(row.get("net")): row
        for row in summary.get("net_fanout", [])
        if row.get("net") is not None
    }


def _top_nodes(summary: Optional[dict[str, Any]]) -> set[str]:
    if not summary:
        return set()
    return {str(node) for node in summary.get("nodes", []) if node is not None}


def _net_fingerprint(summary: Optional[dict[str, Any]], net: Optional[str]) -> dict[str, Any]:
    if not summary or not net:
        return {"net": net, "present": False}
    fanout = _fanout_map(summary).get(net)
    pin_rows = list((fanout or {}).get("pins", []) or [])
    pin_counter: Counter[str] = Counter()
    circuit_counter: Counter[str] = Counter()
    instance_counter: Counter[str] = Counter()
    for pin in pin_rows:
        if pin.get("pin") is not None:
            pin_counter[str(pin.get("pin"))] += 1
        if pin.get("circuit_name") is not None:
            circuit_counter[str(pin.get("circuit_name"))] += 1
        if pin.get("instance") is not None:
            instance_counter[str(pin.get("instance"))] += 1
    return {
        "net": net,
        "present": fanout is not None or net in _top_nodes(summary),
        "is_top_node": net in _top_nodes(summary),
        "pin_count": (fanout or {}).get("pin_count", 0),
        "pins_truncated": (fanout or {}).get("pins_truncated", False),
        "pin_role_counts": dict(sorted(pin_counter.items())),
        "circuit_counts": dict(circuit_counter.most_common(8)),
        "instance_counts": dict(instance_counter.most_common(8)),
        "pins": pin_rows[:_MAX_NET_PINS],
    }


def _unmatched_net_fingerprints(
    lvs: Optional[dict[str, Any]],
    layout_summary: Optional[dict[str, Any]],
    schematic_summary: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not lvs:
        return rows
    for issue in lvs.get("issues", []):
        if issue.get("kind") != "lvs_net_mismatch":
            continue
        left_net = issue.get("left_net")
        right_net = issue.get("right_net")
        rows.append(
            {
                "issue_raw": issue.get("raw"),
                "present_in": issue.get("present_in"),
                "layout_net": left_net,
                "schematic_net": right_net,
                "layout_fingerprint": _net_fingerprint(layout_summary, left_net),
                "schematic_fingerprint": _net_fingerprint(schematic_summary, right_net),
            }
        )
    return rows


def _floating_label_candidates(
    layout_summary: Optional[dict[str, Any]],
    schematic_summary: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    layout_nodes = _top_nodes(layout_summary)
    layout_fanout = _fanout_map(layout_summary)
    schematic_fanout = _fanout_map(schematic_summary)
    rows: list[dict[str, Any]] = []
    for node in sorted(layout_nodes):
        row = layout_fanout.get(node)
        pin_count = int((row or {}).get("pin_count", 0) or 0)
        if pin_count == 0:
            rows.append(
                {
                    "net": node,
                    "reason": "top-level label exists in layout extraction but has no device fanout",
                    "layout_fingerprint": _net_fingerprint(layout_summary, node),
                    "schematic_fingerprint": _net_fingerprint(schematic_summary, node)
                    if node in schematic_fanout
                    else None,
                }
            )
    return rows


def _candidate_call_scores(lvs: Optional[dict[str, Any]]) -> list[tuple[str, float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    if not lvs:
        return []
    for issue in lvs.get("issues", []):
        for candidate in issue.get("candidate_calls", []):
            call_id = candidate.get("call_id")
            if call_id:
                scores[str(call_id)] += float(candidate.get("score") or 0.0)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _repair_port_priority(name: str, matched_terms: set[str]) -> tuple[int, int, str]:
    lowered = name.lower()
    score = 0
    if any(term.lower() in lowered for term in matched_terms):
        score -= 100
    for keyword in (
        "ibias",
        "purpose",
        "source",
        "drain",
        "gate",
        "tap",
        "welltie",
        "plus",
        "minus",
        "multiplier",
        "vdd",
        "vss",
    ):
        if keyword in lowered:
            score -= 10
    if "array_" in lowered or "private" in lowered:
        score += 20
    return (score, len(name), name)


def _component_port_manifest(
    snapshot: ProvenanceSnapshot,
    lvs: Optional[dict[str, Any]],
    *,
    max_calls: int = 5,
) -> list[dict[str, Any]]:
    all_terms: set[str] = set()
    if lvs:
        for issue in lvs.get("issues", []):
            all_terms.update(_names_from_issue(issue))
    rows: list[dict[str, Any]] = []
    for call_id, aggregate_score in _candidate_call_scores(lvs)[:max_calls]:
        call = snapshot.get_call(call_id) or {}
        ports = list(call.get("ports", []) or [])
        ports.sort(key=lambda port: _repair_port_priority(str(port.get("name")), all_terms))
        selected_ports = ports[:_MAX_PORTS_PER_CALL]
        rows.append(
            {
                "call_id": call_id,
                "aggregate_lvs_score": round(aggregate_score, 6),
                "generator_id": call.get("generator_id"),
                "definition": call.get("definition"),
                "callsite": call.get("callsite"),
                "output_component_name": call.get("output_component_name"),
                "output_bbox": call.get("output_bbox"),
                "port_count_total": call.get("port_count_total"),
                "ports_truncated": call.get("ports_truncated"),
                "port_selection_terms": sorted(all_terms)[:32],
                "ports_included": len(selected_ports),
                "ports_omitted": max(0, len(ports) - len(selected_ports)),
                "ports": selected_ports,
            }
        )
    return rows


def _compact_candidate_call(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "call_id": candidate.get("call_id"),
        "score": candidate.get("score"),
        "generator_id": candidate.get("generator_id"),
        "definition": candidate.get("definition"),
        "callsite": candidate.get("callsite"),
        "params": candidate.get("params", {}),
    }


def _drc_repair_hints(drc: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not drc:
        return []
    grouped: dict[tuple[str, Optional[str]], dict[str, Any]] = {}
    for issue in drc.get("issues", []):
        rule = str(issue.get("rule") or "unknown_drc_rule")
        layer_hint = issue.get("layer_hint")
        key = (rule, layer_hint)
        row = grouped.setdefault(
            key,
            {
                "type": "drc_marker_cluster",
                "confidence": "medium",
                "rule": rule,
                "layer_hint": layer_hint,
                "message": (
                    f"Magic DRC reports {rule}. Inspect geometry near the sample "
                    "bboxes and candidate generator calls."
                ),
                "issue_count": 0,
                "sample_bboxes": [],
                "candidate_calls": [],
            },
        )
        row["issue_count"] += 1
        if issue.get("bbox") and len(row["sample_bboxes"]) < 4:
            row["sample_bboxes"].append(issue.get("bbox"))
        for candidate in issue.get("candidate_calls", []):
            compact = _compact_candidate_call(candidate)
            if compact["call_id"] and compact not in row["candidate_calls"]:
                row["candidate_calls"].append(compact)
            if len(row["candidate_calls"]) >= 4:
                break
    return sorted(
        grouped.values(),
        key=lambda row: (-int(row.get("issue_count", 0)), str(row.get("rule"))),
    )[:_MAX_DRC_REPAIR_HINTS]


def _resolve_source_path(raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_file():
        return path
    normalized = raw_path.replace("\\", "/")
    for marker in ("/src/", "/tests/", "/scripts/"):
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            candidate = _REPO_ROOT / marker.strip("/") / relative
            if candidate.is_file():
                return candidate
    return None


def _source_span(raw_path: Optional[str], line: Optional[int]) -> Optional[dict[str, Any]]:
    path = _resolve_source_path(raw_path)
    if path is None or line is None:
        return None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return None
    start = max(1, int(line) - _SOURCE_SPAN_BEFORE)
    end = min(len(lines), int(line) + _SOURCE_SPAN_AFTER)
    text = "\n".join(
        f"{lineno:04d}: {lines[lineno - 1]}"
        for lineno in range(start, end + 1)
    )
    return {
        "file": str(path),
        "original_file": raw_path,
        "focus_line": line,
        "start_line": start,
        "end_line": end,
        "text": text,
    }


def _source_spans(snapshot: ProvenanceSnapshot, lvs: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for call_id, _score in _candidate_call_scores(lvs):
        call = snapshot.get_call(call_id) or {}
        for loc_key in ("definition", "callsite"):
            loc = call.get(loc_key) or {}
            key = (str(loc.get("file")), int(loc.get("line") or 0))
            if key in seen:
                continue
            span = _source_span(loc.get("file"), loc.get("line"))
            if span is None:
                continue
            span["call_id"] = call_id
            span["generator_id"] = call.get("generator_id")
            span["location_kind"] = loc_key
            spans.append(span)
            seen.add(key)
            break
        if len(spans) >= _MAX_SOURCE_SPANS:
            break
    return spans


def _repair_hints(
    lvs: Optional[dict[str, Any]],
    layout_summary: Optional[dict[str, Any]],
    schematic_summary: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    if not lvs:
        return hints
    layout_nodes = _top_nodes(layout_summary)
    schematic_nodes = _top_nodes(schematic_summary)
    layout_issue_nets = {
        issue.get("left_net")
        for issue in lvs.get("issues", [])
        if issue.get("kind") == "lvs_net_mismatch" and issue.get("left_net")
    }

    schematic_internal_nets = [
        row.get("net")
        for row in (schematic_summary or {}).get("net_fanout", [])
        if row.get("net") not in schematic_nodes and int(row.get("pin_count") or 0) >= 2
    ]
    for schematic_net in schematic_internal_nets:
        paired_layout_nets = {
            issue.get("left_net")
            for issue in lvs.get("issues", [])
            if issue.get("kind") == "lvs_net_mismatch"
            and issue.get("right_net") == schematic_net
            and issue.get("left_net")
        }
        candidate_layout_nets = sorted(
            paired_layout_nets
            | {
                net
                for net in layout_issue_nets
                if net not in layout_nodes
                and _net_fingerprint(layout_summary, net).get("pin_count", 0)
            }
        )
        if candidate_layout_nets:
            hints.append(
                {
                    "type": "missing_route_for_schematic_internal_net",
                    "confidence": "high" if paired_layout_nets else "medium",
                    "message": (
                        f"Schematic internal net {schematic_net} connects multiple blocks, "
                        "but layout has separate unmatched internal nets. Inspect/add physical routing."
                    ),
                    "schematic_net": schematic_net,
                    "schematic_fingerprint": _net_fingerprint(schematic_summary, schematic_net),
                    "candidate_layout_nets": [
                        _net_fingerprint(layout_summary, net) for net in candidate_layout_nets[:6]
                    ],
                }
            )

    for floating in _floating_label_candidates(layout_summary, schematic_summary):
        net = floating.get("net")
        schematic_fp = floating.get("schematic_fingerprint") or {}
        if schematic_fp.get("pin_count", 0):
            hints.append(
                {
                    "type": "floating_or_misplaced_top_label",
                    "confidence": "high",
                    "message": (
                        f"Top label {net} exists in layout but has no extracted device fanout; "
                        "move the label to the actual routed conductor."
                    ),
                    "net": net,
                    "layout_fingerprint": floating.get("layout_fingerprint"),
                    "schematic_fingerprint": schematic_fp,
                }
            )

    issue_text = "\n".join(str(issue.get("raw", "")) for issue in lvs.get("issues", []))
    if "B" in issue_text and "VSS" in issue_text:
        hints.append(
            {
                "type": "possible_bulk_source_net_mapping_mismatch",
                "confidence": "medium",
                "message": (
                    "B/VSS mismatches appear in LVS. Check whether schematic bulk mapping "
                    "matches the physical well/substrate tie used by the layout."
                ),
                "layout_B": _net_fingerprint(layout_summary, "B"),
                "layout_VSS": _net_fingerprint(layout_summary, "VSS"),
                "schematic_B": _net_fingerprint(schematic_summary, "B"),
                "schematic_VSS": _net_fingerprint(schematic_summary, "VSS"),
            }
        )
    return hints


def build_lvs_repair_packet(
    snapshot: ProvenanceSnapshot,
    lvs: Optional[dict[str, Any]],
    layout_summary: Optional[dict[str, Any]],
    schematic_summary: Optional[dict[str, Any]],
    drc: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    hints = _repair_hints(lvs, layout_summary, schematic_summary)
    drc_hints = _drc_repair_hints(drc)
    return {
        "purpose": "compact context for an automated or human verification repair pass",
        "status": (lvs or {}).get("status"),
        "matched": (lvs or {}).get("matched"),
        "netlists_matched": (lvs or {}).get("netlists_matched"),
        "issue_count": (lvs or {}).get("issue_count", 0),
        "drc_status": (
            "clean"
            if drc is not None and int(drc.get("issue_count") or 0) == 0
            else "violations"
            if drc is not None
            else None
        ),
        "drc_issue_count": (drc or {}).get("issue_count", 0),
        "primary_hint_types": [
            *(hint.get("type") for hint in hints[:8]),
            *(hint.get("type") for hint in drc_hints[:4]),
        ],
        "repair_hints": hints[:12],
        "drc_repair_hints": drc_hints,
        "unmatched_net_fingerprints": _unmatched_net_fingerprints(
            lvs,
            layout_summary,
            schematic_summary,
        )[:24],
        "floating_label_candidates": _floating_label_candidates(
            layout_summary,
            schematic_summary,
        )[:12],
        "component_port_manifest": _component_port_manifest(snapshot, lvs),
        "source_spans": _source_spans(snapshot, lvs),
        "model_guidance": [
            "Prefer small generator-local patches over broad refactors.",
            "Use DRC hints for geometry and spacing fixes; use LVS hints for topology, label, and netlist fixes.",
            "For LVS net mismatches, first compare schematic internal nets against layout unmatched net fanouts.",
            "If a top-level label has no layout fanout, move the label before changing topology.",
            "After each patch, rerun DRC/LVS and regenerate this locator packet.",
        ],
    }


def summarize_repair_packet(packet: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not packet:
        return None
    return {
        "status": packet.get("status"),
        "matched": packet.get("matched"),
        "netlists_matched": packet.get("netlists_matched"),
        "issue_count": packet.get("issue_count", 0),
        "drc_status": packet.get("drc_status"),
        "drc_issue_count": packet.get("drc_issue_count", 0),
        "primary_hint_types": packet.get("primary_hint_types", []),
        "repair_hint_count": len(packet.get("repair_hints", []) or []),
        "drc_repair_hint_count": len(packet.get("drc_repair_hints", []) or []),
        "unmatched_net_count": len(packet.get("unmatched_net_fingerprints", []) or []),
        "floating_label_count": len(packet.get("floating_label_candidates", []) or []),
        "component_port_manifest_count": len(packet.get("component_port_manifest", []) or []),
        "source_span_count": len(packet.get("source_spans", []) or []),
    }


def _matched_netlist_excerpt(summary: dict[str, Any], names: set[str]) -> dict[str, Any]:
    matched_instances: list[dict[str, Any]] = []
    for instance in summary.get("instances", []):
        pin_hits = [
            pin
            for pin in instance.get("pin_connections", [])
            if pin.get("net") in names or pin.get("pin") in names
        ]
        if (
            instance.get("name") in names
            or instance.get("circuit_name") in names
            or pin_hits
        ):
            matched_instances.append(
                {
                    "name": instance.get("name"),
                    "circuit_name": instance.get("circuit_name"),
                    "pin_hits": pin_hits[:8],
                }
            )

    matched_fanout = [
        {
            "net": fanout.get("net"),
            "pin_count": fanout.get("pin_count"),
            "pins": fanout.get("pins", [])[:8],
        }
        for fanout in summary.get("net_fanout", [])
        if fanout.get("net") in names
    ]
    return {
        "circuit_name": summary.get("circuit_name"),
        "nodes": summary.get("nodes", [])[:16],
        "matched_nodes": [node for node in summary.get("nodes", []) if node in names][:16],
        "instance_count": summary.get("instance_count"),
        "net_count": summary.get("net_count"),
        "matched_instances": matched_instances[:6],
        "matched_fanout": matched_fanout[:6],
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
                "matched_terms": sorted(names)[:16],
                "netlist_excerpt": _matched_netlist_excerpt(summary, names),
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
    include_repair_packet: bool = False,
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
    repair_packet = (
        build_lvs_repair_packet(snapshot, lvs, layout_summary, schematic_summary, drc=drc)
        if lvs is not None or drc is not None
        else None
    )
    result = {
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
        "repair_packet_summary": summarize_repair_packet(repair_packet),
    }
    if include_repair_packet:
        result["repair_packet"] = repair_packet
    return result
