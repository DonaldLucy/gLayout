from __future__ import annotations

import re
from pathlib import Path


def summarize_drc_report(report_path: Path) -> dict:
    if not report_path.is_file():
        return {
            "status": "error",
            "clean": False,
            "total_errors": None,
            "rule_counts": {},
            "report_path": str(report_path),
        }

    content = report_path.read_text()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    rule_counts: dict[str, int] = {}
    current_rule: str | None = None
    total_errors = 0

    for line in lines:
        if line == "----------------------------------------":
            continue
        if line.startswith("Error while reading cell"):
            continue
        if line[0].isalpha():
            current_rule = line
            rule_counts.setdefault(current_rule, 0)
            continue
        if line.endswith("um") and current_rule:
            rule_counts[current_rule] += 1
            total_errors += 1

    count_zero = bool(re.search(r"count:\s*0\s*$", content, re.IGNORECASE | re.MULTILINE))
    clean = total_errors == 0 and (count_zero or "No errors found." in content or "count:" in content)

    return {
        "status": "clean" if clean else "fail",
        "clean": clean,
        "total_errors": total_errors,
        "rule_counts": rule_counts,
        "report_path": str(report_path),
    }


def summarize_lvs_report(report_path: Path) -> dict:
    if not report_path.is_file():
        return {
            "status": "error",
            "clean": False,
            "property_error": False,
            "topology_match": False,
            "report_path": str(report_path),
            "property_error_lines": [],
        }

    content = report_path.read_text()
    clean = "Final result: Circuits match uniquely." in content or "Final result:\nCircuits match uniquely." in content
    property_error = "Property errors were found." in content
    topology_mismatch = (
        "Top level cell failed pin matching." in content
        or "Netlists do not match." in content
        or "Mismatch" in content and not clean
    )

    prop_lines = [
        line.strip()
        for line in content.splitlines()
        if " circuit1:" in line and " circuit2:" in line
    ]

    if clean and property_error:
        status = "property_error"
    elif clean:
        status = "clean"
    elif topology_mismatch:
        status = "topology_mismatch"
    else:
        status = "fail"

    return {
        "status": status,
        "clean": clean and not property_error,
        "property_error": property_error,
        "topology_match": clean,
        "report_path": str(report_path),
        "property_error_lines": prop_lines,
    }


def summarize_log(text: str) -> list[str]:
    interesting = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(
            key in lowered
            for key in [
                "error",
                "warning",
                "mismatch",
                "fail",
                "property errors",
                "no such",
                "could not",
            ]
        ):
            interesting.append(line)
    return interesting[-30:]
