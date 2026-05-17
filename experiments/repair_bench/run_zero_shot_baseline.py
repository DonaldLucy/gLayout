from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from run_repair_bench import IGNORE_COPY_PATTERNS, load_json, verification_env, write_json
except ImportError:  # pragma: no cover - supports python -m execution.
    from experiments.repair_bench.run_repair_bench import (
        IGNORE_COPY_PATTERNS,
        load_json,
        verification_env,
        write_json,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _repo_relative_path(raw_path: Any) -> Any:
    if not isinstance(raw_path, str) or not raw_path:
        return raw_path
    normalized = raw_path.replace("\\", "/")
    if "/workspace/" in normalized:
        normalized = normalized.split("/workspace/", 1)[1]
    for marker in ("src/", "experiments/", "scripts/", "tests/"):
        if marker in normalized:
            return normalized[normalized.index(marker) :]
    return raw_path


def _compact_location(location: Any) -> Any:
    if not isinstance(location, dict):
        return location
    compact = {key: location.get(key) for key in ("file", "line", "function") if location.get(key) is not None}
    if "file" in compact:
        compact["file"] = _repo_relative_path(compact["file"])
    return compact


def _compact_fingerprint(fingerprint: Any) -> Any:
    if not isinstance(fingerprint, dict):
        return fingerprint
    return {
        key: fingerprint.get(key)
        for key in (
            "net",
            "present",
            "is_top_node",
            "pin_count",
            "pin_role_counts",
            "circuit_counts",
            "instance_counts",
        )
        if fingerprint.get(key) not in (None, {}, [])
    }


def _line_budget_text(text: str, line_budget: int) -> str:
    if line_budget <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= line_budget:
        return text
    kept = lines[:line_budget]
    kept.append(f"... truncated {len(lines) - line_budget} additional compact-packet lines ...")
    return "\n".join(kept)


def _focus_span_text(text: Any, focus_line: Any, *, max_lines: int = 36) -> str:
    if not isinstance(text, str):
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    try:
        focus = int(focus_line)
    except (TypeError, ValueError):
        return "\n".join(lines[:max_lines])

    indexed_lines: list[tuple[int, str]] = []
    for line in lines:
        match = re.match(r"\s*(\d+):", line)
        if match:
            indexed_lines.append((int(match.group(1)), line))
    if not indexed_lines:
        return "\n".join(lines[:max_lines])

    before = max_lines // 3
    after = max_lines - before - 1
    selected = [
        line
        for line_no, line in indexed_lines
        if focus - before <= line_no <= focus + after
    ]
    if not selected:
        selected = lines[:max_lines]
    return "\n".join(selected[:max_lines])


def _compact_source_candidate(candidate: Any) -> Any:
    if not isinstance(candidate, dict):
        return candidate
    compact = {
        key: candidate.get(key)
        for key in (
            "type",
            "score",
            "match_kind",
            "label",
            "matched_terms",
            "file",
            "line",
            "text",
        )
        if candidate.get(key) not in (None, [], {})
    }
    if "file" in compact:
        compact["file"] = _repo_relative_path(compact["file"])
    return compact


def _compact_repair_hint(hint: Any) -> Any:
    if not isinstance(hint, dict):
        return hint
    compact: dict[str, Any] = {
        key: hint.get(key)
        for key in (
            "type",
            "confidence",
            "message",
            "net",
            "schematic_net",
        )
        if hint.get(key) not in (None, [], {})
    }
    for key in ("source_label_candidates", "source_netlist_candidates", "source_physical_candidates"):
        if isinstance(hint.get(key), list):
            compact[key] = [_compact_source_candidate(candidate) for candidate in hint[key][:4]]
    if isinstance(hint.get("layout_fingerprint"), dict):
        compact["layout_fingerprint"] = _compact_fingerprint(hint["layout_fingerprint"])
    if isinstance(hint.get("schematic_fingerprint"), dict):
        compact["schematic_fingerprint"] = _compact_fingerprint(hint["schematic_fingerprint"])
    if isinstance(hint.get("candidate_layout_nets"), list):
        compact["candidate_layout_nets"] = [
            _compact_fingerprint(candidate)
            for candidate in hint["candidate_layout_nets"][:4]
        ]
    return compact


def _compact_drc_hint(hint: Any) -> Any:
    if not isinstance(hint, dict):
        return hint
    compact = {
        key: hint.get(key)
        for key in (
            "type",
            "confidence",
            "rule",
            "layer_hint",
            "message",
            "issue_count",
            "sample_bboxes",
        )
        if hint.get(key) not in (None, [], {})
    }
    calls = []
    for call in (hint.get("candidate_calls") or [])[:4]:
        if not isinstance(call, dict):
            continue
        calls.append(
            {
                key: (
                    _compact_location(call.get(key))
                    if key in {"definition", "callsite"}
                    else call.get(key)
                )
                for key in ("call_id", "score", "generator_id", "definition", "callsite")
                if call.get(key) not in (None, [], {})
            }
        )
    if calls:
        compact["candidate_calls"] = calls
    return compact


def _compact_unmatched_net(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    return {
        key: value
        for key, value in {
            "issue_raw": row.get("issue_raw"),
            "present_in": row.get("present_in"),
            "layout_net": row.get("layout_net"),
            "schematic_net": row.get("schematic_net"),
            "layout_fingerprint": _compact_fingerprint(row.get("layout_fingerprint")),
            "schematic_fingerprint": _compact_fingerprint(row.get("schematic_fingerprint")),
        }.items()
        if value not in (None, [], {})
    }


def _compact_component_manifest(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    ports = row.get("ports") or []
    return {
        key: value
        for key, value in {
            "call_id": row.get("call_id"),
            "aggregate_lvs_score": row.get("aggregate_lvs_score"),
            "generator_id": row.get("generator_id"),
            "definition": _compact_location(row.get("definition")),
            "callsite": _compact_location(row.get("callsite")),
            "port_count_total": row.get("port_count_total"),
            "ports_included": row.get("ports_included"),
            "ports_omitted": row.get("ports_omitted"),
            "port_selection_terms": (row.get("port_selection_terms") or [])[:12],
            "top_port_names": [
                port.get("name")
                for port in ports[:12]
                if isinstance(port, dict) and port.get("name")
            ],
        }.items()
        if value not in (None, [], {})
    }


def _compact_source_span(span: Any) -> Any:
    if not isinstance(span, dict):
        return span
    compact = {
        key: span.get(key)
        for key in (
            "location_kind",
            "file",
            "focus_line",
            "start_line",
            "end_line",
            "score",
            "label",
            "matched_terms",
            "call_id",
            "generator_id",
        )
        if span.get(key) not in (None, [], {})
    }
    if "file" in compact:
        compact["file"] = _repo_relative_path(compact["file"])
    compact["text"] = _focus_span_text(span.get("text"), span.get("focus_line"))
    return compact


def raw_repair_packet(path: str, line_budget: int) -> str:
    packet_path = Path(path)
    if not packet_path.exists():
        return "(repair packet missing)"
    lines = packet_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[:line_budget])


def compact_repair_packet(path: str, line_budget: int, *, prompt_style: str = "compact") -> str:
    if prompt_style == "raw":
        return raw_repair_packet(path, line_budget)

    packet_path = Path(path)
    if not packet_path.exists():
        return "(repair packet missing)"
    try:
        packet = json.loads(packet_path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return raw_repair_packet(path, line_budget)

    compact = {
        "summary": {
            "purpose": packet.get("purpose"),
            "status": packet.get("status"),
            "matched": packet.get("matched"),
            "netlists_matched": packet.get("netlists_matched"),
            "issue_count": packet.get("issue_count", 0),
            "drc_status": packet.get("drc_status"),
            "drc_issue_count": packet.get("drc_issue_count", 0),
            "primary_hint_types": packet.get("primary_hint_types", []),
        },
        "model_guidance": packet.get("model_guidance", []),
        "repair_hints": [_compact_repair_hint(hint) for hint in (packet.get("repair_hints") or [])[:8]],
        "drc_repair_hints": [_compact_drc_hint(hint) for hint in (packet.get("drc_repair_hints") or [])[:6]],
        "source_label_candidates": [
            _compact_source_candidate(candidate)
            for candidate in (packet.get("source_label_candidates") or [])[:8]
        ],
        "source_netlist_candidates": [
            _compact_source_candidate(candidate)
            for candidate in (packet.get("source_netlist_candidates") or [])[:8]
        ],
        "source_physical_candidates": [
            _compact_source_candidate(candidate)
            for candidate in (packet.get("source_physical_candidates") or [])[:8]
        ],
        "unmatched_net_fingerprints": [
            _compact_unmatched_net(row)
            for row in (packet.get("unmatched_net_fingerprints") or [])[:8]
        ],
        "floating_label_candidates": [
            {
                key: value
                for key, value in {
                    "net": row.get("net"),
                    "reason": row.get("reason"),
                    "layout_fingerprint": _compact_fingerprint(row.get("layout_fingerprint")),
                    "schematic_fingerprint": _compact_fingerprint(row.get("schematic_fingerprint")),
                }.items()
                if value not in (None, [], {})
            }
            for row in (packet.get("floating_label_candidates") or [])[:6]
            if isinstance(row, dict)
        ],
        "component_call_summaries": [
            _compact_component_manifest(row)
            for row in (packet.get("component_port_manifest") or [])[:4]
        ],
        "source_spans": [
            _compact_source_span(span)
            for span in (packet.get("source_spans") or [])[:6]
        ],
    }
    compact_text = json.dumps(compact, indent=2, sort_keys=True)
    return _line_budget_text(compact_text, line_budget)


def mutation_key(record: dict[str, Any]) -> tuple[str | None, str | None]:
    mutation = record.get("mutation") or {}
    return record.get("case_id"), mutation.get("mutation_id")


def select_records(
    records: list[dict[str, Any]],
    *,
    unique_mutations: bool,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for record in records:
        if unique_mutations:
            key = mutation_key(record)
            if key in seen:
                continue
            seen.add(key)
        selected.append(record)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def build_prompt(
    record: dict[str, Any],
    packet_line_budget: int,
    *,
    prompt_style: str,
    include_oracle_mutation_summary: bool,
    include_oracle_target_context: bool,
) -> str:
    target = record["target"]
    mutation = record["mutation"]
    expected_schema = {
        "repair_actions": [
            {
                "type": "replace_text",
                "file": "relative/path/to/source.py",
                "find": "exact buggy text to replace",
                "replace": "exact corrected text",
            }
        ],
        "rationale": "one concise sentence",
    }
    oracle_mutation_summary = ""
    if include_oracle_mutation_summary:
        oracle_mutation_summary = f"""
Oracle mutation metadata, for ablation only:
Mutation operator: {mutation['operator']}
Bug summary: {mutation['description']}
"""

    oracle_target_context = ""
    if include_oracle_target_context:
        oracle_target_context = f"""
Oracle target context, for ablation only:
Target file: {target['file']}
Target line: {target['focus_line']}

Buggy source context:
{target['buggy_context']['text']}
"""

    return f"""You are a gLayout verification repair agent.
Return ONLY valid JSON matching this schema:
{json.dumps(expected_schema, indent=2)}

Case: {record['case_id']}
The repair packet below is produced from DRC/LVS plus SMGR provenance.
Do not weaken verification criteria. Prefer the smallest generator-local source edit.
Use source spans and candidate locations in the packet to choose the file and exact text replacement.
{oracle_mutation_summary}{oracle_target_context}

Localizer repair packet, truncated:
{compact_repair_packet(record['verification']['repair_packet_path'], packet_line_budget, prompt_style=prompt_style)}
"""


def call_openai_compatible(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only machine-parseable JSON. No Markdown."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"model endpoint returned HTTP {exc.code}: {body}") from exc
    return data["choices"][0]["message"]["content"]


def check_model_endpoint(api_base: str, api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        api_base.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode(errors="replace")
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach OpenAI-compatible model endpoint at {api_base.rstrip('/')}/models: {exc!r}"
        ) from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def parse_action_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def copy_repo(repo_root: Path, workspace: Path, force: bool) -> None:
    if workspace.exists() and force:
        shutil.rmtree(workspace)
    if not workspace.exists():
        shutil.copytree(repo_root, workspace, ignore=IGNORE_COPY_PATTERNS)


def normalize_action_file(workspace: Path, relpath: str) -> Path | None:
    relpath = relpath.strip()
    if relpath.startswith(("a/", "b/")):
        relpath = relpath[2:]
    if relpath.startswith("./"):
        relpath = relpath[2:]
    path = Path(relpath)
    if path.is_absolute():
        parts = path.parts
        for marker in ("src", "experiments", "scripts", "tests"):
            if marker in parts:
                path = Path(*parts[parts.index(marker) :])
                break
    resolved = (workspace / path).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        return None
    return resolved


def apply_text_replace(workspace: Path, action: dict[str, Any]) -> tuple[bool, str]:
    relpath = action.get("file")
    find = action.get("find")
    replace = action.get("replace")
    if not all(isinstance(value, str) for value in (relpath, find, replace)):
        return False, "action must contain string file/find/replace"
    path = normalize_action_file(workspace, relpath)
    if path is None:
        return False, f"file path escapes workspace: {relpath}"
    if not path.exists():
        return False, f"file not found: {relpath}"
    text = path.read_text()
    if find not in text:
        return False, "find text not present"
    path.write_text(text.replace(find, replace, 1))
    return True, "applied"


def run_verification(
    workspace: Path,
    case_id: str,
    output_dir: Path,
    timeout: int,
    pdk_root: Path | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/run_smgr_verification_locator.py",
        "--case",
        case_id,
        "--output-dir",
        str(output_dir),
        "--top-k",
        "8",
    ]
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=workspace,
        env=verification_env(workspace, pdk_root=pdk_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    log_path = output_dir / "verification.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout)
    return {
        "returncode": completed.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "log_path": str(log_path),
        "case_result_path": str(output_dir / case_id / "case_result.json"),
    }


def repair_actions(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    actions = parsed.get("repair_actions")
    if isinstance(actions, list):
        return [action for action in actions if isinstance(action, dict)]
    if all(key in parsed for key in ("type", "file", "find", "replace")):
        return [parsed]
    return []


def is_strict_clean(case_result: Any) -> bool:
    if not isinstance(case_result, dict):
        return False
    for key in ("baseline_drc", "traced_drc", "baseline_lvs", "traced_lvs"):
        status = case_result.get(key) or {}
        if status.get("is_clean") is not True:
            return False
    return True


def exact_expected_action(parsed_actions: list[dict[str, Any]], record: dict[str, Any]) -> bool:
    expected = record.get("expected_repair_action") or {}
    return any(
        action.get("type") == expected.get("type")
        and action.get("file") == expected.get("file")
        and action.get("find") == expected.get("find")
        and action.get("replace") == expected.get("replace")
        for action in parsed_actions
    )


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    parse_success = sum(1 for result in results if result.get("parse_success") is True)
    apply_success = sum(1 for result in results if result.get("apply_success") is True)
    exact_action = sum(1 for result in results if result.get("exact_expected_action") is True)
    verification_runs = sum(1 for result in results if "verification" in result)
    verification_clean = sum(1 for result in results if result.get("verification_strict_clean") is True)
    return {
        "total": total,
        "parse_success": parse_success,
        "parse_success_rate": (parse_success / total) if total else None,
        "apply_success": apply_success,
        "apply_success_rate": (apply_success / total) if total else None,
        "exact_expected_action": exact_action,
        "exact_expected_action_rate": (exact_action / total) if total else None,
        "verification_runs": verification_runs,
        "verification_strict_clean": verification_clean,
        "verification_strict_clean_rate": (verification_clean / verification_runs) if verification_runs else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a zero-shot model baseline on repair-bench JSONL.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20, help="Maximum selected rows. Use <=0 for all selected rows.")
    parser.add_argument(
        "--unique-mutations",
        action="store_true",
        help="Evaluate only the first row for each (case_id, mutation_id).",
    )
    parser.add_argument(
        "--write-selected-dataset",
        type=Path,
        default=None,
        help="Optional JSONL path containing the exact selected rows.",
    )
    parser.add_argument("--packet-line-budget", type=int, default=500)
    parser.add_argument(
        "--prompt-style",
        choices=("compact", "raw"),
        default="compact",
        help=(
            "compact emits a source-evidence-first repair packet; raw preserves the old "
            "first-N-lines JSON packet for ablation runs."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--pdk-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-verification", action="store_true")
    parser.add_argument(
        "--skip-model-preflight",
        action="store_true",
        help="Skip the /models endpoint reachability check before issuing completions.",
    )
    parser.add_argument(
        "--include-oracle-mutation-summary",
        action="store_true",
        help="Ablation mode: include injected mutation operator and description in the prompt.",
    )
    parser.add_argument(
        "--include-oracle-target-context",
        action="store_true",
        help="Ablation mode: include the known mutated line context in the prompt.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_records = read_jsonl(args.dataset)
    records = select_records(
        all_records,
        unique_mutations=args.unique_mutations,
        limit=args.limit,
    )
    output_dir = args.output_dir.resolve()
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_selected_dataset:
        write_jsonl(args.write_selected_dataset.resolve(), records)

    api_base = os.environ.get("QWEN_API_BASE", "http://localhost:8000/v1")
    api_key = os.environ.get("QWEN_API_KEY", "EMPTY")
    model = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct")
    model_preflight: dict[str, Any] | None = None
    if not args.dry_run and not args.skip_model_preflight:
        try:
            model_preflight = check_model_endpoint(api_base, api_key, timeout=min(args.timeout, 30))
            model_count = len(model_preflight.get("data", [])) if isinstance(model_preflight.get("data"), list) else "unknown"
            print(f"[zero-shot] model endpoint reachable: {api_base.rstrip('/')} ({model_count} models)")
        except Exception as exc:
            write_json(
                output_dir / "zero_shot_summary.json",
                {
                    "dataset": str(args.dataset),
                    "model": model,
                    "api_base": api_base,
                    "limit": args.limit,
                    "unique_mutations": args.unique_mutations,
                    "selected_rows": len(records),
                    "available_rows": len(all_records),
                    "dry_run": args.dry_run,
                    "run_verification": args.run_verification,
                    "prompt_style": args.prompt_style,
                    "preflight_error": repr(exc),
                    "aggregate": aggregate_results([]),
                    "results": [],
                },
            )
            raise SystemExit(f"[zero-shot] {exc}") from exc
    summary: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        sample_id = record["sample_id"]
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(
            record,
            args.packet_line_budget,
            prompt_style=args.prompt_style,
            include_oracle_mutation_summary=args.include_oracle_mutation_summary,
            include_oracle_target_context=args.include_oracle_target_context,
        )
        (sample_dir / "prompt.md").write_text(prompt)
        mutation = record.get("mutation") or {}
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "case_id": record["case_id"],
            "mutation_id": mutation.get("mutation_id"),
            "operator": mutation.get("operator"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_path": str(sample_dir / "prompt.md"),
        }
        if args.dry_run:
            result["dry_run"] = True
            print(f"[zero-shot] {index + 1}/{len(records)} {sample_id} dry_run=True")
            summary.append(result)
            continue

        try:
            response = call_openai_compatible(
                api_base=api_base,
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            (sample_dir / "response.txt").write_text(response)
            parsed = parse_action_json(response)
            result["parse_success"] = parsed is not None
            if parsed is not None:
                write_json(sample_dir / "repair_action.json", parsed)
        except Exception as exc:
            result["model_error"] = repr(exc)
            summary.append(result)
            print(f"[zero-shot] {index + 1}/{len(records)} {sample_id} model_error={exc!r}")
            continue

        if parsed is None:
            summary.append(result)
            continue

        workspace = sample_dir / "workspace"
        copy_repo(args.repo_root.resolve(), workspace, force=args.force)
        target_path = workspace / mutation["file_path"]
        target_source = target_path.read_text()
        target_path.write_text(target_source.replace(mutation["clean_text"], mutation["buggy_text"], 1))

        apply_results = []
        parsed_actions = repair_actions(parsed)
        result["exact_expected_action"] = exact_expected_action(parsed_actions, record)
        for action in parsed_actions:
            success, message = apply_text_replace(workspace, action)
            apply_results.append({"success": success, "message": message, "action": action})
        result["apply_results"] = apply_results
        result["apply_success"] = bool(apply_results) and all(item["success"] for item in apply_results)

        if result["apply_success"] and args.run_verification:
            verifier = run_verification(
                workspace,
                record["case_id"],
                sample_dir / "verification",
                args.timeout,
                args.pdk_root.resolve() if args.pdk_root else None,
            )
            result["verification"] = verifier
            result["case_result"] = load_json(Path(verifier["case_result_path"]))
            result["verification_strict_clean"] = is_strict_clean(result["case_result"])

        summary.append(result)
        print(
            f"[zero-shot] {index + 1}/{len(records)} {sample_id} "
            f"parse={result.get('parse_success')} apply={result.get('apply_success')} "
            f"clean={result.get('verification_strict_clean')}"
        )

    write_json(
        output_dir / "zero_shot_summary.json",
        {
            "dataset": str(args.dataset),
            "model": model,
            "api_base": api_base,
            "limit": args.limit,
            "unique_mutations": args.unique_mutations,
            "selected_rows": len(records),
            "available_rows": len(all_records),
            "dry_run": args.dry_run,
            "run_verification": args.run_verification,
            "prompt_style": args.prompt_style,
            "model_preflight": model_preflight,
            "include_oracle_mutation_summary": args.include_oracle_mutation_summary,
            "include_oracle_target_context": args.include_oracle_target_context,
            "aggregate": aggregate_results(summary),
            "results": summary,
        },
    )
    print(f"[zero-shot] summary written to {output_dir / 'zero_shot_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
