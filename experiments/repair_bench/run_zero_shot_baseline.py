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


def compact_repair_packet(path: str, line_budget: int) -> str:
    packet_path = Path(path)
    if not packet_path.exists():
        return "(repair packet missing)"
    lines = packet_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[:line_budget])


def build_prompt(record: dict[str, Any], packet_line_budget: int) -> str:
    target = record["target"]
    mutation = record["mutation"]
    expected_schema = {
        "repair_actions": [
            {
                "type": "replace_text",
                "file": target["file"],
                "find": "exact buggy text to replace",
                "replace": "exact corrected text",
            }
        ],
        "rationale": "one concise sentence",
    }
    return f"""You are a gLayout repair agent.
Return ONLY valid JSON matching this schema:
{json.dumps(expected_schema, indent=2)}

Case: {record['case_id']}
Mutation operator: {mutation['operator']}
Bug summary: {mutation['description']}
Target file: {target['file']}
Target line: {target['focus_line']}

Buggy source context:
{target['buggy_context']['text']}

Localizer repair packet, truncated:
{compact_repair_packet(record['verification']['repair_packet_path'], packet_line_budget)}
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


def apply_text_replace(workspace: Path, action: dict[str, Any]) -> tuple[bool, str]:
    relpath = action.get("file")
    find = action.get("find")
    replace = action.get("replace")
    if not all(isinstance(value, str) for value in (relpath, find, replace)):
        return False, "action must contain string file/find/replace"
    path = workspace / relpath
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a zero-shot model baseline on repair-bench JSONL.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--packet-line-budget", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--pdk-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-verification", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.dataset)[: args.limit]
    output_dir = args.output_dir.resolve()
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_base = os.environ.get("QWEN_API_BASE", "http://localhost:8000/v1")
    api_key = os.environ.get("QWEN_API_KEY", "EMPTY")
    model = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct")
    summary: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        sample_id = record["sample_id"]
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(record, args.packet_line_budget)
        (sample_dir / "prompt.md").write_text(prompt)
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "case_id": record["case_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_path": str(sample_dir / "prompt.md"),
        }
        if args.dry_run:
            result["dry_run"] = True
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
            continue

        if parsed is None:
            summary.append(result)
            continue

        workspace = sample_dir / "workspace"
        copy_repo(args.repo_root.resolve(), workspace, force=args.force)
        mutation = record["mutation"]
        target_path = workspace / mutation["file_path"]
        target_source = target_path.read_text()
        target_path.write_text(target_source.replace(mutation["clean_text"], mutation["buggy_text"], 1))

        apply_results = []
        for action in parsed.get("repair_actions", []):
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

        summary.append(result)
        print(f"[zero-shot] {index + 1}/{len(records)} {sample_id} parse={result.get('parse_success')} apply={result.get('apply_success')}")

    write_json(
        output_dir / "zero_shot_summary.json",
        {
            "dataset": str(args.dataset),
            "model": model,
            "api_base": api_base,
            "limit": args.limit,
            "dry_run": args.dry_run,
            "run_verification": args.run_verification,
            "results": summary,
        },
    )
    print(f"[zero-shot] summary written to {output_dir / 'zero_shot_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
