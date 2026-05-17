#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEFAULT_API_BASE = "http://localhost:8000/v1"
EXCLUDE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "glayout_agentic",
}
EXCLUDE_SUFFIXES = {
    ".gds",
    ".oas",
    ".spice",
    ".ext",
    ".sim",
    ".log",
}


def endpoint_help(api_base: str, model: str) -> str:
    return textwrap.dedent(
        f"""
        Model endpoint is not reachable.

        Current settings:
          QWEN_API_BASE={api_base}
          QWEN_MODEL={model}

        Quick checks:
          curl {api_base.rstrip()}/models
          ps -ef | grep -E "vllm|ollama|lmstudio|openai" | grep -v grep
          nvidia-smi

        If the model server is on this same machine, start an OpenAI-compatible
        server first, for example:
          vllm serve {model} --served-model-name {model} --host 0.0.0.0 --port 8000 --max-model-len 32768

        If the model server is on another machine, point QWEN_API_BASE at that
        machine instead of localhost, for example:
          export QWEN_API_BASE="http://<MODEL_SERVER_INTERNAL_IP>:8000/v1"
        """
    ).strip()


def check_model_endpoint(api_base: str, api_key: str, *, timeout: int) -> tuple[bool, str]:
    url = api_base.rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1024)
        return True, f"Model endpoint reachable: {url}"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, f"Model endpoint returned HTTP {exc.code} from {url}: {body[:1000]}"
    except urllib.error.URLError as exc:
        return False, f"Could not connect to model endpoint {url}: {exc}"


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}\n"
            f"See log: {log_path}"
        )
    return completed


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(name)
        if name in EXCLUDE_DIRS:
            ignored.add(name)
        elif path.suffix in EXCLUDE_SUFFIXES:
            ignored.add(name)
    return ignored


def prepare_workspace(repo_root: Path, run_root: Path, mode: str) -> Path:
    workspace = run_root / "workspace"
    if workspace.exists():
        return workspace
    if mode == "worktree":
        _run(
            ["git", "worktree", "add", "--detach", str(workspace), "HEAD"],
            cwd=repo_root,
            log_path=run_root / "worktree.log",
            check=True,
        )
        return workspace
    if mode == "copy":
        shutil.copytree(repo_root, workspace, ignore=_copy_ignore)
        return workspace
    raise ValueError(f"Unsupported workspace mode: {mode}")


def verifier_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace / "src")
    env["GLAYOUT_SMGR"] = "1"
    for key in (
        "PDK_ROOT",
        "PDKPATH",
        "MAGIC_PDK_ROOT",
        "NETGEN_PDK_ROOT",
        "GLAYOUT_SMGR_CAPTURE_POLYGONS",
        "GLAYOUT_SMGR_CAPTURE_PORT_OBJECTS",
        "GLAYOUT_SMGR_CAPTURE_LIVE_REFS",
    ):
        env.pop(key, None)
    return env


def run_verification(
    workspace: Path,
    iter_dir: Path,
    *,
    case: str,
    top_k: int,
    max_drc_issues: int,
    max_lvs_issues: int,
) -> dict[str, Any]:
    verification_root = iter_dir / "verification"
    cmd = [
        sys.executable,
        "scripts/run_smgr_verification_locator.py",
        "--case",
        case,
        "--output-dir",
        str(verification_root),
        "--top-k",
        str(top_k),
        "--max-drc-issues",
        str(max_drc_issues),
        "--max-lvs-issues",
        str(max_lvs_issues),
    ]
    completed = _run(
        cmd,
        cwd=workspace,
        env=verifier_env(workspace),
        log_path=iter_dir / "verification.log",
    )
    case_dir = verification_root / case
    return {
        "returncode": completed.returncode,
        "case_dir": str(case_dir),
        "case_result_path": str(case_dir / "case_result.json"),
        "locator_path": str(case_dir / "verification_locator.json"),
        "repair_packet_path": str(case_dir / "repair_packet.json"),
        "log_path": str(iter_dir / "verification.log"),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _is_clean_case(case_result: dict[str, Any] | None, locator: dict[str, Any] | None) -> bool:
    if case_result:
        checks = []
        for key in ("baseline_drc", "traced_drc", "baseline_lvs", "traced_lvs"):
            block = case_result.get(key) or {}
            if key.endswith("_lvs"):
                checks.append(bool(block.get("is_clean") and block.get("matched")))
            else:
                checks.append(bool(block.get("is_clean")))
        if checks and all(checks):
            return True
    lvs = (locator or {}).get("lvs") or {}
    drc = (locator or {}).get("drc") or {}
    return bool(lvs.get("matched") and int(drc.get("issue_count") or 0) == 0)


def _truncate_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    omitted = len(lines) - max_lines
    return "\n".join(lines[:max_lines] + [f"... <truncated {omitted} lines>"])


def _short_json(value: Any, *, max_chars: int = 2200) -> str:
    text = json.dumps(value, indent=2, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... <truncated {len(text) - max_chars} chars>"


def _render_fingerprint(row: dict[str, Any]) -> str:
    layout = row.get("layout_fingerprint") or {}
    schematic = row.get("schematic_fingerprint") or {}
    return "\n".join(
        [
            f"- raw: {row.get('issue_raw')}",
            f"  layout_net: {row.get('layout_net')} pins={layout.get('pin_count')} roles={layout.get('pin_role_counts')}",
            f"  schematic_net: {row.get('schematic_net')} pins={schematic.get('pin_count')} roles={schematic.get('pin_role_counts')}",
            f"  layout circuits: {layout.get('circuit_counts')}",
            f"  schematic circuits: {schematic.get('circuit_counts')}",
        ]
    )


def compact_packet_text(packet: dict[str, Any] | None, *, max_lines: int) -> str:
    if not packet:
        return "No repair_packet.json was generated. Inspect verifier log and fix the earliest failure."
    chunks: list[str] = []
    chunks.append(
        "\n".join(
            [
                f"status: {packet.get('status')}",
                f"matched: {packet.get('matched')}",
                f"netlists_matched: {packet.get('netlists_matched')}",
                f"issue_count: {packet.get('issue_count')}",
                f"drc_status: {packet.get('drc_status')}",
                f"drc_issue_count: {packet.get('drc_issue_count')}",
                f"primary_hint_types: {packet.get('primary_hint_types')}",
            ]
        )
    )
    for index, hint in enumerate(packet.get("repair_hints", [])[:6], start=1):
        chunks.append(f"\nLVS repair hint {index}:\n{_short_json(hint)}")
    for index, hint in enumerate(packet.get("drc_repair_hints", [])[:4], start=1):
        chunks.append(f"\nDRC repair hint {index}:\n{_short_json(hint)}")
    fingerprints = packet.get("unmatched_net_fingerprints", [])[:8]
    if fingerprints:
        chunks.append("\nUnmatched net fingerprints:\n" + "\n".join(_render_fingerprint(row) for row in fingerprints))
    floating = packet.get("floating_label_candidates", [])[:6]
    if floating:
        chunks.append("\nFloating label candidates:\n" + _short_json(floating, max_chars=3000))
    manifests = packet.get("component_port_manifest", [])[:5]
    if manifests:
        compact_manifests = []
        for row in manifests:
            compact_manifests.append(
                {
                    "call_id": row.get("call_id"),
                    "generator_id": row.get("generator_id"),
                    "definition": row.get("definition"),
                    "callsite": row.get("callsite"),
                    "ports_included": row.get("ports_included"),
                    "ports": row.get("ports", [])[:24],
                }
            )
        chunks.append("\nRelevant component ports:\n" + _short_json(compact_manifests, max_chars=6500))
    if packet.get("model_guidance"):
        chunks.append("\nLocator model guidance:\n" + "\n".join(f"- {row}" for row in packet["model_guidance"]))
    return _truncate_lines("\n".join(chunks), max_lines)


def source_spans_text(packet: dict[str, Any] | None, *, max_spans: int, max_lines_per_span: int) -> str:
    spans = (packet or {}).get("source_spans", [])[:max_spans]
    if not spans:
        return "No source spans were provided. Use repo search only if necessary."
    rendered: list[str] = []
    for span in spans:
        header = (
            f"Source span: {span.get('file')} focus_line={span.get('focus_line')} "
            f"call_id={span.get('call_id')} generator={span.get('generator_id')}"
        )
        rendered.append(header)
        rendered.append(_truncate_lines(str(span.get("text") or ""), max_lines_per_span))
    return "\n\n".join(rendered)


def _file_excerpt(path_text: str | None, *, max_chars: int = 1200) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    text = path.read_text(errors="replace").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...<truncated>"


def previous_iterations_text(iterations: list[dict[str, Any]]) -> str:
    if not iterations:
        return "No previous repair attempts in this run."
    rows = []
    for item in iterations[-3:]:
        apply_log = _file_excerpt(item.get("apply_log_path"), max_chars=800)
        patch_excerpt = _file_excerpt(item.get("model_patch_path"), max_chars=1200)
        rows.append(
            {
                "iteration": item.get("iteration"),
                "verifier_returncode": item.get("verifier", {}).get("returncode"),
                "clean": item.get("clean"),
                "patch_applied": item.get("patch_applied"),
                "model_patch_path": item.get("model_patch_path"),
                "apply_log_excerpt": apply_log,
                "model_patch_excerpt": patch_excerpt,
            }
        )
    return _short_json(rows, max_chars=4200)


def verifier_log_tail(log_path: str | None, *, max_lines: int = 180) -> str:
    if not log_path:
        return "No verifier log path was recorded."
    path = Path(log_path)
    if not path.is_file():
        return f"Verifier log was not found: {log_path}"
    return _truncate_lines("\n".join(path.read_text(errors="replace").splitlines()[-max_lines:]), max_lines)


def build_prompt(
    *,
    case: str,
    skill_text: str,
    packet: dict[str, Any] | None,
    iterations: list[dict[str, Any]],
    verifier_log: str | None,
    skill_line_budget: int,
    packet_line_budget: int,
    max_source_spans: int,
    source_lines_per_span: int,
    verifier_log_lines: int,
) -> str:
    skill = _truncate_lines(skill_text, skill_line_budget)
    packet_block = compact_packet_text(packet, max_lines=packet_line_budget)
    spans = source_spans_text(
        packet,
        max_spans=max_source_spans,
        max_lines_per_span=source_lines_per_span,
    )
    previous = previous_iterations_text(iterations)
    log_tail = verifier_log_tail(verifier_log, max_lines=verifier_log_lines)
    return textwrap.dedent(
        f"""
        You are a gLayout verification repair agent. Fix one failing cell using the compact skill
        sheet and repair packet below. Return ONLY a unified diff patch.

        Case: {case}

        === Compact gLayout Skill ===
        {skill}

        === Current Repair Packet Summary ===
        {packet_block}

        === Candidate Source Spans ===
        {spans}

        === Previous Iterations ===
        {previous}

        === Current Verifier Log Tail ===
        {log_tail}

        === Required Response ===
        Return a unified diff only. Do not include Markdown fences, prose, or explanations.
        Prefer the smallest source-local repair that should reduce or eliminate DRC/LVS issues.
        Generate patches against the exact current source spans shown above. Do not repeat edits
        that are already present. If a previous patch failed to apply, use the apply log to
        correct the hunk/path/context and produce a new valid patch.
        """
    ).strip() + "\n"


def call_openai_compatible(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a senior analog layout code repair agent. Output unified diffs only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        hint = ""
        if exc.code == 400 and "maximum context length" in body:
            hint = textwrap.dedent(
                """

                Context budget hint:
                  Reduce output tokens first, then trim prompt sections. For the 24576-token server, try:
                    --max-tokens 2048 --skill-line-budget 900 --packet-line-budget 600 \\
                    --max-source-spans 3 --source-lines-per-span 80 --verifier-log-lines 80
                """
            )
        raise RuntimeError(f"Model endpoint returned HTTP {exc.code}: {body}{hint}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to model endpoint {url}: {exc}\n"
            + endpoint_help(api_base, model)
        ) from exc
    return str(data["choices"][0]["message"]["content"])


def extract_unified_diff(response: str) -> str:
    text = response.strip()
    fence = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    markers = [
        "\ndiff --git ",
        "diff --git ",
        "\n--- a/",
        "--- a/",
    ]
    starts = [text.find(marker.strip() if marker.startswith("\n") else marker) for marker in markers]
    starts = [index for index in starts if index >= 0]
    if starts:
        text = text[min(starts) :].strip()
    return text + "\n"


def normalize_model_patch_paths(patch_text: str, workspace: Path) -> str:
    workspace_path = workspace.resolve().as_posix()

    def normalize_token(token: str) -> str:
        prefix = ""
        body = token
        if body.startswith(("a/", "b/")):
            prefix = body[:2]
            body = body[2:]

        body = body.replace("\\", "/")
        if body.startswith("./"):
            body = body[2:]
        if body.startswith(f"{workspace_path}/"):
            body = body[len(workspace_path) + 1 :]
        if "/workspace/" in body:
            body = body.split("/workspace/", 1)[1]
        if body.startswith("workspace/"):
            body = body[len("workspace/") :]
        return prefix + body

    normalized_lines: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                parts[2] = normalize_token(parts[2])
                parts[3] = normalize_token(parts[3])
                line = " ".join(parts)
        elif line.startswith(("--- ", "+++ ")):
            marker = line[:4]
            body = line[4:]
            if body != "/dev/null":
                if "\t" in body:
                    path, suffix = body.split("\t", 1)
                    body = normalize_token(path) + "\t" + suffix
                else:
                    body = normalize_token(body)
                line = marker + body
        normalized_lines.append(line)
    return "\n".join(normalized_lines).rstrip() + "\n"


def apply_model_patch(workspace: Path, patch_path: Path, *, apply_command: str) -> tuple[bool, str]:
    if not patch_path.read_text().strip():
        return False, "empty patch"
    if apply_command == "git":
        check = _run(["git", "apply", "--check", str(patch_path)], cwd=workspace)
        if check.returncode != 0:
            return False, check.stdout
        apply = _run(["git", "apply", str(patch_path)], cwd=workspace)
        return apply.returncode == 0, apply.stdout
    if apply_command == "patch":
        apply = _run(["patch", "-p1", "-i", str(patch_path)], cwd=workspace)
        return apply.returncode == 0, apply.stdout
    raise ValueError(f"Unsupported apply command: {apply_command}")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Iteratively repair one gLayout verification case with Qwen.")
    parser.add_argument("--case", default="diff_pair_ibias")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--workspace-mode", choices=("worktree", "copy"), default="worktree")
    parser.add_argument("--apply-command", choices=("git", "patch"), default="git")
    parser.add_argument(
        "--stop-on-patch-failure",
        action="store_true",
        help="Exit immediately when the model emits a patch that cannot be applied.",
    )
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-drc-issues", type=int, default=24)
    parser.add_argument("--max-lvs-issues", type=int, default=48)
    parser.add_argument("--skill-file", default=str(EXPERIMENT_DIR / "glayout_skill_compact.md"))
    parser.add_argument("--skill-line-budget", type=int, default=1500)
    parser.add_argument("--packet-line-budget", type=int, default=1000)
    parser.add_argument("--max-source-spans", type=int, default=4)
    parser.add_argument("--source-lines-per-span", type=int, default=180)
    parser.add_argument("--verifier-log-lines", type=int, default=120)
    parser.add_argument("--api-base", default=os.environ.get("QWEN_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--endpoint-check-timeout", type=int, default=10)
    parser.add_argument(
        "--skip-endpoint-preflight",
        action="store_true",
        help="Skip /v1/models preflight for nonstandard OpenAI-compatible servers.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write prompt and stop before calling the model.")
    args = parser.parse_args()

    run_root = Path(args.run_root or (REPO_ROOT / "build" / "qwen_repair_loop" / _timestamp())).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not args.skip_endpoint_preflight:
        ok, message = check_model_endpoint(
            args.api_base,
            args.api_key,
            timeout=args.endpoint_check_timeout,
        )
        if not ok:
            error_path = run_root / "endpoint_error.txt"
            error_path.write_text(message + "\n\n" + endpoint_help(args.api_base, args.model) + "\n")
            print(f"[qwen-loop] model endpoint preflight failed; see {error_path}")
            print(message)
            print(endpoint_help(args.api_base, args.model))
            return 3
        print(f"[qwen-loop] {message}")
    workspace = prepare_workspace(REPO_ROOT, run_root, args.workspace_mode)
    skill_text = Path(args.skill_file).read_text()

    summary: dict[str, Any] = {
        "case": args.case,
        "run_root": str(run_root),
        "workspace": str(workspace),
        "workspace_mode": args.workspace_mode,
        "model": args.model,
        "iterations": [],
    }
    summary_path = run_root / "summary.json"

    for iteration in range(args.max_iters):
        iter_dir = run_root / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        verifier = run_verification(
            workspace,
            iter_dir,
            case=args.case,
            top_k=args.top_k,
            max_drc_issues=args.max_drc_issues,
            max_lvs_issues=args.max_lvs_issues,
        )
        case_result = _load_json(Path(verifier["case_result_path"]))
        locator = _load_json(Path(verifier["locator_path"]))
        packet = _load_json(Path(verifier["repair_packet_path"]))
        clean = _is_clean_case(case_result, locator)
        iteration_summary: dict[str, Any] = {
            "iteration": iteration,
            "verifier": verifier,
            "clean": clean,
            "patch_applied": False,
        }
        summary["iterations"].append(iteration_summary)
        write_summary(summary_path, summary)
        if clean:
            print(f"[qwen-loop] case {args.case} is clean at iteration {iteration}")
            return 0

        prompt = build_prompt(
            case=args.case,
            skill_text=skill_text,
            packet=packet,
            iterations=summary["iterations"][:-1],
            verifier_log=verifier["log_path"],
            skill_line_budget=args.skill_line_budget,
            packet_line_budget=args.packet_line_budget,
            max_source_spans=args.max_source_spans,
            source_lines_per_span=args.source_lines_per_span,
            verifier_log_lines=args.verifier_log_lines,
        )
        prompt_path = iter_dir / "prompt.md"
        prompt_path.write_text(prompt)
        iteration_summary["prompt_path"] = str(prompt_path)
        iteration_summary["prompt_line_count"] = len(prompt.splitlines())
        iteration_summary["prompt_char_count"] = len(prompt)
        if args.dry_run:
            print(f"[qwen-loop] dry-run prompt written to {prompt_path}")
            write_summary(summary_path, summary)
            return 0

        try:
            response = call_openai_compatible(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
            )
        except RuntimeError as exc:
            error_path = iter_dir / "model_error.txt"
            error_path.write_text(str(exc) + "\n")
            iteration_summary["model_error_path"] = str(error_path)
            iteration_summary["model_error"] = str(exc)
            write_summary(summary_path, summary)
            print(f"[qwen-loop] model call failed at iteration {iteration}; see {error_path}")
            print(str(exc))
            return 3
        response_path = iter_dir / "model_response.txt"
        response_path.write_text(response)
        raw_patch_text = extract_unified_diff(response)
        raw_patch_path = iter_dir / "model.raw.patch"
        raw_patch_path.write_text(raw_patch_text)
        patch_text = normalize_model_patch_paths(raw_patch_text, workspace)
        patch_path = iter_dir / "model.patch"
        patch_path.write_text(patch_text)
        ok, apply_log = apply_model_patch(workspace, patch_path, apply_command=args.apply_command)
        apply_log_path = iter_dir / "apply.log"
        apply_log_path.write_text(apply_log)
        iteration_summary.update(
            {
                "model_response_path": str(response_path),
                "model_raw_patch_path": str(raw_patch_path),
                "model_patch_path": str(patch_path),
                "apply_log_path": str(apply_log_path),
                "patch_applied": ok,
            }
        )
        write_summary(summary_path, summary)
        if not ok:
            print(f"[qwen-loop] patch failed to apply at iteration {iteration}; see {apply_log_path}")
            if args.stop_on_patch_failure:
                return 2
            print("[qwen-loop] continuing; the next prompt will include the failed patch and apply log")
            continue

    print(f"[qwen-loop] reached max iterations ({args.max_iters}) without a clean result")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
