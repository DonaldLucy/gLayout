#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(errors="replace")


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def sha256_short(text: str) -> str | None:
    if not text.strip():
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def normalize_repo_path(path_text: str | None, workspace: Path | None = None) -> str | None:
    if not path_text:
        return None
    path = str(path_text).replace("\\", "/")
    for prefix in ("a/", "b/", "./"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    if workspace is not None:
        workspace_path = workspace.resolve().as_posix()
        if path.startswith(f"{workspace_path}/"):
            path = path[len(workspace_path) + 1 :]
    if "/workspace/" in path:
        path = path.split("/workspace/", 1)[1]
    if path.startswith("workspace/"):
        path = path[len("workspace/") :]
    for root in ("src/", "scripts/", "tests/", "experiments/"):
        marker = f"/{root}"
        if marker in path:
            return root + path.split(marker, 1)[1]
        if path.startswith(root):
            return path
    return path if "/" in path else None


def source_span_files(packet: dict[str, Any] | None) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for span in (packet or {}).get("source_spans", []):
        path = normalize_repo_path(span.get("file"))
        if path and path not in seen:
            files.append(path)
            seen.add(path)
    return files


def changed_files_from_patch(patch_text: str) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"^diff --git\s+a/(.*?)\s+b/(.*?)$", patch_text, flags=re.MULTILINE):
        path = normalize_repo_path(match.group(2))
        if path:
            files.append(path)
    return files


def patch_line_stats(patch_text: str) -> dict[str, int]:
    added = 0
    deleted = 0
    hunks = 0
    for line in patch_text.splitlines():
        if line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
    return {"added_lines": added, "deleted_lines": deleted, "hunks": hunks}


def categorize_apply_failure(apply_log: str) -> str:
    text = apply_log.lower()
    if not text.strip():
        return "none_or_unknown"
    if "patch quality guard failed" in text:
        return "quality_guard"
    if "empty patch" in text:
        return "empty_patch"
    if "no such file or directory" in text or "can't find file to patch" in text:
        return "path_error"
    if "corrupt patch" in text or "malformed patch" in text:
        return "malformed_patch"
    if "patch does not apply" in text or "hunk" in text and "failed" in text:
        return "context_mismatch"
    if "already exists" in text:
        return "file_exists"
    return "other_apply_error"


def added_lines_already_present(patch_text: str, workspace: Path) -> dict[str, Any]:
    files = changed_files_from_patch(patch_text)
    total_added = 0
    already_present = 0
    per_file: list[dict[str, Any]] = []
    for rel in files:
        source = read_text(workspace / rel)
        if not source:
            continue
        source_lines = set(source.splitlines())
        file_added = 0
        file_present = 0
        in_file = False
        for line in patch_text.splitlines():
            if line.startswith("diff --git "):
                in_file = line.endswith(f" b/{rel}")
            elif in_file and line.startswith("+") and not line.startswith("+++"):
                value = line[1:]
                if not value.strip():
                    continue
                file_added += 1
                if value in source_lines:
                    file_present += 1
        total_added += file_added
        already_present += file_present
        if file_added:
            per_file.append(
                {
                    "file": rel,
                    "added_nonblank": file_added,
                    "already_present": file_present,
                    "already_present_fraction": round(file_present / file_added, 3),
                }
            )
    fraction = round(already_present / total_added, 3) if total_added else 0.0
    return {
        "added_nonblank": total_added,
        "already_present": already_present,
        "already_present_fraction": fraction,
        "per_file": per_file,
    }


def detect_iteration_issues(
    *,
    response: str,
    raw_patch: str,
    patch: str,
    apply_log: str,
    workspace: Path,
    packet: dict[str, Any] | None,
    seen_hashes: Counter[str],
) -> list[str]:
    issues: list[str] = []
    if "```" in response:
        issues.append("markdown_fence_in_response")
    if response.strip() and not response.lstrip().startswith(("diff --git", "--- ")):
        issues.append("response_has_non_diff_wrapper")
    if raw_patch != patch:
        issues.append("patch_path_normalized_by_loop")
    if "workspace/" in raw_patch:
        issues.append("raw_patch_used_workspace_prefix")
    if re.search(r"index\s+abcdef|ghijkl", patch):
        issues.append("placeholder_git_index")
    failure = categorize_apply_failure(apply_log)
    if failure != "none_or_unknown":
        issues.append(f"apply_failure:{failure}")
    allowed_files = source_span_files(packet)
    changed_files = changed_files_from_patch(patch)
    outside = [path for path in changed_files if allowed_files and path not in allowed_files]
    if outside:
        issues.append("changed_files_outside_source_spans")
    duplicate_stats = added_lines_already_present(patch, workspace)
    if duplicate_stats["added_nonblank"] and duplicate_stats["already_present_fraction"] >= 0.65:
        issues.append("patch_mostly_repeats_existing_source")
    patch_hash = sha256_short(patch)
    if patch_hash and seen_hashes[patch_hash] > 0:
        issues.append("repeated_identical_patch")
    return issues


def summarize_iteration(
    iter_dir: Path,
    workspace: Path,
    seen_hashes: Counter[str],
    *,
    packet: dict[str, Any] | None,
) -> dict[str, Any]:
    response = read_text(iter_dir / "model_response.txt")
    raw_patch = read_text(iter_dir / "model.raw.patch")
    patch = read_text(iter_dir / "model.patch")
    apply_log = read_text(iter_dir / "apply.log")
    patch_hash = sha256_short(patch)
    raw_hash = sha256_short(raw_patch)
    duplicate_stats = added_lines_already_present(patch, workspace)
    issues = detect_iteration_issues(
        response=response,
        raw_patch=raw_patch,
        patch=patch,
        apply_log=apply_log,
        workspace=workspace,
        packet=packet,
        seen_hashes=seen_hashes,
    )
    if patch_hash:
        seen_hashes[patch_hash] += 1
    return {
        "iteration": int(iter_dir.name.split("_")[-1]),
        "prompt_lines": len(read_text(iter_dir / "prompt.md").splitlines()),
        "response_chars": len(response),
        "response_has_diff_fence": "```" in response,
        "raw_patch_hash": raw_hash,
        "patch_hash": patch_hash,
        "raw_patch_bytes": len(raw_patch.encode("utf-8")),
        "patch_bytes": len(patch.encode("utf-8")),
        "path_normalized": raw_patch != patch,
        "changed_files": changed_files_from_patch(patch),
        "source_span_files": source_span_files(packet),
        "changed_files_outside_source_spans": [
            path for path in changed_files_from_patch(patch) if source_span_files(packet) and path not in source_span_files(packet)
        ],
        "patch_line_stats": patch_line_stats(patch),
        "apply_failure_category": categorize_apply_failure(apply_log),
        "apply_log_excerpt": apply_log.strip()[:1000],
        "added_lines_already_present": duplicate_stats,
        "issues": issues,
    }


def training_hooks(iterations: list[dict[str, Any]]) -> list[str]:
    issue_counts = Counter(issue for item in iterations for issue in item["issues"])
    hooks: list[str] = []
    if issue_counts["raw_patch_used_workspace_prefix"]:
        hooks.append(
            "Path discipline SFT: train outputs to use repo-root paths like `src/...`, never run-root or `workspace/...` paths."
        )
    if issue_counts["markdown_fence_in_response"] or issue_counts["response_has_non_diff_wrapper"]:
        hooks.append(
            "Diff-only formatting SFT: include negative examples where Markdown fences/prose are rejected, and target output is a bare unified diff."
        )
    if issue_counts["apply_failure:context_mismatch"]:
        hooks.append(
            "Current-source grounding SFT: train model to edit only lines present in provided source spans, not remembered/stale gLayout code."
        )
    if issue_counts["changed_files_outside_source_spans"]:
        hooks.append(
            "File-scope SFT: train model to stay inside repair_packet source spans unless explicitly justified by the packet."
        )
    if issue_counts["apply_failure:quality_guard"]:
        hooks.append(
            "Patch-quality SFT: reject duplicate/stale edits and require patches that pass pre-apply guards before verification."
        )
    if issue_counts["patch_mostly_repeats_existing_source"]:
        hooks.append(
            "Duplicate-edit rejection: add examples where proposed edits are already present and the correct next action is to inspect LVS-specific nets/labels instead."
        )
    if issue_counts["repeated_identical_patch"]:
        hooks.append(
            "Iteration-awareness SFT: condition on previous apply logs and require a materially different patch after a failed attempt."
        )
    if not hooks:
        hooks.append(
            "No dominant patch-format failure found. Inspect verification logs and model patches for domain-level LVS reasoning failures."
        )
    return hooks


def write_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Qwen Repair Run Analysis")
    lines.append("")
    lines.append(f"- Run root: `{report['run_root']}`")
    lines.append(f"- Workspace: `{report['workspace']}`")
    lines.append(f"- Iterations: {len(report['iterations'])}")
    lines.append("")
    lines.append("## Failure Counts")
    for key, value in report["failure_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Issue Counts")
    for key, value in report["issue_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Training Hooks")
    for hook in report["training_hooks"]:
        lines.append(f"- {hook}")
    lines.append("")
    lines.append("## Iterations")
    for item in report["iterations"]:
        lines.append(f"### Iteration {item['iteration']:02d}")
        lines.append(f"- Changed files: {', '.join(item['changed_files']) or 'none'}")
        if item.get("changed_files_outside_source_spans"):
            lines.append(
                "- Outside source spans: "
                + ", ".join(item["changed_files_outside_source_spans"])
            )
        lines.append(f"- Patch hash: `{item['patch_hash']}`")
        lines.append(f"- Apply failure: `{item['apply_failure_category']}`")
        lines.append(f"- Path normalized: {item['path_normalized']}")
        lines.append(f"- Patch stats: {item['patch_line_stats']}")
        lines.append(f"- Existing-line overlap: {item['added_lines_already_present']['already_present_fraction']}")
        lines.append(f"- Issues: {', '.join(item['issues']) or 'none'}")
        if item["apply_log_excerpt"]:
            lines.append("")
            lines.append("```text")
            lines.append(item["apply_log_excerpt"])
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def analyze_run(run_root: Path) -> dict[str, Any]:
    summary = load_json(run_root / "summary.json") or {}
    workspace = Path(summary.get("workspace") or run_root / "workspace")
    seen_hashes: Counter[str] = Counter()
    summary_iterations = {
        int(item.get("iteration")): item
        for item in summary.get("iterations", [])
        if item.get("iteration") is not None
    }
    iterations = []
    for iter_dir in sorted(run_root.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        iteration = int(iter_dir.name.split("_")[-1])
        repair_packet_path = (
            summary_iterations.get(iteration, {})
            .get("verifier", {})
            .get("repair_packet_path")
        )
        packet = load_json(Path(repair_packet_path)) if repair_packet_path else None
        iterations.append(summarize_iteration(iter_dir, workspace, seen_hashes, packet=packet))
    failure_counts = Counter(item["apply_failure_category"] for item in iterations)
    issue_counts = Counter(issue for item in iterations for issue in item["issues"])
    return {
        "run_root": str(run_root),
        "workspace": str(workspace),
        "summary": summary,
        "failure_counts": dict(failure_counts),
        "issue_counts": dict(issue_counts),
        "training_hooks": training_hooks(iterations),
        "iterations": iterations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a Qwen repair-loop run for debugging and SFT planning.")
    parser.add_argument("run_root", help="Path such as /tmp/qwen_repair_loop_diff_pair_ibias_v4")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    report = analyze_run(run_root)
    json_out = Path(args.json_out) if args.json_out else run_root / "analysis.json"
    md_out = Path(args.md_out) if args.md_out else run_root / "analysis.md"
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    md_out.write_text(write_markdown(report))
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    print(write_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
