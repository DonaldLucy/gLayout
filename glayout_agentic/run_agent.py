from __future__ import annotations

import argparse
import sys
from pathlib import Path

from workflow.agent import AgentRequest, GLayoutCodeAgent
from workflow.backends import BackendError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Natural-language or code-repair agent for gLayout generators."
    )
    parser.add_argument("--task", type=str, default=None, help="Natural-language task.")
    parser.add_argument(
        "--task-file",
        type=Path,
        default=None,
        help="Optional file containing the natural-language task.",
    )
    parser.add_argument(
        "--input-code",
        type=Path,
        default=None,
        help="Optional existing Python generator to repair.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "skill", "local-hf"],
        default="auto",
        help="Generation backend. `auto` prefers a deterministic skill when one matches.",
    )
    parser.add_argument(
        "--disable-skills",
        action="store_true",
        help="Disable all skill matching and skill hints so generation relies only on the selected LLM backend.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HF model name or local path for the local-hf backend.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional LoRA adapter path for the local-hf backend.",
    )
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-drc-lvs", action="store_true")
    parser.add_argument("--output-py", type=Path, default=None)
    parser.add_argument("--output-gds", type=Path, default=None)
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("glayout_agentic/runs")
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("glayout_agentic/data/training"),
    )
    parser.add_argument("--pdk-root", type=str, default=None)
    parser.add_argument("--pdk-path", type=str, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--no-record-training", action="store_true")
    return parser.parse_args()


def resolve_task(args: argparse.Namespace) -> str:
    task_parts: list[str] = []
    if args.task:
        task_parts.append(args.task)
    if args.task_file:
        task_parts.append(args.task_file.read_text(encoding="utf-8"))
    return "\n\n".join(part.strip() for part in task_parts if part and part.strip())


def main() -> int:
    args = parse_args()
    task = resolve_task(args)
    input_code = (
        args.input_code.read_text(encoding="utf-8") if args.input_code else None
    )
    if not task and input_code is None:
        print("Provide at least --task/--task-file or --input-code.", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    asset_root = Path(__file__).resolve().parent
    agent = GLayoutCodeAgent(repo_root=repo_root, asset_root=asset_root)

    request = AgentRequest(
        task=task,
        input_code=input_code,
        backend=args.backend,
        disable_skills=args.disable_skills,
        model_name=args.model,
        adapter_path=args.adapter_path,
        max_attempts=args.max_attempts,
        execute=args.execute,
        run_drc_lvs=args.run_drc_lvs,
        output_py=args.output_py,
        output_gds=args.output_gds,
        runs_dir=args.runs_dir,
        dataset_dir=args.dataset_dir,
        pdk_root=args.pdk_root,
        pdk_path=args.pdk_path,
        load_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_sec=args.timeout_sec,
        record_training=not args.no_record_training,
    )

    try:
        result = agent.run(request)
    except (BackendError, ValueError) as exc:
        print(f"Agent failed before validation: {exc}", file=sys.stderr)
        return 1

    print(f"success: {result.success}")
    print(f"backend: {result.backend_used}")
    print(f"run_dir: {result.run_dir}")
    print(f"python_file: {result.final_python_path}")
    if result.final_gds_path:
        print(f"gds_file: {result.final_gds_path}")
    if result.best_partial_python_path:
        print(f"best_partial_python_file: {result.best_partial_python_path}")
    if result.best_partial_gds_path:
        print(f"best_partial_gds_file: {result.best_partial_gds_path}")
    if result.skill_name:
        print(f"skill: {result.skill_name}")
    if result.drc_pass is not None:
        print(f"drc_pass: {result.drc_pass}")
    if result.lvs_pass is not None:
        print(f"lvs_pass: {result.lvs_pass}")
    print(f"attempts: {result.attempts}")
    print(f"message: {result.message}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
