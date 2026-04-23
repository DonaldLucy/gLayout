from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflow.agent import AgentRequest, GLayoutCodeAgent
from workflow.backends import BackendError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent local-HF session for gLayout coding tasks."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HF model name or local path.",
    )
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--disable-skills", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-drc-lvs", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--pdk-root", type=str, default=None)
    parser.add_argument("--pdk-path", type=str, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("glayout_agentic/runs"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("glayout_agentic/data/training"),
    )
    parser.add_argument("--timeout-sec", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    asset_root = Path(__file__).resolve().parent
    agent = GLayoutCodeAgent(repo_root=repo_root, asset_root=asset_root)

    print("Persistent gLayout agent session ready.")
    print("Commands:")
    print("  /exit                exit the session")
    print("  /status              print current defaults")
    print("  /json {..}           run a request from a JSON object with a `task` field")
    print("Any other line is treated as a task string.")

    while True:
        try:
            line = input("task> ").strip()
        except EOFError:
            print()
            break

        if not line:
            continue
        if line in {"/exit", "exit", "quit"}:
            break
        if line == "/status":
            print(
                json.dumps(
                    {
                        "model": args.model,
                        "disable_skills": args.disable_skills,
                        "execute": args.execute,
                        "run_drc_lvs": args.run_drc_lvs,
                        "max_attempts": args.max_attempts,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    indent=2,
                )
            )
            continue

        task = line
        input_code = None
        if line.startswith("/json "):
            payload = json.loads(line[6:])
            task = payload["task"]
            input_code = payload.get("input_code")

        request = AgentRequest(
            task=task,
            input_code=input_code,
            backend="local-hf",
            disable_skills=args.disable_skills,
            model_name=args.model,
            adapter_path=args.adapter_path,
            max_attempts=args.max_attempts,
            execute=args.execute,
            run_drc_lvs=args.run_drc_lvs,
            runs_dir=args.runs_dir,
            dataset_dir=args.dataset_dir,
            pdk_root=args.pdk_root,
            pdk_path=args.pdk_path,
            load_in_4bit=not args.no_4bit,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout_sec=args.timeout_sec,
        )
        try:
            result = agent.run(request)
        except (BackendError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"error: {exc}")
            continue

        print(f"success: {result.success}")
        print(f"run_dir: {result.run_dir}")
        print(f"python_file: {result.final_python_path}")
        if result.final_gds_path:
            print(f"gds_file: {result.final_gds_path}")
        if result.drc_pass is not None:
            print(f"drc_pass: {result.drc_pass}")
        if result.lvs_pass is not None:
            print(f"lvs_pass: {result.lvs_pass}")
        print(f"attempts: {result.attempts}")
        print(f"message: {result.message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
