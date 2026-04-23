from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from glayout_agentic.examples.two_fet_interdigitized import build_two_fet_interdigitized
from glayout_agentic.examples.two_fet_separate import build_two_fet_separate
from glayout_agentic.examples.two_fet_shared_diffusion import (
    build_two_fet_shared_diffusion,
)
from glayout_agentic.workflow.validator import build_runtime_env, validate_generated_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Progressive hello-world flow: separate -> interdigitized -> merged diffusion -> repair."
    )
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--length", type=float, default=0.15)
    parser.add_argument("--fingers", type=int, default=1)
    parser.add_argument("--run-drc-lvs", action="store_true")
    parser.add_argument("--run-repair-demo", action="store_true")
    parser.add_argument("--pdk-root", type=str, default=None)
    parser.add_argument("--pdk-path", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("glayout_agentic/progression_runs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)

    env = build_runtime_env(REPO_ROOT, pdk_root=args.pdk_root, pdk_path=args.pdk_path)
    summaries: list[dict[str, object]] = []

    stages = [
        (
            "01_two_fet_separate",
            REPO_ROOT / "glayout_agentic" / "examples" / "two_fet_separate.py",
            build_two_fet_separate,
        ),
        (
            "02_two_fet_interdigitized",
            REPO_ROOT / "glayout_agentic" / "examples" / "two_fet_interdigitized.py",
            build_two_fet_interdigitized,
        ),
        (
            "03_two_fet_explicit_merged_diffusion",
            REPO_ROOT / "glayout_agentic" / "examples" / "two_fet_shared_diffusion.py",
            build_two_fet_shared_diffusion,
        ),
    ]

    for stage_name, source_path, builder in stages:
        python_path = out_dir / f"{stage_name}.py"
        if source_path.exists():
            shutil.copy2(source_path, python_path)
        else:
            python_path.write_text(
                f"# Source file copied from repo examples is unavailable for {stage_name}.\n",
                encoding="utf-8",
            )
        component = builder(width=args.width, length=args.length, fingers=args.fingers)
        gds_path = out_dir / f"{stage_name}.gds"
        component.write_gds(gds_path)
        validation = validate_generated_file(
            repo_root=REPO_ROOT,
            python_file=python_path,
            execute=True,
            gds_output=gds_path,
            env=env,
            run_drc_lvs=args.run_drc_lvs,
        )
        summaries.append(
            {
                "stage": stage_name,
                "gds_path": str(gds_path),
                "success": validation.success,
                "summary": validation.summary,
                "drc_pass": validation.drc_pass,
                "lvs_pass": validation.lvs_pass,
            }
        )

    if args.run_repair_demo:
        broken_path = out_dir / "04_broken_merged_diffusion.py"
        broken_source = (REPO_ROOT / "glayout_agentic" / "examples" / "two_fet_shared_diffusion.py").read_text(
            encoding="utf-8"
        )
        broken_source = broken_source.replace(
            "from glayout.pdk.sky130_mapped import sky130_mapped_pdk",
            "from glayout.pdk.sky130_mapped import sky130_mapped_pdk_broken",
            1,
        )
        broken_path.write_text(broken_source, encoding="utf-8")

        repair_cmd = [
            sys.executable,
            str(REPO_ROOT / "glayout_agentic" / "run_agent.py"),
            "--input-code",
            str(broken_path),
            "--task",
            "Repair this two FET merged diffusion SKY130 generator and keep the merged diffusion topology.",
            "--backend",
            "auto",
            "--execute",
            "--output-py",
            str(out_dir / "04_repaired_merged_diffusion.py"),
            "--output-gds",
            str(out_dir / "04_repaired_merged_diffusion.gds"),
        ]
        if args.run_drc_lvs:
            repair_cmd.append("--run-drc-lvs")
        if args.pdk_root:
            repair_cmd.extend(["--pdk-root", args.pdk_root])
        if args.pdk_path:
            repair_cmd.extend(["--pdk-path", args.pdk_path])

        repair_proc = subprocess.run(
            repair_cmd,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        summaries.append(
            {
                "stage": "04_repair_demo",
                "success": repair_proc.returncode == 0,
                "stdout": repair_proc.stdout,
                "stderr": repair_proc.stderr,
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
