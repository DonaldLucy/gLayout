from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from glayout.pdk.sky130_mapped import sky130_mapped_pdk
from glayout.cells.elementary.current_mirror.current_mirror import current_mirror
from glayout.cells.elementary.diff_pair.diff_pair import diff_pair
from glayout.cells.elementary.FVF.fvf import flipped_voltage_follower
from glayout_agentic.examples.two_fet_shared_diffusion import (
    build_two_fet_shared_diffusion,
)
from glayout_agentic.workflow.validator import build_runtime_env, validate_generated_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a merge-diffusion-oriented curriculum over repo cells."
    )
    parser.add_argument("--run-drc-lvs", action="store_true")
    parser.add_argument("--pdk-root", type=str, default=None)
    parser.add_argument("--pdk-path", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("glayout_agentic/curriculum_runs"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdk = sky130_mapped_pdk
    if pdk is None:
        raise RuntimeError("sky130_mapped_pdk is unavailable.")
    pdk.activate()

    env = build_runtime_env(REPO_ROOT, pdk_root=args.pdk_root, pdk_path=args.pdk_path)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    stages = [
        (
            "01_explicit_two_fet_shared_diffusion",
            "Direct shared-diffusion primitive; best first benchmark.",
            REPO_ROOT / "glayout_agentic" / "examples" / "two_fet_shared_diffusion.py",
            lambda: build_two_fet_shared_diffusion(width=2.0, length=0.15, fingers=1),
        ),
        (
            "02_current_mirror_nfet",
            "Strong next benchmark: built from two-transistor interdigitized placement.",
            REPO_ROOT / "src" / "glayout" / "cells" / "elementary" / "current_mirror" / "current_mirror.py",
            lambda: current_mirror(
                pdk,
                numcols=2,
                device="nfet",
                with_substrate_tap=False,
                width=2.0,
                length=0.15,
                fingers=1,
            ),
        ),
        (
            "03_current_mirror_pfet",
            "Same topology family as above, but PFET flavor.",
            REPO_ROOT / "src" / "glayout" / "cells" / "elementary" / "current_mirror" / "current_mirror.py",
            lambda: current_mirror(
                pdk,
                numcols=2,
                device="pfet",
                with_substrate_tap=False,
                width=2.0,
                length=0.15,
                fingers=1,
            ),
        ),
        (
            "04_diff_pair_common_centroid",
            "Relevant but harder: strong symmetry/common-centroid test, not the cleanest merge-diffusion benchmark.",
            REPO_ROOT / "src" / "glayout" / "cells" / "elementary" / "diff_pair" / "diff_pair.py",
            lambda: diff_pair(
                pdk,
                width=2.0,
                length=0.15,
                fingers=2,
                n_or_p_fet=True,
                substrate_tap=False,
            ),
        ),
        (
            "05_fvf_reference",
            "Useful routing benchmark, but not primarily a merge-diffusion topology.",
            REPO_ROOT / "src" / "glayout" / "cells" / "elementary" / "FVF" / "fvf.py",
            lambda: flipped_voltage_follower(
                pdk,
                width=(2.0, 1.0),
                length=(0.15, 0.15),
                fingers=(1, 1),
                multipliers=(1, 1),
                sd_rmult=1,
            ),
        ),
    ]

    summary: list[dict[str, object]] = []
    for stage_name, note, reference_file, builder in stages:
        component = builder()
        gds_path = run_dir / f"{stage_name}.gds"
        component.write_gds(gds_path)
        validation = validate_generated_file(
            repo_root=REPO_ROOT,
            python_file=reference_file,
            execute=True,
            gds_output=gds_path,
            env=env,
            run_drc_lvs=args.run_drc_lvs,
        )
        summary.append(
            {
                "stage": stage_name,
                "note": note,
                "reference_file": str(reference_file),
                "gds_path": str(gds_path),
                "success": validation.success,
                "summary": validation.summary,
                "drc_pass": validation.drc_pass,
                "lvs_pass": validation.lvs_pass,
            }
        )

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
