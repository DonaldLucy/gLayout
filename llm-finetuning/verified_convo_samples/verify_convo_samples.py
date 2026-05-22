#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(THIS_DIR))


def _candidate_pdk_roots() -> list[Path]:
    candidates: list[Path] = []
    for raw in [
        os.environ.get("PDK_ROOT"),
        os.environ.get("MAGIC_PDK_ROOT"),
        os.environ.get("NETGEN_PDK_ROOT"),
        os.environ.get("PDKPATH"),
        str(Path(os.environ.get("CONDA_PREFIX", "")) / "share" / "pdk") if os.environ.get("CONDA_PREFIX") else None,
        "/foss/pdks",
        "/headless/conda-env/miniconda3/share/pdk",
    ]:
        if raw and raw != "None":
            path = Path(raw).resolve()
            candidates.append(path.parent if path.name == "sky130A" else path)

    volare_root = Path("/foss/pdks")
    if volare_root.exists():
        for magicrc in sorted(volare_root.glob("**/sky130A/libs.tech/magic/sky130A.magicrc")):
            candidates.append(magicrc.parents[3])

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _resolve_pdk_paths(override: str | None = None) -> dict[str, Path]:
    candidates = [Path(override).resolve()] if override else _candidate_pdk_roots()
    lvs_ref = REPO_ROOT / "src" / "glayout" / "pdk" / "sky130_mapped" / "sky130_fd_sc_hd.spice"
    for pdk_root in candidates:
        sky130_dir = pdk_root if pdk_root.name == "sky130A" else pdk_root / "sky130A"
        root = pdk_root.parent if pdk_root.name == "sky130A" else pdk_root
        magicrc = sky130_dir / "libs.tech" / "magic" / "sky130A.magicrc"
        lvs_setup = sky130_dir / "libs.tech" / "netgen" / "sky130A_setup.tcl"
        if magicrc.is_file() and lvs_setup.is_file() and lvs_ref.is_file():
            os.environ["PDK_ROOT"] = str(root)
            os.environ["PDKPATH"] = str(sky130_dir)
            os.environ["PDK"] = "sky130A"
            os.environ["MAGIC_PDK_ROOT"] = str(root)
            os.environ["NETGEN_PDK_ROOT"] = str(root)
            return {"pdk_root": root, "magicrc": magicrc, "lvs_setup": lvs_setup, "lvs_ref": lvs_ref}
    raise FileNotFoundError("Could not locate sky130A magic/netgen files. Checked: " + ", ".join(str(p) for p in candidates))


def _strict_magic_report(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise AssertionError(f"Magic DRC report missing: {report_path}")
    content = report_path.read_text()
    count_match = re.search(r"count:\s*(\d+)\s*$", content, re.IGNORECASE | re.MULTILINE)
    count = int(count_match.group(1)) if count_match else None
    coordinate_errors = len(re.findall(r"\d+\.\d+um\s+\d+\.\d+um\s+\d+\.\d+um\s+\d+\.\d+um", content))
    clean = count == 0 or coordinate_errors == 0 or "No errors found." in content
    return {"report": str(report_path), "is_clean": clean, "error_count": 0 if clean else count}


def _strict_lvs_report(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise AssertionError(f"Netgen LVS report missing: {report_path}")
    content = report_path.read_text()
    matched = "Circuits match uniquely" in content
    property_error = "Property errors were found." in content
    return {
        "report": str(report_path),
        "is_clean": matched and not property_error,
        "matched": matched,
        "property_error": property_error,
    }


def _run_drc(component: Any, design_name: str, case_dir: Path, paths: dict[str, Path]) -> dict[str, Any]:
    from glayout import sky130

    output_dir = case_dir / "magic_drc"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    sky130.drc_magic(
        component,
        design_name,
        pdk_root=paths["pdk_root"],
        magic_drc_file=paths["magicrc"],
        output_file=output_dir,
    )
    return _strict_magic_report(output_dir / "drc" / design_name / f"{design_name}.rpt")


def _run_lvs(component: Any, design_name: str, case_dir: Path, paths: dict[str, Path]) -> dict[str, Any]:
    from glayout import sky130

    output_dir = case_dir / "netgen_lvs"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    sky130.lvs_netgen(
        component,
        design_name,
        pdk_root=paths["pdk_root"],
        lvs_setup_tcl_file=paths["lvs_setup"],
        lvs_schematic_ref_file=paths["lvs_ref"],
        output_file_path=output_dir,
        copy_intermediate_files=True,
    )
    return _strict_lvs_report(output_dir / "lvs" / design_name / f"{design_name}_lvs.rpt")


def _build_traced(builder, pdk):
    from glayout import enable_source_mapping, reset_source_mapping
    from gdsfactory.cell import clear_cache

    clear_cache()
    reset_source_mapping()
    enable_source_mapping(reset=True, auto_emit_sidecar=True)
    return builder(pdk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify generated OpenFASOC convo gLayout samples.")
    parser.add_argument("--output-dir", default="build/verified_convo_samples")
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--pdk-root", default=None)
    parser.add_argument("--skip-drc", action="store_true")
    parser.add_argument("--skip-lvs", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    paths = _resolve_pdk_paths(args.pdk_root)
    from glayout import sky130
    from convo_layouts.generated import SAMPLE_BUILDERS

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = args.samples or sorted(SAMPLE_BUILDERS)
    summary: dict[str, Any] = {"pdk_root": str(paths["pdk_root"]), "samples": {}, "failures": {}}

    for sample in selected:
        if sample not in SAMPLE_BUILDERS:
            raise KeyError(f"Unknown sample {sample}. Known: {sorted(SAMPLE_BUILDERS)}")
        case_id = sample.lower()
        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            component = _build_traced(SAMPLE_BUILDERS[sample], sky130)
            gds_path = case_dir / f"{case_id}.gds"
            component.write_gds(str(gds_path))
            sidecar = gds_path.with_suffix(".provenance.json")
            result: dict[str, Any] = {
                "gds": str(gds_path),
                "sidecar": str(sidecar),
                "sidecar_exists": sidecar.is_file(),
            }
            if not args.skip_drc:
                result["drc"] = _run_drc(component, case_id, case_dir, paths)
            if not args.skip_lvs:
                result["lvs"] = _run_lvs(component, case_id, case_dir, paths)
            result["is_clean"] = (
                result.get("sidecar_exists", False)
                and (args.skip_drc or result["drc"]["is_clean"])
                and (args.skip_lvs or result["lvs"]["is_clean"])
            )
            summary["samples"][sample] = result
            print(f"[{'PASS' if result['is_clean'] else 'FAIL'}] {sample}")
        except Exception as exc:
            summary["failures"][sample] = str(exc)
            (case_dir / "error.txt").write_text(str(exc))
            print(f"[ERROR] {sample}: {exc}")
            if not args.continue_on_error:
                raise

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary written to {summary_path}")
    return 1 if summary["failures"] or any(not item.get("is_clean", False) for item in summary["samples"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
