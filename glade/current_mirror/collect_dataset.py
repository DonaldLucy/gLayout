#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

# Allow direct execution via:
#   python glade/current_mirror/collect_dataset.py
# in addition to:
#   python -m glade.current_mirror.collect_dataset
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from glade.common.env import prepare_chipathon_environment, repo_root_from_file
from glade.common.reports import summarize_drc_report, summarize_log, summarize_lvs_report
from glade.common.schema import flatten_entry
from glade.current_mirror.parameter_space import iter_current_mirror_space, total_combinations


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parse_simple_parasitics(spice_path: Path) -> tuple[float, float]:
    total_resistance = 0.0
    total_capacitance = 0.0
    if not spice_path.exists():
        return total_resistance, total_capacitance

    with open(spice_path, "r") as f:
        for line in f:
            orig_line = line.strip()
            parts = orig_line.split()
            if not parts:
                continue
            name = parts[0].upper()
            if name.startswith("R") and len(parts) >= 4:
                try:
                    total_resistance += float(parts[3])
                except ValueError:
                    continue
            elif name.startswith("C") and len(parts) >= 4:
                value = parts[3]
                try:
                    unit = value[-1]
                    base = float(value[:-1])
                    scale = {
                        "f": 1e-15,
                        "F": 1e-15,
                        "p": 1e-12,
                        "P": 1e-12,
                        "n": 1e-9,
                        "N": 1e-9,
                        "u": 1e-6,
                        "U": 1e-6,
                    }.get(unit, None)
                    total_capacitance += base * scale if scale is not None else float(value)
                except ValueError:
                    continue
    return total_resistance, total_capacitance


def compute_geometry(component) -> dict:
    from glayout.verification.physical_features import calculate_area, calculate_symmetry_scores

    area = calculate_area(component)
    sym_h, sym_v = calculate_symmetry_scores(component)
    return {
        "area_um2": area,
        "sym_h": sym_h,
        "sym_v": sym_v,
    }


def evaluate_current_mirror(sample_id: str, params: dict, sample_dir: Path, paths: dict) -> dict:
    from glayout import sky130
    from glayout.cells.elementary.current_mirror.current_mirror import current_mirror, add_cm_labels

    artifact_dir = sample_dir / "artifacts"
    log_dir = sample_dir / "logs"
    report_dir = sample_dir / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "sample_id": sample_id,
        "generator_id": "current_mirror",
        "pdk": "sky130",
        "params": params,
        "source_netlist": None,
        "build_status": "not_run",
        "drc": {"status": "not_run"},
        "lvs": {"status": "not_run"},
        "pex": {"status": "not_run"},
        "geom": {},
        "runtime_s": {},
        "failure_tags": [],
        "structure_features": {
            "generator_family": "elementary",
            "device_is_pfet": int(params["device"] == "pfet"),
            "device_is_nfet": int(params["device"] == "nfet"),
            "numcols": params["numcols"],
            "aspect_ratio": params["width"] / params["length"],
            "estimated_main_devices": 2 * params["numcols"],
            "estimated_dummy_devices": 2,
        },
        "logs": {},
        "artifacts": {},
    }

    build_start = time.time()
    try:
        comp = add_cm_labels(
            current_mirror(
                sky130,
                device=params["device"],
                numcols=params["numcols"],
                width=params["width"],
                length=params["length"],
            ),
            sky130,
        )
        comp.name = sample_id.upper()
        netlist = comp.info["netlist"]
        entry["source_netlist"] = (
            netlist.generate_netlist() if hasattr(netlist, "generate_netlist") else str(netlist)
        )
        gds_path = artifact_dir / f"{sample_id}.gds"
        comp.write_gds(gds_path)
        entry["artifacts"]["gds"] = str(gds_path)
        entry["geom"] = compute_geometry(comp)
        entry["build_status"] = "success"
    except Exception:
        entry["build_status"] = "fail"
        entry["failure_tags"].append("build_fail")
        entry["logs"]["build"] = traceback.format_exc()
        (log_dir / "build.log").write_text(entry["logs"]["build"])
        entry["runtime_s"]["build"] = round(time.time() - build_start, 4)
        return entry

    entry["runtime_s"]["build"] = round(time.time() - build_start, 4)

    drc_start = time.time()
    drc_buffer = io.StringIO()
    try:
        with redirect_stdout(drc_buffer), redirect_stderr(drc_buffer):
            drc_result = sky130.drc_magic(
                entry["artifacts"]["gds"],
                sample_id.upper(),
                pdk_root=paths["pdk_root"],
                magic_drc_file=paths["magicrc"],
                output_file=report_dir,
            )
        drc_report = report_dir / "drc" / sample_id.upper() / f"{sample_id.upper()}.rpt"
        entry["drc"] = {
            **summarize_drc_report(drc_report),
            "result": drc_result,
        }
        entry["logs"]["drc"] = summarize_log(drc_buffer.getvalue())
        (log_dir / "drc.log").write_text(drc_buffer.getvalue())
        if not entry["drc"]["clean"]:
            entry["failure_tags"].append("drc_fail")
            for rule, count in entry["drc"].get("rule_counts", {}).items():
                if count > 0:
                    entry["failure_tags"].append(f"drc_{rule}")
    except Exception:
        entry["drc"] = {"status": "error", "clean": False}
        entry["failure_tags"].append("drc_fail")
        raw = traceback.format_exc() + "\n" + drc_buffer.getvalue()
        entry["logs"]["drc"] = summarize_log(raw)
        (log_dir / "drc.log").write_text(raw)
    entry["runtime_s"]["drc"] = round(time.time() - drc_start, 4)

    lvs_start = time.time()
    lvs_buffer = io.StringIO()
    try:
        with redirect_stdout(lvs_buffer), redirect_stderr(lvs_buffer):
            lvs_result = sky130.lvs_netgen(
                layout=comp,
                design_name=sample_id.upper(),
                pdk_root=paths["pdk_root"],
                magic_drc_file=paths["magicrc"],
                lvs_setup_tcl_file=paths["lvs_setup"],
                lvs_schematic_ref_file=paths["lvs_ref"],
                output_file_path=report_dir,
                copy_intermediate_files=True,
            )
        lvs_report = report_dir / "lvs" / sample_id.upper() / f"{sample_id.upper()}_lvs.rpt"
        entry["lvs"] = {
            **summarize_lvs_report(lvs_report),
            "result": lvs_result,
        }
        entry["logs"]["lvs"] = summarize_log(lvs_buffer.getvalue())
        (log_dir / "lvs.log").write_text(lvs_buffer.getvalue())
        if entry["lvs"]["status"] == "property_error":
            entry["failure_tags"].append("lvs_property_error")
        elif entry["lvs"]["status"] != "clean":
            entry["failure_tags"].append("lvs_topology_mismatch")
        entry["artifacts"]["lvs_dir"] = str(report_dir / "lvs" / sample_id.upper())
    except Exception:
        entry["lvs"] = {"status": "error", "clean": False}
        entry["failure_tags"].append("lvs_fail")
        raw = traceback.format_exc() + "\n" + lvs_buffer.getvalue()
        entry["logs"]["lvs"] = summarize_log(raw)
        (log_dir / "lvs.log").write_text(raw)
    entry["runtime_s"]["lvs"] = round(time.time() - lvs_start, 4)

    pex_start = time.time()
    pex_buffer = io.StringIO()
    try:
        env = os.environ.copy()
        env["PDK_ROOT"] = str(paths["pdk_root"])
        env["PDKPATH"] = str(paths["pdk_root"])
        env["PDK"] = "sky130A"
        with pushd(artifact_dir):
            run = subprocess.run(
                [str(paths["run_pex"]), f"{sample_id}.gds", sample_id.upper()],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
        pex_buffer.write(run.stdout)
        pex_buffer.write(run.stderr)
        pex_path = artifact_dir / f"{sample_id.upper()}_pex.spice"
        total_r, total_c = parse_simple_parasitics(pex_path)
        entry["pex"] = {
            "status": "complete",
            "total_resistance_ohms": total_r,
            "total_capacitance_farads": total_c,
            "spice_path": str(pex_path),
        }
        entry["artifacts"]["pex_spice"] = str(pex_path)
    except Exception:
        entry["pex"] = {"status": "error"}
        entry["failure_tags"].append("pex_fail")
        raw = traceback.format_exc() + "\n" + pex_buffer.getvalue()
        entry["logs"]["pex"] = summarize_log(raw)
        (log_dir / "pex.log").write_text(raw)
    else:
        entry["logs"]["pex"] = summarize_log(pex_buffer.getvalue())
        (log_dir / "pex.log").write_text(pex_buffer.getvalue())
    entry["runtime_s"]["pex"] = round(time.time() - pex_start, 4)
    entry["runtime_s"]["total"] = round(sum(entry["runtime_s"].values()), 4)
    entry["artifacts"]["log_dir"] = str(log_dir)
    return entry


def write_jsonl(path: Path, entries: list[dict]) -> None:
    with open(path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def write_csv(path: Path, rows: list[dict], write_header: bool) -> None:
    if not rows:
        return
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect the current mirror GLADE benchmark dataset.")
    parser.add_argument("--output-root", default="output/glade/current_mirror", help="Dataset output root.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this sample index.")
    parser.add_argument("--resume", action="store_true", help="Skip sample directories that already contain entry.json.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--flush-every", type=int, default=25, help="Flush JSONL/CSV every N completed samples.")
    return parser.parse_args()


def build_sample_plan(args, samples_root: Path) -> list[tuple[int, dict, Path]]:
    plan = []
    for idx, params in enumerate(iter_current_mirror_space()):
        if idx < args.start_index:
            continue
        if args.limit is not None and idx >= args.start_index + args.limit:
            break
        sample_id = f"current_mirror_{idx:05d}"
        sample_dir = samples_root / sample_id
        entry_path = sample_dir / "entry.json"
        if args.resume and entry_path.exists():
            continue
        plan.append((idx, params, sample_dir))
    return plan


def write_summary(output_root: Path, entries: list[dict], planned: int) -> None:
    summary = {
        "planned_samples": planned,
        "completed_samples": len(entries),
        "build_success": sum(1 for e in entries if e.get("build_status") == "success"),
        "drc_clean": sum(1 for e in entries if (e.get("drc") or {}).get("clean")),
        "lvs_clean": sum(1 for e in entries if (e.get("lvs") or {}).get("clean")),
        "lvs_property_error": sum(1 for e in entries if (e.get("lvs") or {}).get("status") == "property_error"),
        "pex_complete": sum(1 for e in entries if (e.get("pex") or {}).get("status") == "complete"),
        "avg_runtime_total_s": (
            sum(e.get("runtime_s", {}).get("total", 0.0) for e in entries) / len(entries) if entries else None
        ),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_file(Path(__file__).resolve().parent)
    paths = prepare_chipathon_environment(repo_root)

    output_root = (repo_root / args.output_root).resolve()
    samples_root = output_root / "samples"
    output_root.mkdir(parents=True, exist_ok=True)
    samples_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "generator_id": "current_mirror",
        "pdk": "sky130",
        "parameter_grid_size": total_combinations(),
        "space_definition": {
            "device": ["nfet", "pfet"],
            "numcols": [1, 2, 3, 4, 5],
            "width": {"start": 0.5, "stop": 20.0, "step": 0.25},
            "length": {"start": 0.15, "stop": 3.95, "step": 0.2},
        },
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2))

    jsonl_path = output_root / "dataset.jsonl"
    csv_path = output_root / "dataset.csv"
    write_header = not csv_path.exists()

    plan = build_sample_plan(args, samples_root)
    if not plan:
        print("No samples scheduled.")
        return 0

    print(f"Scheduled {len(plan)} samples (workers={args.workers}, flush_every={args.flush_every}).")

    batch_entries = []
    batch_rows = []
    all_entries = []

    def flush_batch():
        nonlocal write_header
        if not batch_entries:
            return
        write_jsonl(jsonl_path, batch_entries)
        write_csv(csv_path, batch_rows, write_header)
        write_header = False
        batch_entries.clear()
        batch_rows.clear()

    if args.workers == 1:
        for idx, params, sample_dir in plan:
            sample_id = f"current_mirror_{idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            entry = evaluate_current_mirror(sample_id, params, sample_dir, paths)
            (sample_dir / "entry.json").write_text(json.dumps(entry, indent=2))
            batch_entries.append(entry)
            batch_rows.append(flatten_entry(entry))
            all_entries.append(entry)
            if len(batch_entries) >= args.flush_every:
                flush_batch()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {}
            for idx, params, sample_dir in plan:
                sample_id = f"current_mirror_{idx:05d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                fut = ex.submit(evaluate_current_mirror, sample_id, params, sample_dir, paths)
                futures[fut] = sample_dir

            for fut in as_completed(futures):
                sample_dir = futures[fut]
                entry = fut.result()
                (sample_dir / "entry.json").write_text(json.dumps(entry, indent=2))
                batch_entries.append(entry)
                batch_rows.append(flatten_entry(entry))
                all_entries.append(entry)
                if len(batch_entries) >= args.flush_every:
                    flush_batch()

    flush_batch()
    write_summary(output_root, all_entries, len(plan))

    print(f"Wrote dataset artifacts under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
