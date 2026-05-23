from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from surrogate_qor.registry import add_repo_import_paths, load_generator_specs, specs_as_manifest
    from surrogate_qor.sampling import generate_sample_plan, shard_plan, write_plan
else:
    from .registry import add_repo_import_paths, load_generator_specs, specs_as_manifest
    from .sampling import generate_sample_plan, shard_plan, write_plan


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def _bbox(component: Any) -> list[float] | None:
    candidates = []
    for attr in ("bbox", "bbox_np"):
        try:
            value = getattr(component, attr)
            value = value() if callable(value) else value
            candidates.append(value)
        except Exception:
            pass
    for value in candidates:
        try:
            flat = [float(x) for row in value for x in row]
            if len(flat) >= 4:
                return [flat[0], flat[1], flat[2], flat[3]]
        except Exception:
            pass
    return None


def _component_features(component: Any) -> dict[str, Any]:
    bbox = _bbox(component)
    width = height = None
    if bbox:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
    polygon_count = 0
    layer_count = 0
    try:
        polygons = component.get_polygons(by_spec=True)
        layer_count = len(polygons)
        polygon_count = sum(len(items) for items in polygons.values())
    except Exception:
        pass
    area = None
    try:
        area = float(component.area())
    except Exception:
        pass
    return {
        "bbox": bbox,
        "bbox_width": width,
        "bbox_height": height,
        "area_um2": area,
        "port_count": len(getattr(component, "ports", {}) or {}),
        "reference_count": len(getattr(component, "references", []) or []),
        "polygon_count": polygon_count,
        "layer_count": layer_count,
    }


def _wrap_reference_if_needed(case_id: str, component: Any) -> Any:
    if isinstance(component, tuple):
        component = component[0]
    if hasattr(component, "write_gds"):
        return component
    if hasattr(component, "parent") and hasattr(component, "ports"):
        from gdsfactory.component import Component

        wrapper = Component(f"{case_id}_wrapped")
        ref = wrapper.add_ref(component.parent)
        try:
            ref.move(component.center)
        except Exception:
            pass
        try:
            wrapper.add_ports(ref.get_ports_list())
        except Exception:
            pass
        info = getattr(component, "info", None)
        if info:
            wrapper.info.update(info)
        return wrapper
    return component


def _sidecar_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return {"exists": True, "error": str(exc), "bytes": path.stat().st_size}
    calls = data.get("calls", {})
    objects = data.get("objects", {})
    route_intents = 0
    max_depth = 0
    for call in calls.values():
        route_intents += len(call.get("route_intents", []) or [])
        depth = 0
        parent = call.get("parent_call_id")
        while parent and parent in calls and depth < 100:
            depth += 1
            parent = calls[parent].get("parent_call_id")
        max_depth = max(max_depth, depth)
    return {
        "exists": True,
        "bytes": path.stat().st_size,
        "call_count": len(calls),
        "object_count": len(objects),
        "route_intent_count": route_intents,
        "max_call_depth": max_depth,
    }


def _load_spec_map(selected: set[str] | None = None) -> dict[str, Any]:
    add_repo_import_paths()
    return {spec.generator_id: spec for spec in load_generator_specs(selected)}


def _collect_one(sample: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    add_repo_import_paths()
    spec_map = _load_spec_map()
    spec = spec_map[sample["generator_id"]]
    sample_dir = Path(settings["output_dir"]) / "samples" / sample["sample_id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    record = dict(sample)
    record["sample_dir"] = str(sample_dir)
    record["timings_s"] = {}
    design_name = f"{sample['family']}_{sample['sample_id']}".replace(".", "_").replace("-", "_")
    record["design_name"] = design_name

    try:
        from glayout import disable_source_mapping, enable_source_mapping, reset_source_mapping, sky130
        from gdsfactory.cell import clear_cache

        try:
            clear_cache()
        except Exception:
            pass
        reset_source_mapping()
        if settings.get("enable_smgr", True):
            enable_source_mapping(reset=True, auto_emit_sidecar=True)
        else:
            disable_source_mapping()

        t0 = time.time()
        component = _wrap_reference_if_needed(sample["sample_id"], spec.builder(sky130, sample.get("params", {})))
        record["timings_s"]["build"] = time.time() - t0
        record["build_ok"] = hasattr(component, "write_gds")
        record["features"] = {
            "geometric": _component_features(component),
            "code": sample.get("code_features", {}),
        }

        gds_path = sample_dir / f"{design_name}.gds"
        t0 = time.time()
        component.write_gds(str(gds_path))
        record["timings_s"]["write_gds"] = time.time() - t0
        record["gds"] = str(gds_path)
        sidecar = gds_path.with_suffix(".provenance.json")
        record["sidecar"] = str(sidecar)
        record["features"]["provenance"] = _sidecar_summary(sidecar)

        if settings.get("run_drc", False):
            from run_smgr_regression import _run_drc

            t0 = time.time()
            record["drc"] = _run_drc(component, design_name, sample_dir)
            record["timings_s"]["drc"] = time.time() - t0
            record["drc_pass"] = bool(record["drc"].get("is_clean"))

        if settings.get("run_lvs", False):
            from run_smgr_regression import _run_lvs

            t0 = time.time()
            record["lvs"] = _run_lvs(component, design_name, sample_dir)
            record["timings_s"]["lvs"] = time.time() - t0
            record["lvs_pass"] = bool(record["lvs"].get("is_clean"))

        if settings.get("run_pex", False):
            from glayout.verification.physical_features import run_physical_feature_extraction

            old_cwd = Path.cwd()
            os.chdir(sample_dir)
            try:
                t0 = time.time()
                record["physical"] = run_physical_feature_extraction(str(gds_path), design_name, component)
                record["timings_s"]["pex"] = time.time() - t0
                record["pex_pass"] = record["physical"].get("pex", {}).get("status") == "PEX Complete"
            finally:
                os.chdir(old_cwd)

        record["ok"] = True
    except Exception as exc:
        record["ok"] = False
        record["build_ok"] = False
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc(limit=12)
        record.setdefault("drc_pass", False)
        record.setdefault("lvs_pass", False)
        record.setdefault("pex_pass", False)
        (sample_dir / "error.txt").write_text(record["traceback"])
    finally:
        record["timings_s"]["total"] = time.time() - started
    return record


def _existing_sample_ids(dataset_path: Path) -> set[str]:
    if not dataset_path.exists():
        return set()
    ids: set[str] = set()
    with dataset_path.open() as handle:
        for line in handle:
            try:
                ids.add(json.loads(line)["sample_id"])
            except Exception:
                continue
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect gLayout surrogate QoR training records.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-parameterized-cell", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--cases", nargs="*", default=None, help="Generator ids or family names to include.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-drc", action="store_true")
    parser.add_argument("--run-lvs", action="store_true")
    parser.add_argument("--run-pex", action="store_true")
    parser.add_argument("--skip-pex", action="store_true", help="Accepted for explicitness; PEX is off unless --run-pex is set.")
    parser.add_argument("--disable-smgr", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.cases) if args.cases else None
    specs = load_generator_specs(selected)
    manifest_path = output_dir / "registry_manifest.json"
    manifest_path.write_text(json.dumps(specs_as_manifest(specs), indent=2, sort_keys=True, default=_json_default))

    plan = generate_sample_plan(
        specs,
        samples_per_parameterized_cell=args.samples_per_parameterized_cell,
        seed=args.seed,
        include_fixed=True,
    )
    plan = shard_plan(plan, args.num_shards, args.shard_index)
    if args.offset:
        plan = plan[args.offset :]
    if args.limit is not None:
        plan = plan[: args.limit]
    plan_path = output_dir / "sample_plan.jsonl"
    write_plan(plan_path, plan)
    print(f"[surrogate] specs={len(specs)} planned_samples={len(plan)} plan={plan_path}")
    if args.plan_only:
        return 0

    dataset_path = output_dir / "dataset.jsonl"
    if args.skip_existing:
        done = _existing_sample_ids(dataset_path)
        plan = [row for row in plan if row["sample_id"] not in done]
        print(f"[surrogate] remaining_after_skip_existing={len(plan)}")

    settings = {
        "output_dir": str(output_dir),
        "run_drc": args.run_drc,
        "run_lvs": args.run_lvs,
        "run_pex": args.run_pex and not args.skip_pex,
        "enable_smgr": not args.disable_smgr,
    }

    completed = 0
    failures = 0
    with dataset_path.open("a") as handle:
        if args.workers <= 1:
            for sample in plan:
                record = _collect_one(sample, settings)
                failures += 0 if record.get("ok") else 1
                completed += 1
                handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
                handle.flush()
                print(f"[{completed}/{len(plan)}] {record['sample_id']} ok={record.get('ok')} drc={record.get('drc_pass')} lvs={record.get('lvs_pass')}")
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_collect_one, sample, settings) for sample in plan]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    failures += 0 if record.get("ok") else 1
                    completed += 1
                    handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
                    handle.flush()
                    print(f"[{completed}/{len(plan)}] {record['sample_id']} ok={record.get('ok')} drc={record.get('drc_pass')} lvs={record.get('lvs_pass')}")

    summary = {
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "plan": str(plan_path),
        "samples": completed,
        "failures": failures,
        "settings": settings,
    }
    (output_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[surrogate] wrote {dataset_path} failures={failures}")
    # Build/runtime failures are retained as negative feasibility records, so a
    # sweep with invalid sampled parameters should still be usable downstream.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
