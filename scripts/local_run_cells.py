#!/usr/bin/env python3
"""Local helper to smoke-test gLayout cells without touching upstream code.

This script is intentionally local-only. It works around the current
`glayout.blocks` package import behavior by installing light-weight package
stubs in `sys.modules` before importing individual cell generators.

What it does:
1. Validates the active Python / tool / PDK environment.
2. Imports selected generators.
3. Builds each cell.
4. Writes a GDS file for each successful build.

Default behavior:
- Run environment checks.
- Build a stable starter set of cells using Inventory-constrained parameters.
- Write output GDS files under `local_cell_runs/`.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import os
import random
import shutil
import sys
import types
from pathlib import Path


REQUIRED_TOOLS = ("magic", "netgen", "ngspice")
PDK_SENTINEL = Path("sky130A/libs.tech/magic/sky130A.magicrc")

# Inventory-constrained sample points. These are explicit values inside the
# published parameter ranges from "GLayout Inventory 2025 (2).xlsx".
INVENTORY_SAMPLES = {
    "current_mirror": dict(device="pfet", numcols=3, width=2.0, length=0.5),
    "diff_pair": dict(width=2.0, length=0.5, fingers=4, n_or_p_fet=True, rmult=1),
    "fvf": dict(
        device_type="nmos",
        placement="horizontal",
        width=(2.0, 1.0),
        length=(1.0, 0.5),
        fingers=(2, 2),
        multipliers=(1, 1),
        sd_rmult=1,
    ),
    "transmission_gate": dict(
        width=(1.0, 2.0),
        length=(0.5, 0.5),
        fingers=(2, 2),
        multipliers=(1, 1),
    ),
    "resistor": dict(width=5.0, length=1.0, num_series=2),
    "mimcap_array": dict(rows=2, columns=2, size=(5.0, 5.0), rmult=1),
    "dse": dict(rmult=2, half_pload=(2.0, 0.5, 3), via_xlocation=10),
    "diff_pair_ibias": dict(
        half_diffpair_params=(2.0, 0.5, 2),
        diffpair_bias=(2.0, 0.5, 2),
        rmult=2,
        with_antenna_diode_on_diffinputs=0,
    ),
    "low_voltage_cmirror": dict(width=(4.0, 1.5), length=1.0, fingers=(2, 1), multipliers=(1, 1)),
    "opamp": dict(
        half_diffpair_params=(5.5, 1.0, 2),
        diffpair_bias=(5.5, 2.0, 2),
        half_common_source_params=(6.5, 1.0, 8, 2),
        half_common_source_bias=(5.5, 2.0, 7, 2),
        output_stage_params=(5.0, 1.0, 4),
        output_stage_bias=(5.5, 2.0, 3),
        half_pload=(5.5, 1.0, 4),
        mim_cap_size=(12.0, 12.0),
        mim_cap_rows=2,
        rmult=2,
        with_antenna_diode_on_diffinputs=0,
        add_output_stage=False,
    ),
    "super_class_ab_ota": dict(
        input_pair_params=(4.0, 2.0),
        fvf_shunt_params=(2.75, 1.0),
        local_current_bias_params=(3.5, 3.0),
        diff_pair_load_params=(9.0, 1.0),
        ratio=1,
        current_mirror_params=(2.25, 1.0),
        resistor_params=(0.5, 3.0, 4.0, 4.0),
        global_current_bias_params=(8.5, 1.4, 2.0),
    ),
}


def find_repo_root(start: Path) -> Path:
    """Find the repository root even if this helper is copied outside `scripts/`.

    A valid repo root must contain:
    - `setup.py`
    - `src/glayout`
    """

    for candidate in (start, *start.parents):
        if (candidate / "setup.py").is_file() and (candidate / "src" / "glayout").is_dir():
            return candidate
    raise RuntimeError(
        f"Could not locate the gLayout repo root from {start}. "
        "Run this script from inside the repository or keep it under the repo tree."
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
SRC_ROOT = REPO_ROOT / "src"
BLOCKS_ROOT = SRC_ROOT / "glayout" / "blocks"


def install_package_stub(name: str, path: Path) -> None:
    """Install a namespace-like stub package to avoid executing eager __init__ files."""
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__file__ = str(path / "__init__.py")
    module.__package__ = name
    module.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None, is_package=True)
    sys.modules[name] = module

    parent_name, _, child_name = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], child_name, module)


def prepare_import_path() -> None:
    """Ensure local source tree is importable and override problematic package roots."""
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    # Import the top-level package first. This does not import glayout.blocks.
    import glayout  # noqa: F401

    install_package_stub("glayout.blocks", BLOCKS_ROOT)
    install_package_stub("glayout.blocks.elementary", BLOCKS_ROOT / "elementary")
    install_package_stub("glayout.blocks.composite", BLOCKS_ROOT / "composite")
    install_package_stub("glayout.blocks.evaluator_box", BLOCKS_ROOT / "evaluator_box")


def detect_pdk_root() -> Path | None:
    """Resolve a usable PDK_ROOT by checking env first, then common locations."""
    candidates: list[Path] = []

    env_pdk_root = os.environ.get("PDK_ROOT")
    if env_pdk_root:
        candidates.append(Path(env_pdk_root))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "share" / "pdk")

    conda_root = os.environ.get("CONDA_ROOT")
    if conda_root:
        candidates.append(Path(conda_root) / "share" / "pdk")

    candidates.extend(
        [
            Path("/opt/conda/envs/GLdev/share/pdk"),
            Path("/headless/conda-env/miniconda3/share/pdk"),
            Path("/usr/bin/miniconda3/share/pdk"),
            Path("/usr/local/share/pdk"),
            Path("/foss/pdks"),
            Path("/foss/pdk"),
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / PDK_SENTINEL).is_file():
            return candidate

    # Fallback search for unusual container layouts.
    search_roots = [Path("/headless"), Path("/opt"), Path("/foss"), Path("/usr")]
    for root in search_roots:
        if not root.exists():
            continue
        for match in root.glob("**/sky130A/libs.tech/magic/sky130A.magicrc"):
            return match.parents[3]

    return None


def check_environment() -> tuple[bool, object]:
    """Check the Python, tool, and PDK environment before building any cell."""
    print("[INFO] Python executable:", sys.executable)
    print("[INFO] Python version:", sys.version.split()[0])
    print("[INFO] CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
    print("[INFO] PDK_ROOT:", os.environ.get("PDK_ROOT", ""))
    print("[INFO] PDKPATH:", os.environ.get("PDKPATH", ""))

    env_pdk_root = os.environ.get("PDK_ROOT")
    if not env_pdk_root or not (Path(env_pdk_root) / PDK_SENTINEL).is_file():
        print("[FAIL] Could not locate a valid PDK_ROOT containing sky130A.magicrc")
        return False, None

    missing_tools = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing_tools:
        print(f"[FAIL] Missing tools in PATH: {', '.join(missing_tools)}")
        return False, None

    for tool in REQUIRED_TOOLS:
        print(f"[INFO] {tool}: {shutil.which(tool)}")

    try:
        import gdsfactory
    except Exception as exc:
        print(f"[FAIL] Could not import gdsfactory: {exc}")
        return False, None

    print("[INFO] gdsfactory version:", gdsfactory.__version__)

    try:
        from glayout import sky130
    except Exception as exc:
        print(f"[FAIL] Could not import glayout.sky130: {exc}")
        return False, None

    if sky130 is None:
        print("[FAIL] glayout.sky130 resolved to None")
        return False, None

    print("[PASS] Environment looks ready.")
    return True, sky130


def refresh_pdk_environment(pdk_root: Path) -> None:
    """Reset PDK-related environment variables before each verification step.

    The chipathon container's `magic` wrapper already knows where its runtime
    files live. For this environment, forcing `CAD_ROOT` to a conda path can
    break `magicdnull`, so we explicitly remove it.
    """
    os.environ["PDK_ROOT"] = str(pdk_root)
    os.environ["PDKPATH"] = str(pdk_root)
    os.environ["PDK"] = "sky130A"
    os.environ["MAGIC_PDK_ROOT"] = str(pdk_root)
    os.environ["NETGEN_PDK_ROOT"] = str(pdk_root)
    os.environ.pop("CAD_ROOT", None)


def get_drc_report_path(report_dir: Path, design_name: str) -> Path:
    return report_dir / "drc" / design_name / f"{design_name}.rpt"


def get_lvs_report_path(report_dir: Path, design_name: str) -> Path:
    return report_dir / "lvs" / design_name / f"{design_name}_lvs.rpt"


def classify_drc(report_path: Path) -> tuple[bool, str]:
    if not report_path.is_file():
        return False, "missing DRC report"

    content = report_path.read_text()
    if "count: 0" in content:
        return True, "clean"
    return False, "report contains DRC errors"


def classify_lvs(report_path: Path) -> tuple[bool, str]:
    if not report_path.is_file():
        return False, "missing LVS report"

    content = report_path.read_text()
    if "Top level cell failed pin matching" in content or "Netlists do not match." in content:
        return False, "LVS mismatch"
    if "Property errors were found." in content and "Circuits match uniquely." in content:
        return True, "match with property errors"
    if "Circuits match uniquely." in content or "Netlists match uniquely." in content:
        return True, "clean"
    return False, "inconclusive LVS report"


def get_sample_params(cell_name: str, profile: str, seed: int) -> dict[str, object]:
    """Return parameters for a cell under the selected sampling profile."""
    if profile == "defaults":
        return {}
    if profile == "inventory":
        return INVENTORY_SAMPLES.get(cell_name, {}).copy()
    raise ValueError(f"Unsupported profile: {profile}")


def build_registry(sky130: object, profile: str, seed: int) -> dict[str, dict[str, object]]:
    """Return supported cell builders.

    The imports happen lazily so one broken cell does not block the whole script.
    """

    def current_mirror():
        mod = importlib.import_module("glayout.blocks.elementary.current_mirror.current_mirror")
        return mod.current_mirror(sky130, **get_sample_params("current_mirror", profile, seed))

    def current_mirror_verify():
        mod = importlib.import_module("glayout.blocks.elementary.current_mirror.current_mirror")
        return mod.add_cm_labels(mod.current_mirror(sky130, **get_sample_params("current_mirror", profile, seed)), sky130)

    def diff_pair():
        mod = importlib.import_module("glayout.blocks.elementary.diff_pair.diff_pair")
        return mod.diff_pair(sky130, **get_sample_params("diff_pair", profile, seed))

    def diff_pair_verify():
        mod = importlib.import_module("glayout.blocks.elementary.diff_pair.diff_pair")
        return mod.add_df_labels(mod.diff_pair(sky130, **get_sample_params("diff_pair", profile, seed)), sky130)

    def fvf():
        mod = importlib.import_module("glayout.blocks.elementary.FVF.fvf")
        return mod.flipped_voltage_follower(sky130, **get_sample_params("fvf", profile, seed))

    def fvf_verify():
        mod = importlib.import_module("glayout.blocks.elementary.FVF.fvf")
        return mod.sky130_add_fvf_labels(
            mod.flipped_voltage_follower(sky130, **get_sample_params("fvf", profile, seed))
        )

    def transmission_gate():
        mod = importlib.import_module("glayout.blocks.elementary.transmission_gate.transmission_gate")
        return mod.transmission_gate(sky130, **get_sample_params("transmission_gate", profile, seed))

    def transmission_gate_verify():
        mod = importlib.import_module("glayout.blocks.elementary.transmission_gate.transmission_gate")
        return mod.add_tg_labels(mod.transmission_gate(sky130, **get_sample_params("transmission_gate", profile, seed)), sky130)

    def resistor():
        mod = importlib.import_module("glayout.primitives.resistor")
        params = get_sample_params("resistor", profile, seed)
        return mod.resistor(sky130, with_tie=True, **params)

    def mimcap_array():
        mod = importlib.import_module("glayout.primitives.mimcap")
        return mod.mimcap_array(sky130, **get_sample_params("mimcap_array", profile, seed))

    def dse():
        mod = importlib.import_module(
            "glayout.blocks.composite.differential_to_single_ended_converter.differential_to_single_ended_converter"
        )
        return mod.differential_to_single_ended_converter(sky130, **get_sample_params("dse", profile, seed))

    def diff_pair_ibias():
        mod = importlib.import_module(
            "glayout.blocks.composite.diffpair_cmirror_bias.diff_pair_cmirrorbias"
        )
        return mod.diff_pair_ibias(sky130, **get_sample_params("diff_pair_ibias", profile, seed))

    def low_voltage_cmirror():
        mod = importlib.import_module(
            "glayout.blocks.composite.low_voltage_cmirror.low_voltage_cmirror"
        )
        return mod.low_voltage_cmirror(sky130, **get_sample_params("low_voltage_cmirror", profile, seed))

    def opamp():
        mod = importlib.import_module("glayout.blocks.composite.opamp.opamp")
        return mod.opamp(sky130, **get_sample_params("opamp", profile, seed))

    def super_class_ab_ota():
        mod = importlib.import_module("glayout.blocks.composite.fvf_based_ota.ota")
        return mod.super_class_AB_OTA(sky130, **get_sample_params("super_class_ab_ota", profile, seed))

    return {
        "current_mirror": {
            "build": current_mirror,
            "verify": current_mirror_verify,
            "design_name": "CMIRROR",
        },
        "diff_pair": {
            "build": diff_pair,
            "verify": diff_pair_verify,
            "design_name": "DIFF_PAIR",
        },
        "fvf": {
            "build": fvf,
            "verify": fvf_verify,
            "design_name": "FVF",
        },
        "transmission_gate": {
            "build": transmission_gate,
            "verify": transmission_gate_verify,
            "design_name": "TRANSMISSION_GATE",
        },
        "resistor": {"build": resistor, "design_name": "RESISTOR"},
        "mimcap_array": {"build": mimcap_array, "design_name": "MIMCAP_ARRAY"},
        "dse": {"build": dse, "design_name": "DSE"},
        "diff_pair_ibias": {"build": diff_pair_ibias, "design_name": "DIFF_PAIR_IBIAS"},
        "low_voltage_cmirror": {"build": low_voltage_cmirror, "design_name": "LOW_VOLTAGE_CMIRROR"},
        "opamp": {"build": opamp, "design_name": "OPAMP"},
        "super_class_ab_ota": {"build": super_class_ab_ota, "design_name": "SUPER_CLASS_AB_OTA"},
    }


DEFAULT_CELLS = [
    "current_mirror",
    "diff_pair",
    "fvf",
    "transmission_gate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local gLayout cell smoke runner.")
    parser.add_argument(
        "--profile",
        choices=("inventory", "defaults"),
        default="inventory",
        help="Parameter profile to use. Default: inventory-constrained sample points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Reserved seed for future randomized sampling modes. Kept for reproducibility.",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=DEFAULT_CELLS,
        help="Cell names to build. Default: stable starter set.",
    )
    parser.add_argument(
        "--output-dir",
        default="local_cell_runs",
        help="Directory for generated GDS files.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run DRC and LVS after GDS generation for cells that support verification.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported cells and exit.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip environment checks if you already validated the session.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running the remaining cells after a failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detected_pdk_root = detect_pdk_root()
    if detected_pdk_root is not None:
        refresh_pdk_environment(detected_pdk_root)

    prepare_import_path()

    # We only need sky130 after imports are prepared.
    env_ok, sky130 = (True, None) if args.no_check else check_environment()
    if not args.no_check and not env_ok:
        return 1
    if args.no_check:
        from glayout import sky130 as imported_sky130

        sky130 = imported_sky130

    registry = build_registry(sky130, args.profile, args.seed)

    if args.list:
        print("Supported cells:")
        for name in registry:
            print(f"  - {name}")
        return 0

    unknown = [name for name in args.cells if name not in registry]
    if unknown:
        print(f"[FAIL] Unknown cells: {', '.join(unknown)}")
        print("Use --list to see supported names.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for cell_name in args.cells:
        print(f"[RUN] {cell_name}")
        try:
            if detected_pdk_root is not None:
                refresh_pdk_environment(detected_pdk_root)

            entry = registry[cell_name]
            build_fn = entry.get("verify", entry["build"]) if args.verify else entry["build"]
            component = build_fn()
            design_name = entry["design_name"]
            component.name = design_name
            gds_path = output_dir / f"{cell_name}.gds"
            component.write_gds(gds_path)
            print(f"[OK] {cell_name} -> {gds_path} (ports={len(component.ports)})")

            if args.verify:
                try:
                    if detected_pdk_root is not None:
                        refresh_pdk_environment(detected_pdk_root)

                    from glayout import sky130 as verify_pdk

                    report_dir = output_dir / cell_name / "reports"
                    report_dir.mkdir(parents=True, exist_ok=True)

                    drc_result = verify_pdk.drc_magic(gds_path, design_name, output_file=report_dir)
                    drc_report = get_drc_report_path(report_dir, design_name)
                    drc_ok, drc_summary = classify_drc(drc_report)
                    print(f"[INFO] {cell_name} DRC -> {drc_result}")
                    print(f"[INFO] {cell_name} DRC report -> {drc_report} ({drc_summary})")
                    if not drc_ok:
                        failures.append((f"{cell_name}::drc", RuntimeError(drc_summary)))
                        print(f"[FAIL] {cell_name} DRC: {drc_summary}")
                        if not args.keep_going:
                            break

                    lvs_result = verify_pdk.lvs_netgen(
                        layout=component,
                        design_name=design_name,
                        output_file_path=report_dir,
                    )
                    lvs_report = get_lvs_report_path(report_dir, design_name)
                    lvs_ok, lvs_summary = classify_lvs(lvs_report)
                    print(f"[INFO] {cell_name} LVS -> {lvs_result}")
                    print(f"[INFO] {cell_name} LVS report -> {lvs_report} ({lvs_summary})")
                    if not lvs_ok:
                        failures.append((f"{cell_name}::lvs", RuntimeError(lvs_summary)))
                        print(f"[FAIL] {cell_name} LVS: {lvs_summary}")
                        if not args.keep_going:
                            break
                except Exception as exc:
                    failures.append((f"{cell_name}::verify", exc))
                    print(f"[FAIL] {cell_name} verify: {type(exc).__name__}: {exc}")
                    if not args.keep_going:
                        break
        except Exception as exc:
            failures.append((cell_name, exc))
            print(f"[FAIL] {cell_name}: {type(exc).__name__}: {exc}")
            if not args.keep_going:
                break

    if failures:
        print("\nFailures:")
        for cell_name, exc in failures:
            print(f"  - {cell_name}: {type(exc).__name__}: {exc}")
        return 1

    print("\n[PASS] All requested cells completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
