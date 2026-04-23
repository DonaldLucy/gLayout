from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

from workflow.backends import ensure_runtime_identity_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local gLayout agent environment.")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_module(name: str):
    try:
        module = importlib.import_module(name)
        return True, getattr(module, "__version__", "unknown"), ""
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    args = parse_args()
    ensure_runtime_identity_env()
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    issues: list[str] = []
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"repo_root: {repo_root}")
    print(f"PDK_ROOT: {os.getenv('PDK_ROOT')}")
    print(f"PDKPATH: {os.getenv('PDKPATH')}")

    modules = [
        "numpy",
        "gdsfactory",
        "glayout",
        "torch",
        "transformers",
        "datasets",
        "peft",
        "bitsandbytes",
    ]
    module_results = {}
    print("\nPython modules:")
    for name in modules:
        ok, version, error = load_module(name)
        module_results[name] = (ok, version, error)
        if ok:
            print(f"  {name}: OK ({version})")
        else:
            print(f"  {name}: FAIL ({error})")
            if name in {"gdsfactory", "glayout", "torch"}:
                issues.append(f"Missing required module: {name}")

    numpy_ok, numpy_version, _ = module_results["numpy"]
    gds_ok, gds_version, _ = module_results["gdsfactory"]
    if numpy_ok and gds_ok and numpy_version and gds_version:
        try:
            numpy_major = int(str(numpy_version).split(".")[0])
        except ValueError:
            numpy_major = 0
        if numpy_major >= 2:
            issues.append(
                "Detected numpy>=2. Current repo execution is known to fail with "
                "gdsfactory 7.7.0 because `np.float_` was removed. Pin `numpy<2`."
            )

    commands = ["magic", "netgen", "ngspice", "klayout", "docker"]
    print("\nEDA tools:")
    for command in commands:
        resolved = shutil.which(command)
        print(f"  {command}: {resolved}")
        if command in {"magic", "netgen", "ngspice"} and resolved is None:
            issues.append(f"Missing required EDA tool on PATH: {command}")

    if not os.getenv("PDK_ROOT"):
        issues.append("PDK_ROOT is not set.")

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nEnvironment looks ready.")

    if args.strict and issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
