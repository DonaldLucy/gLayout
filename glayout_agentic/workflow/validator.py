from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ValidationResult:
    success: bool
    compile_ok: bool
    execution_ok: bool
    gds_created: bool
    drc_lvs_requested: bool
    drc_pass: bool
    lvs_pass: bool
    stage: str
    returncode: int
    command: list[str]
    stdout: str
    stderr: str
    summary: str
    gds_path: Optional[Path] = None
    verification: Optional[dict[str, Any]] = None


def build_runtime_env(
    repo_root: Path,
    pdk_root: Optional[str] = None,
    pdk_path: Optional[str] = None,
) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(repo_root / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if pdk_root:
        env["PDK_ROOT"] = pdk_root
    if pdk_path:
        env["PDKPATH"] = pdk_path
    return env


def _trim_output(text: str, limit: int = 6000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _load_component_from_generated_file(
    python_file: Path,
    env: dict[str, str],
):
    previous_env = os.environ.copy()
    os.environ.update(env)
    try:
        module_name = f"_glayout_generated_{python_file.stem}_{abs(hash(str(python_file)))}"
        spec = importlib.util.spec_from_file_location(module_name, python_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module from {python_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        build_candidates = []
        for name in dir(module):
            if not name.startswith("build_"):
                continue
            candidate = getattr(module, name)
            if callable(candidate):
                build_candidates.append(candidate)
        if not build_candidates:
            raise RuntimeError(
                f"No callable build_* function found in generated file {python_file.name}."
            )
        component = build_candidates[0]()
        return component
    finally:
        os.environ.clear()
        os.environ.update(previous_env)


def _run_drc_lvs(
    repo_root: Path,
    python_file: Path,
    gds_output: Path,
    env: dict[str, str],
) -> tuple[bool, bool, dict[str, Any]]:
    previous_env = os.environ.copy()
    os.environ.update(env)
    try:
        if str(repo_root / "src") not in sys.path:
            sys.path.insert(0, str(repo_root / "src"))
        from glayout.verification.verification import run_verification

        component = _load_component_from_generated_file(python_file, env)
        component_name = getattr(component, "name", None) or python_file.stem
        verification = run_verification(str(gds_output), component_name, component)
        drc_pass = bool(verification["drc"]["is_pass"])
        lvs_pass = bool(verification["lvs"]["is_pass"])
        return drc_pass, lvs_pass, verification
    finally:
        os.environ.clear()
        os.environ.update(previous_env)


def validate_generated_file(
    repo_root: Path,
    python_file: Path,
    execute: bool,
    gds_output: Optional[Path],
    env: dict[str, str],
    run_drc_lvs: bool = False,
    timeout_sec: int = 180,
) -> ValidationResult:
    compile_cmd = [sys.executable, "-m", "py_compile", str(python_file)]
    compile_proc = subprocess.run(
        compile_cmd,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    if compile_proc.returncode != 0:
        return ValidationResult(
            success=False,
            compile_ok=False,
            execution_ok=False,
            gds_created=False,
            drc_lvs_requested=run_drc_lvs,
            drc_pass=False,
            lvs_pass=False,
            stage="compile",
            returncode=compile_proc.returncode,
            command=compile_cmd,
            stdout=_trim_output(compile_proc.stdout),
            stderr=_trim_output(compile_proc.stderr),
            summary="Python compilation failed.",
        )

    if not execute:
        return ValidationResult(
            success=True,
            compile_ok=True,
            execution_ok=False,
            gds_created=False,
            drc_lvs_requested=run_drc_lvs,
            drc_pass=False,
            lvs_pass=False,
            stage="compile_only",
            returncode=0,
            command=compile_cmd,
            stdout=_trim_output(compile_proc.stdout),
            stderr=_trim_output(compile_proc.stderr),
            summary="Compilation succeeded.",
        )

    if gds_output is None:
        gds_output = python_file.with_suffix(".gds")
    run_cmd = [sys.executable, str(python_file), "--output-gds", str(gds_output)]
    run_proc = subprocess.run(
        run_cmd,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    gds_created = gds_output.exists()
    if run_proc.returncode != 0:
        return ValidationResult(
            success=False,
            compile_ok=True,
            execution_ok=False,
            gds_created=gds_created,
            drc_lvs_requested=run_drc_lvs,
            drc_pass=False,
            lvs_pass=False,
            stage="execute",
            returncode=run_proc.returncode,
            command=run_cmd,
            stdout=_trim_output(run_proc.stdout),
            stderr=_trim_output(run_proc.stderr),
            summary="Generated file did not execute successfully.",
            gds_path=gds_output,
        )
    if not gds_created:
        return ValidationResult(
            success=False,
            compile_ok=True,
            execution_ok=True,
            gds_created=False,
            drc_lvs_requested=run_drc_lvs,
            drc_pass=False,
            lvs_pass=False,
            stage="gds_missing",
            returncode=0,
            command=run_cmd,
            stdout=_trim_output(run_proc.stdout),
            stderr=_trim_output(run_proc.stderr),
            summary="Execution finished but no GDS file was written.",
            gds_path=gds_output,
        )

    if run_drc_lvs:
        try:
            drc_pass, lvs_pass, verification = _run_drc_lvs(
                repo_root=repo_root,
                python_file=python_file,
                gds_output=gds_output,
                env=env,
            )
        except Exception as exc:
            return ValidationResult(
                success=False,
                compile_ok=True,
                execution_ok=True,
                gds_created=True,
                drc_lvs_requested=True,
                drc_pass=False,
                lvs_pass=False,
                stage="verify",
                returncode=1,
                command=run_cmd,
                stdout=_trim_output(run_proc.stdout),
                stderr=_trim_output(f"{run_proc.stderr}\n\n{type(exc).__name__}: {exc}"),
                summary="DRC/LVS verification failed to run.",
                gds_path=gds_output,
            )
        if not (drc_pass and lvs_pass):
            return ValidationResult(
                success=False,
                compile_ok=True,
                execution_ok=True,
                gds_created=True,
                drc_lvs_requested=True,
                drc_pass=drc_pass,
                lvs_pass=lvs_pass,
                stage="drc_lvs",
                returncode=1,
                command=run_cmd,
                stdout=_trim_output(run_proc.stdout),
                stderr=_trim_output(run_proc.stderr),
                summary=(
                    f"Execution succeeded, but DRC/LVS did not fully pass "
                    f"(DRC={drc_pass}, LVS={lvs_pass})."
                ),
                gds_path=gds_output,
                verification=verification,
            )
        return ValidationResult(
            success=True,
            compile_ok=True,
            execution_ok=True,
            gds_created=True,
            drc_lvs_requested=True,
            drc_pass=True,
            lvs_pass=True,
            stage="verified",
            returncode=0,
            command=run_cmd,
            stdout=_trim_output(run_proc.stdout),
            stderr=_trim_output(run_proc.stderr),
            summary="Compilation, execution, DRC, and LVS all passed.",
            gds_path=gds_output,
            verification=verification,
        )

    return ValidationResult(
        success=True,
        compile_ok=True,
        execution_ok=True,
        gds_created=True,
        drc_lvs_requested=False,
        drc_pass=False,
        lvs_pass=False,
        stage="success",
        returncode=0,
        command=run_cmd,
        stdout=_trim_output(run_proc.stdout),
        stderr=_trim_output(run_proc.stderr),
        summary="Compilation and execution succeeded, and a GDS file was produced.",
        gds_path=gds_output,
    )
