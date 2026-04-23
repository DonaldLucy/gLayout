from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ValidationResult:
    success: bool
    compile_ok: bool
    execution_ok: bool
    gds_created: bool
    stage: str
    returncode: int
    command: list[str]
    stdout: str
    stderr: str
    summary: str
    gds_path: Optional[Path] = None


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


def validate_generated_file(
    repo_root: Path,
    python_file: Path,
    execute: bool,
    gds_output: Optional[Path],
    env: dict[str, str],
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
            stage="gds_missing",
            returncode=0,
            command=run_cmd,
            stdout=_trim_output(run_proc.stdout),
            stderr=_trim_output(run_proc.stderr),
            summary="Execution finished but no GDS file was written.",
            gds_path=gds_output,
        )
    return ValidationResult(
        success=True,
        compile_ok=True,
        execution_ok=True,
        gds_created=True,
        stage="success",
        returncode=0,
        command=run_cmd,
        stdout=_trim_output(run_proc.stdout),
        stderr=_trim_output(run_proc.stderr),
        summary="Compilation and execution succeeded, and a GDS file was produced.",
        gds_path=gds_output,
    )
