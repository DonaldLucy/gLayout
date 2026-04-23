from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .artifacts import new_run_directory, write_json, write_text
from .backends import BackendError, LocalHFBackend, SkillBackend
from .dataset import record_training_example
from .prompts import PromptLibrary
from .skills import SkillLibrary, SkillMatch
from .validator import ValidationResult, build_runtime_env, validate_generated_file


@dataclass
class AgentRequest:
    task: str
    input_code: Optional[str] = None
    backend: str = "auto"
    disable_skills: bool = False
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path: Optional[str] = None
    max_attempts: int = 10
    execute: bool = False
    run_drc_lvs: bool = False
    output_py: Optional[Path] = None
    output_gds: Optional[Path] = None
    runs_dir: Path = Path("glayout_agentic/runs")
    dataset_dir: Path = Path("glayout_agentic/data/training")
    pdk_root: Optional[str] = None
    pdk_path: Optional[str] = None
    load_in_4bit: bool = True
    max_new_tokens: int = 1536
    temperature: float = 0.0
    top_p: float = 0.95
    timeout_sec: int = 180
    record_training: bool = True


@dataclass
class AgentRunResult:
    success: bool
    run_dir: Path
    final_python_path: Path
    final_gds_path: Optional[Path]
    attempts: int
    backend_used: str
    skill_name: Optional[str]
    message: str
    drc_pass: Optional[bool] = None
    lvs_pass: Optional[bool] = None


class GLayoutCodeAgent:
    def __init__(self, repo_root: Path, asset_root: Path):
        self.repo_root = repo_root.resolve()
        self.asset_root = asset_root.resolve()
        self.prompts = PromptLibrary(self.asset_root / "prompts", self.repo_root)
        self.skills = SkillLibrary(self.asset_root / "skills")
        self._backend_cache: dict[tuple[object, ...], LocalHFBackend] = {}

    @staticmethod
    def _validation_score(validation: ValidationResult) -> tuple[int, int, int, int, int]:
        return (
            int(validation.compile_ok),
            int(validation.execution_ok),
            int(validation.gds_created),
            int(validation.drc_pass),
            int(validation.lvs_pass),
        )

    @staticmethod
    def _repair_focus(validation: ValidationResult) -> str:
        if validation.verification_feedback and "missing `component.info['netlist']`" in validation.verification_feedback:
            return (
                "Preserve the current working geometry, placement, routing, CLI, and GDS-writing path. "
                "Focus only on attaching the correct glayout netlist metadata for LVS, preferably via "
                "`component.info['netlist']` and/or a repo-native Netlist construction pattern."
            )
        if validation.execution_ok and validation.gds_created:
            return (
                "Do not regress compile/execute/GDS behavior. Preserve the best runnable skeleton and make only the "
                "smallest changes needed to improve DRC/LVS feedback."
            )
        if validation.compile_ok:
            return (
                "Preserve imports and CLI structure where they already work. Fix the runtime error without switching "
                "back to generic gdsfactory-only APIs."
            )
        return (
            "First restore a clean, runnable gLayout-native script. Use repository-native imports, ports, and routing helpers."
        )

    def run(self, request: AgentRequest) -> AgentRunResult:
        task = request.task.strip()
        if not task and request.input_code:
            task = "Repair the supplied gLayout generator so it compiles and writes a GDS."
        if not task:
            raise ValueError("A task description or input code is required.")

        run_dir = new_run_directory(request.runs_dir)
        skill_match = None if request.disable_skills else self.skills.match(task)
        primary_backend, backend_label = self._select_backend(request, skill_match)
        repair_backend = self._select_repair_backend(
            request, skill_match, primary_backend
        )
        env = build_runtime_env(
            self.repo_root, pdk_root=request.pdk_root, pdk_path=request.pdk_path
        )

        write_json(
            run_dir / "request.json",
            {
                "task": task,
                "backend": request.backend,
                "disable_skills": request.disable_skills,
                "model_name": request.model_name,
                "adapter_path": request.adapter_path,
                "execute": request.execute,
                "pdk_root": request.pdk_root,
                "pdk_path": request.pdk_path,
                "skill_name": skill_match.name if skill_match else None,
                "reference_files": self.prompts.reference_descriptions,
            },
        )
        print(f"[agent] Run directory: {run_dir}", flush=True)
        if request.disable_skills:
            print("[agent] Skills are disabled. Using only LLM generation plus repo guidance/reference snippets.", flush=True)
        else:
            print(f"[agent] Skill match: {skill_match.name if skill_match else 'none'}", flush=True)
        print("[agent] Reference files loaded before generation:", flush=True)
        for ref in self.prompts.reference_descriptions:
            print(
                f"  - {ref['path']}:{ref['start']}-{ref['end']} ({ref['purpose']})",
                flush=True,
            )

        if request.input_code is not None:
            current_code = request.input_code
            current_prompt = self.prompts.build_generation_prompt(
                task=task,
                source_code=request.input_code,
                skill_hint=skill_match.prompt_hint if skill_match else None,
            )
        else:
            current_prompt = self.prompts.build_generation_prompt(
                task=task,
                skill_hint=skill_match.prompt_hint if skill_match else None,
            )
            current_code = primary_backend.generate(
                current_prompt, skill_match=skill_match
            )

        attempt_records: list[dict[str, object]] = []
        final_python_path = request.output_py or run_dir / "final_generated.py"
        final_gds_path = request.output_gds
        best_code = current_code
        best_validation: Optional[ValidationResult] = None

        for attempt_index in range(1, request.max_attempts + 1):
            print(
                f"[agent] Attempt {attempt_index}/{request.max_attempts}: validating current candidate",
                flush=True,
            )
            candidate_path = run_dir / f"attempt_{attempt_index:02d}.py"
            gds_candidate_path = run_dir / f"attempt_{attempt_index:02d}.gds"
            prompt_path = run_dir / f"attempt_{attempt_index:02d}_prompt.txt"
            log_path = run_dir / f"attempt_{attempt_index:02d}_validator.log"

            write_text(prompt_path, current_prompt)
            write_text(candidate_path, current_code)

            validation = validate_generated_file(
                repo_root=self.repo_root,
                python_file=candidate_path,
                execute=request.execute,
                run_drc_lvs=request.run_drc_lvs,
                gds_output=gds_candidate_path if request.execute else None,
                env=env,
                timeout_sec=request.timeout_sec,
            )
            self._write_validation_log(log_path, validation)
            write_json(
                run_dir / f"attempt_{attempt_index:02d}_validation.json", validation
            )

            attempt_record = self._attempt_record(
                attempt_index=attempt_index,
                candidate_path=candidate_path,
                validation=validation,
            )
            attempt_records.append(attempt_record)

            if best_validation is None or self._validation_score(validation) > self._validation_score(best_validation):
                best_validation = validation
                best_code = current_code
                print(
                    f"[agent] Updated best candidate at attempt {attempt_index} with score {self._validation_score(validation)}",
                    flush=True,
                )

            if validation.success:
                final_python_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_path, final_python_path)
                copied_gds_path = None
                if request.execute and validation.gds_path and validation.gds_path.exists():
                    copied_gds_path = final_gds_path or run_dir / "final_generated.gds"
                    copied_gds_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(validation.gds_path, copied_gds_path)

                final_code = candidate_path.read_text(encoding="utf-8")
                if request.record_training:
                    record_training_example(
                        dataset_dir=request.dataset_dir,
                        task=task,
                        final_code=final_code,
                        final_filename=final_python_path.name,
                        attempts=attempt_records,
                        backend=backend_label,
                        model_name=request.model_name if "hf" in backend_label else None,
                        skill_name=skill_match.name if skill_match else None,
                    )
                return AgentRunResult(
                    success=True,
                    run_dir=run_dir,
                    final_python_path=final_python_path,
                    final_gds_path=copied_gds_path,
                    attempts=attempt_index,
                    backend_used=backend_label,
                    skill_name=skill_match.name if skill_match else None,
                    message=validation.summary,
                    drc_pass=validation.drc_pass if request.run_drc_lvs else None,
                    lvs_pass=validation.lvs_pass if request.run_drc_lvs else None,
                )

            if attempt_index == request.max_attempts:
                break

            print(
                f"[agent] Attempt {attempt_index} failed at stage `{validation.stage}`. Starting repair generation.",
                flush=True,
            )
            if best_validation is not None and self._validation_score(best_validation) > self._validation_score(validation):
                print(
                    f"[agent] Current candidate regressed below best-so-far score {self._validation_score(best_validation)}. "
                    "Repair will use the best known candidate as the anchor.",
                    flush=True,
                )
            current_prompt = self.prompts.build_repair_prompt(
                task=task,
                previous_code=current_code,
                validation_log=self._validation_text(validation),
                attempt_history=self._attempt_history_text(attempt_records),
                best_candidate_code=best_code if best_validation is not None else None,
                best_candidate_summary=(
                    f"score={self._validation_score(best_validation)} | {best_validation.summary}"
                    if best_validation is not None
                    else None
                ),
                repair_focus=self._repair_focus(best_validation or validation),
                skill_hint=skill_match.prompt_hint if skill_match else None,
            )
            repair_started = time.monotonic()
            current_code = repair_backend.generate(
                current_prompt, skill_match=skill_match
            )
            print(
                f"[agent] Repair candidate generated in {time.monotonic() - repair_started:.1f}s",
                flush=True,
            )

        failed_python_path = run_dir / "final_failed.py"
        shutil.copy2(candidate_path, failed_python_path)
        return AgentRunResult(
            success=False,
            run_dir=run_dir,
            final_python_path=failed_python_path,
            final_gds_path=None,
            attempts=request.max_attempts,
            backend_used=backend_label,
            skill_name=skill_match.name if skill_match else None,
            message=attempt_records[-1]["summary"],  # type: ignore[index]
            drc_pass=None,
            lvs_pass=None,
        )

    def _select_backend(
        self, request: AgentRequest, skill_match: Optional[SkillMatch]
    ):
        if request.backend == "skill":
            return SkillBackend(), "skill"
        if request.backend == "local-hf":
            return self._build_hf_backend(request), "local-hf"
        if request.backend != "auto":
            raise ValueError(f"Unsupported backend: {request.backend}")
        if skill_match is not None:
            return SkillBackend(), "skill(auto)"
        return self._build_hf_backend(request), "local-hf(auto)"

    def _select_repair_backend(
        self,
        request: AgentRequest,
        skill_match: Optional[SkillMatch],
        primary_backend,
    ):
        if request.backend == "skill":
            return SkillBackend()
        if request.backend == "local-hf":
            return primary_backend
        if skill_match is not None:
            return SkillBackend()
        return self._build_hf_backend(request)

    def _build_hf_backend(self, request: AgentRequest) -> LocalHFBackend:
        key = (
            request.model_name,
            request.adapter_path,
            request.load_in_4bit,
            request.max_new_tokens,
            request.temperature,
            request.top_p,
        )
        if key not in self._backend_cache:
            self._backend_cache[key] = LocalHFBackend(
                model_name_or_path=request.model_name,
                adapter_path=request.adapter_path,
                load_in_4bit=request.load_in_4bit,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        return self._backend_cache[key]

    @staticmethod
    def _validation_text(validation: ValidationResult) -> str:
        parts = [
            f"Stage: {validation.stage}",
            f"Summary: {validation.summary}",
            f"Return code: {validation.returncode}",
            f"Command: {' '.join(validation.command)}",
        ]
        if validation.verification_feedback:
            parts.append(f"Verification feedback:\n{validation.verification_feedback}")
        if validation.stdout:
            parts.append(f"STDOUT:\n{validation.stdout}")
        if validation.stderr:
            parts.append(f"STDERR:\n{validation.stderr}")
        return "\n\n".join(parts)

    @staticmethod
    def _attempt_history_text(attempt_records: list[dict[str, object]], limit: int = 3) -> str:
        if not attempt_records:
            return ""
        lines: list[str] = []
        for attempt in attempt_records[-limit:]:
            lines.append(
                f"Attempt {attempt.get('attempt')}: stage={attempt.get('stage')} success={attempt.get('success')}"
            )
            summary = attempt.get("summary")
            if summary:
                lines.append(f"Summary: {summary}")
            verification_feedback = attempt.get("verification_feedback")
            if verification_feedback:
                lines.append(f"Verification feedback:\n{verification_feedback}")
        return "\n".join(lines)

    @staticmethod
    def _write_validation_log(path: Path, validation: ValidationResult) -> None:
        write_text(path, GLayoutCodeAgent._validation_text(validation))

    @staticmethod
    def _attempt_record(
        attempt_index: int,
        candidate_path: Path,
        validation: ValidationResult,
    ) -> dict[str, object]:
        return {
            "attempt": attempt_index,
            "candidate_path": str(candidate_path),
            "success": validation.success,
            "stage": validation.stage,
            "summary": validation.summary,
            "stdout": validation.stdout,
            "stderr": validation.stderr,
            "gds_path": str(validation.gds_path) if validation.gds_path else None,
            "drc_pass": validation.drc_pass,
            "lvs_pass": validation.lvs_pass,
            "verification": validation.verification,
            "verification_feedback": validation.verification_feedback,
        }
