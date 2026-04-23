from __future__ import annotations

from pathlib import Path
from typing import Optional

from .references import ReferenceLibrary


class PromptLibrary:
    def __init__(self, prompt_dir: Path, repo_root: Path):
        self._system_prompt = (prompt_dir / "system_prompt.txt").read_text(
            encoding="utf-8"
        )
        self._repair_prompt = (prompt_dir / "repair_prompt.txt").read_text(
            encoding="utf-8"
        )
        self._repo_guidance = (prompt_dir / "repo_guidance.txt").read_text(
            encoding="utf-8"
        )
        self._references = ReferenceLibrary(
            repo_root=repo_root,
            config_path=prompt_dir / "reference_files.json",
        )

    @property
    def reference_descriptions(self) -> list[dict[str, object]]:
        return self._references.describe()

    def references_for_task(self, task: str) -> list[dict[str, object]]:
        return self._references.describe(task)

    @staticmethod
    def task_guidance(task: str) -> Optional[str]:
        normalized = task.lower()
        if any(marker in normalized for marker in ["two fet", "2 fet", "two transistor", "merged diffusion", "merge diffusion", "shared diffusion"]):
            return (
                "Benchmark constraint for this task:\n"
                "- Implement the merged-diffusion structure yourself from primitive-level blocks.\n"
                "- Do not import or call `two_transistor_interdigitized`, `two_nfet_interdigitized`, `two_pfet_interdigitized`, "
                "`current_mirror`, `diff_pair`, or `flipped_voltage_follower`.\n"
                "- Primitive-level imports such as `multiplier`, `nmos`, `pmos`, `via_stack`, `straight_route`, "
                "`c_route`, `L_route`, and `align_comp_to_port` are allowed.\n"
                "- Scalar sizing parameters should stay as scalars, not one-element lists."
            )
        return None

    def build_generation_prompt(
        self,
        task: str,
        source_code: Optional[str] = None,
        skill_hint: Optional[str] = None,
    ) -> str:
        sections = [
            self._system_prompt.strip(),
            "Repository guidance:\n" + self._repo_guidance.strip(),
            "Repository reference snippets:\n" + self._references.render_for_prompt(task),
            f"User request:\n{task.strip()}",
        ]
        task_guidance = self.task_guidance(task)
        if task_guidance:
            sections.append(task_guidance)
        if skill_hint:
            sections.append(f"Relevant repo skill:\n{skill_hint.strip()}")
        if source_code:
            sections.append(f"Existing code to improve:\n```python\n{source_code}\n```")
        sections.append(
            "Return only Python source code. Do not wrap the answer in Markdown fences."
        )
        return "\n\n".join(section for section in sections if section.strip())

    def build_repair_prompt(
        self,
        task: str,
        previous_code: str,
        validation_log: str,
        attempt_history: Optional[str] = None,
        best_candidate_code: Optional[str] = None,
        best_candidate_summary: Optional[str] = None,
        repair_focus: Optional[str] = None,
        skill_hint: Optional[str] = None,
    ) -> str:
        sections = [
            self._repair_prompt.strip(),
            "Repository guidance:\n" + self._repo_guidance.strip(),
            "Repository reference snippets:\n" + self._references.render_for_prompt(task),
            f"Original task:\n{task.strip()}",
        ]
        task_guidance = self.task_guidance(task)
        if task_guidance:
            sections.append(task_guidance)
        if skill_hint:
            sections.append(f"Relevant repo skill:\n{skill_hint.strip()}")
        if attempt_history:
            sections.append(f"Recent attempt history:\n{attempt_history.strip()}")
        if repair_focus:
            sections.append(f"Repair focus:\n{repair_focus.strip()}")
        if best_candidate_summary:
            sections.append(f"Best candidate so far:\n{best_candidate_summary.strip()}")
        if best_candidate_code:
            sections.append(
                f"Best candidate code so far (preserve working parts unless the feedback proves they are wrong):\n```python\n{best_candidate_code}\n```"
            )
        sections.extend(
            [
                f"Previous candidate:\n```python\n{previous_code}\n```",
                f"Validator output:\n```\n{validation_log}\n```",
                "Return only corrected Python source code.",
            ]
        )
        return "\n\n".join(section for section in sections if section.strip())
