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

    def build_generation_prompt(
        self,
        task: str,
        source_code: Optional[str] = None,
        skill_hint: Optional[str] = None,
    ) -> str:
        sections = [
            self._system_prompt.strip(),
            "Repository guidance:\n" + self._repo_guidance.strip(),
            "Repository reference snippets:\n" + self._references.render_for_prompt(),
            f"User request:\n{task.strip()}",
        ]
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
        skill_hint: Optional[str] = None,
    ) -> str:
        sections = [
            self._repair_prompt.strip(),
            "Repository guidance:\n" + self._repo_guidance.strip(),
            "Repository reference snippets:\n" + self._references.render_for_prompt(),
            f"Original task:\n{task.strip()}",
        ]
        if skill_hint:
            sections.append(f"Relevant repo skill:\n{skill_hint.strip()}")
        sections.extend(
            [
                f"Previous candidate:\n```python\n{previous_code}\n```",
                f"Validator output:\n```\n{validation_log}\n```",
                "Return only corrected Python source code.",
            ]
        )
        return "\n\n".join(section for section in sections if section.strip())
