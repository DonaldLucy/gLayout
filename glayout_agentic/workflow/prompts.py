from __future__ import annotations

from pathlib import Path
from typing import Optional


class PromptLibrary:
    def __init__(self, prompt_dir: Path):
        self._system_prompt = (prompt_dir / "system_prompt.txt").read_text(
            encoding="utf-8"
        )
        self._repair_prompt = (prompt_dir / "repair_prompt.txt").read_text(
            encoding="utf-8"
        )

    def build_generation_prompt(
        self,
        task: str,
        source_code: Optional[str] = None,
        skill_hint: Optional[str] = None,
    ) -> str:
        sections = [self._system_prompt.strip(), f"User request:\n{task.strip()}"]
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
