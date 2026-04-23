from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SkillMatch:
    name: str
    description: str
    rendered_code: str
    prompt_hint: str
    parameters: dict[str, object]


class SkillLibrary:
    def __init__(self, skill_dir: Path):
        self.skill_dir = skill_dir
        self._pair_template = skill_dir / "two_fet_shared_diffusion_template.py"
        self._pair_description = skill_dir / "two_fet_shared_diffusion.md"

    def match(self, task: str) -> Optional[SkillMatch]:
        normalized = task.lower()
        pair_markers = [
            "two fet",
            "2 fet",
            "fet pair",
            "two transistor",
            "两个fet",
            "两个 fet",
            "两个transistor",
            "两个 transistor",
        ]
        diffusion_markers = [
            "shared diffusion",
            "merge diffusion",
            "merged diffusion",
            "shared source/drain",
            "interdigitized",
            "共享扩散",
            "合并扩散",
            "共享源漏",
        ]
        if not any(marker in normalized for marker in pair_markers):
            return None
        if not any(marker in normalized for marker in diffusion_markers):
            return None

        params = self._extract_parameters(normalized)
        description = self._pair_description.read_text(encoding="utf-8")
        rendered_code = self._render_template(self._pair_template, params)
        prompt_hint = (
            description.strip()
            + "\n\nWhen this skill matches, prefer the explicit merged-diffusion "
            "generator in `glayout_agentic/examples/two_fet_shared_diffusion.py` "
            "so the shared diffusion node is physically explicit while width, "
            "length, and fingers remain runtime parameters."
        )
        return SkillMatch(
            name="two_fet_shared_diffusion",
            description=description,
            rendered_code=rendered_code,
            prompt_hint=prompt_hint,
            parameters=params,
        )

    def _extract_parameters(self, text: str) -> dict[str, object]:
        device = "pfet" if any(token in text for token in ("pfet", "pmos")) else "nfet"
        width = self._extract_float(
            text, [r"(?:width|w)\s*[:=]?\s*(\d+(?:\.\d+)?)"], default=2.0
        )
        length = self._extract_float(
            text,
            [
                r"(?:length|l)\s*[:=]?\s*(\d+(?:\.\d+)?)",
                r"channel\s+length\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            default=0.15,
        )
        fingers = self._extract_int(
            text, [r"(?:fingers|nf)\s*[:=]?\s*(\d+)"], default=1
        )
        numcols = self._extract_int(
            text,
            [
                r"(?:numcols|columns|cols)\s*[:=]?\s*(\d+)",
                r"(\d+)\s+columns",
            ],
            default=2,
        )
        with_dummy = not any(
            phrase in text
            for phrase in ("no dummy", "without dummy", "不要 dummy", "无 dummy")
        )
        return {
            "device": device,
            "width": width,
            "length": length,
            "fingers": fingers,
            "numcols": numcols,
            "with_dummy": with_dummy,
        }

    @staticmethod
    def _extract_float(text: str, patterns: list[str], default: float) -> float:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return default

    @staticmethod
    def _extract_int(text: str, patterns: list[str], default: int) -> int:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return default

    @staticmethod
    def _render_template(template_path: Path, params: dict[str, object]) -> str:
        template = template_path.read_text(encoding="utf-8")
        replacements = {
            "__DEVICE__": str(params["device"]),
            "__WIDTH__": str(params["width"]),
            "__LENGTH__": str(params["length"]),
            "__FINGERS__": str(params["fingers"]),
            "__NUMCOLS__": str(params["numcols"]),
            "__WITH_DUMMY__": "True" if params["with_dummy"] else "False",
        }
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template
