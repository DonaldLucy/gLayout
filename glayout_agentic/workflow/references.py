from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceSnippet:
    path: str
    purpose: str
    start: int
    end: int
    text: str


class ReferenceLibrary:
    def __init__(self, repo_root: Path, config_path: Path):
        self.repo_root = repo_root.resolve()
        self.config_path = config_path.resolve()
        self._snippets = self._load_config()

    def _load_config(self) -> list[ReferenceSnippet]:
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        snippets: list[ReferenceSnippet] = []
        for item in payload:
            rel_path = item["path"]
            start = int(item["start"])
            end = int(item["end"])
            purpose = item["purpose"]
            lines = (self.repo_root / rel_path).read_text(encoding="utf-8").splitlines()
            snippet_text = "\n".join(lines[start - 1 : end])
            snippets.append(
                ReferenceSnippet(
                    path=rel_path,
                    purpose=purpose,
                    start=start,
                    end=end,
                    text=snippet_text,
                )
            )
        return snippets

    @property
    def snippets(self) -> list[ReferenceSnippet]:
        return self._snippets

    def render_for_prompt(self) -> str:
        blocks = []
        for snippet in self._snippets:
            blocks.append(
                "\n".join(
                    [
                        f"Reference file: {snippet.path}:{snippet.start}-{snippet.end}",
                        f"Purpose: {snippet.purpose}",
                        "```python",
                        snippet.text,
                        "```",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "path": snippet.path,
                "purpose": snippet.purpose,
                "start": snippet.start,
                "end": snippet.end,
            }
            for snippet in self._snippets
        ]
