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
    keywords: tuple[str, ...]


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
            keywords = tuple(item.get("keywords", []))
            lines = (self.repo_root / rel_path).read_text(encoding="utf-8").splitlines()
            snippet_text = "\n".join(lines[start - 1 : end])
            snippets.append(
                ReferenceSnippet(
                    path=rel_path,
                    purpose=purpose,
                    start=start,
                    end=end,
                    text=snippet_text,
                    keywords=keywords,
                )
            )
        return snippets

    @property
    def snippets(self) -> list[ReferenceSnippet]:
        return self._snippets

    def select(self, task: str) -> list[ReferenceSnippet]:
        normalized = task.lower()
        matched = [
            snippet
            for snippet in self._snippets
            if any(keyword in normalized for keyword in snippet.keywords)
        ]
        return matched or self._snippets

    def render_for_prompt(self, task: str) -> str:
        blocks = []
        for snippet in self.select(task):
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

    def describe(self, task: str | None = None) -> list[dict[str, object]]:
        snippets = self.select(task) if task else self._snippets
        return [
            {
                "path": snippet.path,
                "purpose": snippet.purpose,
                "start": snippet.start,
                "end": snippet.end,
            }
            for snippet in snippets
        ]
