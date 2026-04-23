from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_run_directory(root: Path) -> Path:
    ensure_directory(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_text(path: Path, text: str) -> None:
    ensure_directory(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
