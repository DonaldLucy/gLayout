from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .artifacts import ensure_directory


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _last_failure(attempts: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for attempt in reversed(attempts):
        if not attempt.get("success"):
            return attempt
    return None


def record_training_example(
    dataset_dir: Path,
    task: str,
    final_code: str,
    final_filename: str,
    attempts: list[dict[str, Any]],
    backend: str,
    model_name: Optional[str],
    skill_name: Optional[str],
) -> None:
    failure = _last_failure(attempts)
    trace_item = {
        "task": task,
        "backend": backend,
        "model_name": model_name,
        "skill_name": skill_name,
        "final_filename": final_filename,
        "fixed_code": final_code,
        "attempts": attempts,
    }
    append_jsonl(dataset_dir / "repair_traces.jsonl", trace_item)

    if failure is None:
        legacy_item = {
            "filename": final_filename,
            "analysis": [
                {
                    "issue": "No repair required",
                    "explanation": {
                        "problem": "The generator passed validation on the first attempt.",
                        "reason": "The prompt or skill already matched the required topology.",
                        "fix": "Keep the same topology template and continue collecting harder cases.",
                    },
                }
            ],
            "fixed_code": final_code,
        }
    else:
        legacy_item = {
            "filename": final_filename,
            "analysis": [
                {
                    "issue": f"{failure.get('stage', 'unknown')} validation failure",
                    "explanation": {
                        "problem": failure.get("summary", "Validation failed."),
                        "reason": failure.get("stderr", "")[:1200],
                        "fix": "Update the generated gLayout code so it compiles, runs, and writes a GDS file.",
                    },
                }
            ],
            "fixed_code": final_code,
        }
    append_jsonl(dataset_dir / "legacy_train_records.jsonl", legacy_item)
