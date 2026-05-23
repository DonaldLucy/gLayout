from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge surrogate JSONL datasets by sample_id.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--overlay", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.base))
    by_id = {row["sample_id"]: index for index, row in enumerate(rows)}
    inserted = 0
    updated = 0
    for overlay_path in args.overlay:
        for row in _read_jsonl(Path(overlay_path)):
            sample_id = row["sample_id"]
            if sample_id in by_id:
                rows[by_id[sample_id]] = _deep_merge(rows[by_id[sample_id]], row)
                updated += 1
            else:
                by_id[sample_id] = len(rows)
                rows.append(row)
                inserted += 1
    output = Path(args.output)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"base": args.base, "output": str(output), "records": len(rows), "updated": updated, "inserted": inserted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

