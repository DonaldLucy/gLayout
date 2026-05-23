from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from surrogate_qor.collect_dataset import parse_pex_totals
else:
    from .collect_dataset import parse_pex_totals


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh PEX R/C totals in an existing surrogate JSONL dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output) if args.output else dataset.with_suffix(".refreshed.jsonl")
    updated = 0
    total = 0
    with dataset.open() as src, output.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            sample_dir = Path(record.get("sample_dir", ""))
            if sample_dir.exists():
                totals = parse_pex_totals(sample_dir)
                if totals.get("pex_spice"):
                    record.setdefault("physical", {}).setdefault("pex", {}).update(totals)
                    record["pex_pass"] = record["physical"]["pex"].get("status") == "PEX Complete"
                    updated += 1
            dst.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"input": str(dataset), "output": str(output), "records": total, "updated": updated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

