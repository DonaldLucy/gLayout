from __future__ import annotations

import argparse
import json
from pathlib import Path

from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle

from glayout.cells.elementary.diff_pair.diff_pair import diff_pair
from glayout.pdk.sky130_mapped import sky130_mapped_pdk


def _bbox_to_list(ref) -> list[list[float]]:
    bbox = ref.bbox
    return [
        [float(bbox[0][0]), float(bbox[0][1])],
        [float(bbox[1][0]), float(bbox[1][1])],
    ]


def build_intentional_drc_fail_diff_pair(
    width: float = 3.0,
    length: float = 0.15,
    fingers: int = 4,
) -> tuple[Component, dict[str, object]]:
    pdk = sky130_mapped_pdk
    if pdk is None:
        raise RuntimeError(
            "sky130_mapped_pdk is unavailable. Set PDK_ROOT and PYTHONPATH before running."
        )
    pdk.activate()

    base = diff_pair(
        pdk,
        width=width,
        length=length,
        fingers=fingers,
        n_or_p_fet=True,
        substrate_tap=False,
    )

    top = Component("intentional_drc_fail_diff_pair")
    base_ref = top << base
    top.add_ports(base_ref.get_ports_list())

    met2_min_sep = float(pdk.get_grule("met2")["min_separation"])
    met2_min_width = float(pdk.get_grule("met2")["min_width"])

    bar_width = max(met2_min_width, 0.30)
    bar_height = max(3.0, 5.0 * met2_min_width)
    gap = met2_min_sep * 0.25

    bar = rectangle(
        size=(bar_width, bar_height),
        layer=pdk.get_glayer("met2"),
        centered=True,
    )

    hotspot_y = float(base_ref.ymax + bar_height)
    left_center_x = -gap / 2 - bar_width / 2
    right_center_x = gap / 2 + bar_width / 2

    # Intentional DRC fail hotspot:
    # These two MET2 rectangles are closer than the MET2 minimum separation rule.
    left_bar = top << bar
    left_bar.movex(left_center_x).movey(hotspot_y)
    right_bar = top << bar
    right_bar.movex(right_center_x).movey(hotspot_y)

    metadata = {
        "rule": "met2 minimum separation",
        "met2_min_separation": met2_min_sep,
        "actual_gap": gap,
        "description": (
            "Two MET2 rectangles were intentionally placed too close together "
            "to trigger a DRC spacing violation."
        ),
        "hotspot": {
            "left_bar_bbox": _bbox_to_list(left_bar),
            "right_bar_bbox": _bbox_to_list(right_bar),
            "center_y": hotspot_y,
        },
        "code_hint": {
            "file": "scripts/generate_intentional_diff_pair_drc_fail.py",
            "note": (
                "The intentional violation is created at the two rectangle placements "
                "marked by the 'Intentional DRC fail hotspot' comment."
            ),
        },
    }
    return top, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an intentionally DRC-failing diff pair layout."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/manual_intentional_diff_pair_drc_fail"),
    )
    parser.add_argument("--width", type=float, default=3.0)
    parser.add_argument("--length", type=float, default=0.15)
    parser.add_argument("--fingers", type=int, default=4)
    parser.add_argument("--skip-drc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layout, metadata = build_intentional_drc_fail_diff_pair(
        width=args.width,
        length=args.length,
        fingers=args.fingers,
    )

    gds_path = output_dir / f"{layout.name}.gds"
    layout.write_gds(gds_path)

    metadata_path = output_dir / f"{layout.name}.hotspot.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"GDS: {gds_path}")
    print(f"Hotspot metadata: {metadata_path}")

    if not args.skip_drc:
        drc_output_dir = output_dir / "magic_drc"
        result = sky130_mapped_pdk.drc_magic(
            layout,
            layout.name,
            output_file=drc_output_dir,
        )
        report_path = (
            drc_output_dir
            / "drc"
            / layout.name
            / f"{layout.name}.rpt"
        )
        print(f"Magic DRC report: {report_path}")
        print(f"Magic DRC result: {result}")


if __name__ == "__main__":
    main()
