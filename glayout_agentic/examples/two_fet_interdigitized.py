from __future__ import annotations

import argparse
from pathlib import Path

from glayout.pdk.sky130_mapped import sky130_mapped_pdk
from glayout.placement.two_transistor_interdigitized import two_transistor_interdigitized


def build_two_fet_interdigitized(
    width: float = 2.0,
    length: float = 0.15,
    fingers: int = 1,
):
    pdk = sky130_mapped_pdk
    if pdk is None:
        raise RuntimeError(
            "sky130_mapped_pdk is unavailable. Set PDK_ROOT and PYTHONPATH before running."
        )
    pdk.activate()
    component = two_transistor_interdigitized(
        pdk=pdk,
        device="nfet",
        numcols=2,
        dummy=True,
        with_substrate_tap=False,
        with_tie=True,
        width=width,
        length=length,
        fingers=fingers,
    )
    component.name = "nfet_two_fet_interdigitized"
    return component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--length", type=float, default=0.15)
    parser.add_argument("--fingers", type=int, default=1)
    parser.add_argument("--output-gds", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    component = build_two_fet_interdigitized(
        width=args.width,
        length=args.length,
        fingers=args.fingers,
    )
    output_path = args.output_gds.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    component.write_gds(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
