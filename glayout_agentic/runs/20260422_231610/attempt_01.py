from __future__ import annotations

import argparse
from pathlib import Path

from glayout.pdk.sky130_mapped import sky130_mapped_pdk
from glayout.placement.two_transistor_interdigitized import two_transistor_interdigitized


DEFAULT_DEVICE = "nfet"
DEFAULT_WIDTH = 2.0
DEFAULT_LENGTH = 0.15
DEFAULT_FINGERS = 1
DEFAULT_NUMCOLS = 2
DEFAULT_WITH_DUMMY = True


def build_two_fet_shared_diffusion(
    width: float = DEFAULT_WIDTH,
    length: float = DEFAULT_LENGTH,
    fingers: int = DEFAULT_FINGERS,
):
    pdk = sky130_mapped_pdk
    if pdk is None:
        raise RuntimeError(
            "sky130_mapped_pdk is unavailable. Set PDK_ROOT and PYTHONPATH before running."
        )
    pdk.activate()
    component = two_transistor_interdigitized(
        pdk=pdk,
        device=DEFAULT_DEVICE,
        numcols=DEFAULT_NUMCOLS,
        dummy=DEFAULT_WITH_DUMMY,
        with_substrate_tap=False,
        with_tie=True,
        width=width,
        length=length,
        fingers=fingers,
    )
    component.name = f"{DEFAULT_DEVICE}_two_fet_shared_diffusion"
    return component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH)
    parser.add_argument("--length", type=float, default=DEFAULT_LENGTH)
    parser.add_argument("--fingers", type=int, default=DEFAULT_FINGERS)
    parser.add_argument("--output-gds", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    component = build_two_fet_shared_diffusion(
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
