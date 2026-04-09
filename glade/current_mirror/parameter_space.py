from __future__ import annotations

from decimal import Decimal


def decimal_range(start: str, stop: str, step: str) -> list[float]:
    values = []
    cur = Decimal(start)
    stop_dec = Decimal(stop)
    step_dec = Decimal(step)
    while cur <= stop_dec:
        values.append(float(cur))
        cur += step_dec
    return values


CURRENT_MIRROR_SPACE = {
    "device": ["nfet", "pfet"],
    "numcols": [1, 2, 3, 4, 5],
    "width": decimal_range("0.5", "20.0", "0.25"),
    "length": decimal_range("0.15", "3.95", "0.2"),
}


def iter_current_mirror_space():
    for device in CURRENT_MIRROR_SPACE["device"]:
        for numcols in CURRENT_MIRROR_SPACE["numcols"]:
            for width in CURRENT_MIRROR_SPACE["width"]:
                for length in CURRENT_MIRROR_SPACE["length"]:
                    yield {
                        "device": device,
                        "numcols": numcols,
                        "width": width,
                        "length": length,
                    }


def total_combinations() -> int:
    return (
        len(CURRENT_MIRROR_SPACE["device"])
        * len(CURRENT_MIRROR_SPACE["numcols"])
        * len(CURRENT_MIRROR_SPACE["width"])
        * len(CURRENT_MIRROR_SPACE["length"])
    )
