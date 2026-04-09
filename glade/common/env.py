from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root_from_file(anchor: Path) -> Path:
    for candidate in (anchor, *anchor.parents):
        if (candidate / "setup.py").is_file() and (candidate / "src" / "glayout").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {anchor}")


def prepare_chipathon_environment(repo_root: Path) -> dict[str, Path]:
    """Prepare a stable environment for gLayout evaluation inside the chipathon container."""
    pdk_root = Path("/foss/pdks")
    magicrc = pdk_root / "sky130A" / "libs.tech" / "magic" / "sky130A.magicrc"
    lvs_setup = pdk_root / "sky130A" / "libs.tech" / "netgen" / "sky130A_setup.tcl"
    lvs_ref = repo_root / "src" / "glayout" / "pdk" / "sky130_mapped" / "sky130_fd_sc_hd.spice"
    run_pex = repo_root / "src" / "glayout" / "verification" / "run_pex.sh"

    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    os.environ["PYTHONPATH"] = str(repo_root / "src")
    os.environ["PDK_ROOT"] = str(pdk_root)
    os.environ["PDKPATH"] = str(pdk_root)
    os.environ["PDK"] = "sky130A"
    os.environ["MAGIC_PDK_ROOT"] = str(pdk_root)
    os.environ["NETGEN_PDK_ROOT"] = str(pdk_root)
    os.environ.pop("CAD_ROOT", None)

    return {
        "pdk_root": pdk_root,
        "magicrc": magicrc,
        "lvs_setup": lvs_setup,
        "lvs_ref": lvs_ref,
        "run_pex": run_pex,
    }
