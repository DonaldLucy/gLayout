# Research generators

Each generator `<name>.py` is a new high-level gLayout assembly. Its electrical reference `<name>_reference.py` is written independently and is never derived from the layout's `info['netlist']`. The assemblies share only the gLayout MOS, via and MIM primitives, gdsfactory geometry and `terminal_access.py`. None of them calls an existing gLayout OTA, differential pair or current-mirror cell, or another assembly in this directory.

| Generator | Circuit |
| --- | --- |
| `five_t.py` | Five-transistor OTA: NMOS input pair, PMOS mirror load, NMOS tail and diode reference. Six MOS groups in one row with separated vertical tracks and horizontal buses. Ports `VDD GND IBIAS VP VN VOUT`. |
| `compact_miller.py` | Two-stage Miller OTA with one MIM compensation capacitor, in two rows. Ports `VDD GND IBIASN IBIASP VP VN VOUT`. |
| `telescopic_ota.py` | Telescopic OTA with two vertical cascode branches. |
| `cascode_mirror.py` | Three-column stacked cascode current mirror. |
| `common_source_stage.py` | Common-source stage with PMOS reference/load above an offset NMOS signal branch. |
| `resistive_diff.py` | Four-group differential stage with separated inputs and bias. |
| `source_follower.py` | L-shaped common-drain buffer with a separate sink/reference pair. |
| `stacked_switch.py` | Vertical complementary switch with two shared signal columns. |
| `terminal_access.py` | Shared MOS terminal access and Manhattan drawing helpers; it encodes no circuit graph, placement or routing policy. |
