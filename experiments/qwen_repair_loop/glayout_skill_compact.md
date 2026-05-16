# Compact gLayout Repair Skill

This is the stable repair context for Qwen-style iterative verification repair. Keep patches small, generator-local, and easy to re-run through DRC/LVS.

## Goal

Repair one gLayout cell so both baseline and traced outputs are DRC-clean and LVS-clean, without weakening verification criteria. Prefer preserving the intended schematic over making cosmetic layout-only changes.

## What gLayout Is

- gLayout is a Python layout-generation framework for analog/mixed-signal IC cells.
- It builds GDS geometry with `gdsfactory.Component` objects and maps generic layer names through a `MappedPDK`.
- It targets PDK-portable analog layout: the same generator style should work across mapped PDKs such as Sky130 when layer/rule access goes through `pdk`.
- It also carries SPICE intent through `glayout.spice.Netlist`, so DRC checks physical geometry and LVS checks layout extraction against the intended schematic.
- In this project, SMGR records provenance while the generator runs. The repair agent should use provenance source spans and port manifests to patch the Python generator, not patch generated GDS manually.

## Output Contract

- Return only a unified diff patch.
- Do not include Markdown fences or commentary in the model response.
- Touch the smallest number of files needed.
- Preserve existing public function names unless the task explicitly asks for a new candidate generator.
- Preserve SMGR/provenance behavior: do not remove decorators, netlist metadata, component names, labels, or ports that verification depends on.

## gLayout Mental Model

- A generator is a Python function that takes `pdk: MappedPDK` plus sizing/options and returns a `Component`.
- Generators often use `@cell`, `@validate_arguments`, and `@tracked_generator("name")`; do not remove these.
- A child component becomes a placed reference with `ref = parent << child`.
- A placed reference can be moved/rotated/mirrored with methods such as `movex`, `movey`, `move`, and `mirror_y`.
- After transformations, use `rename_ports_by_orientation(...)` when port names must reflect new physical directions.
- References expose ports with names such as `source_E`, `drain_N`, `multiplier_0_gate_W`, or generator-specific aliases.
- A port has a name, center, orientation, width, and layer. LVS repair usually means making the right ports physically connected and correctly labeled.
- Routing functions create geometry between compatible ports. The route must physically connect the same nets that the schematic/netlist says are connected.
- Top-level labels and pins define the circuit interface for extraction. Labels are electrical only when placed on real extracted conductor geometry.
- The SPICE/netlist side must match the layout topology. Fixing a layout route without updating the netlist mapping, or updating the netlist without the route, often leaves LVS mismatched.

## Common Imports and Objects

- Typical imports include `Component` and `cell` from `gdsfactory`, `MappedPDK`, `Netlist`, primitive generators, routing helpers, and port/geometry utilities.
- `Component(name="...")` creates a layout cell.
- `component << child` inserts a child and returns a mutable reference.
- `component.add(ref)` can add a pre-centered or transformed reference.
- `component.add_ports(ref.get_ports_list(), prefix="...")` exposes child ports under a prefix.
- `component.ports["name"]` accesses a named port; `ref.ports["name"]` accesses a transformed child port.
- `component_snap_to_grid(...)` and `pdk.snap_to_2xgrid(...)` help keep geometry legal.
- `evaluate_bbox(component_or_ref)` gives physical extents.
- `align_comp_to_port(child, port, alignment=(...))` places geometry relative to a port.
- `movex(port, destination=...)` and `movey(port, destination=...)` can create shifted port-like targets for route alignment.

## PDK and Layer Rules

- Never hard-code Sky130 layer tuples unless the surrounding file already does so for labels/pins.
- Prefer `pdk.get_glayer("met1")`, `pdk.get_glayer("met2_label")`, and similar mapped layer names.
- Use `pdk.layer_to_glayer(port.layer)` to infer a glayer from an existing port when placing a label on that exact conductor.
- Use `pdk.get_grule("met2")` or `pdk.get_grule("met2", "via1")` for spacing/enclosure/width rules when changing geometry.
- `pdk.util_max_metal_seperation()` is commonly used to space analog blocks/routes conservatively.

## Primitive and Composite Generators

- `nmos(...)`, `pmos(...)`, and `multiplier(...)` create FET layouts with ports for gate/source/drain/bulk/taps and store transistor netlist info.
- Placement helpers such as `two_nfet_interdigitized(...)`, `two_pfet_interdigitized(...)`, and `generic_4T_interdigitzed(...)` build matched/interdigitized transistor structures.
- Composite cells instantiate primitives, route between child ports, expose selected ports, add labels, and attach a hierarchical `Netlist`.
- Port prefixes such as `N_`, `P_`, `M_1_`, `ibias_`, or `purposegndports` are semantically important; do not rename them casually.

## Routing Rules

- `straight_route(pdk, port1, port2, ...)` is best for aligned ports on compatible metal paths.
- `c_route(pdk, port1, port2, ...)` requires ports to be parallel and have compatible orientation.
- If `c_route` raises "Ports must be parralel and have same orientation", choose a different helper, use a route-created port, or add an intermediate segment/via.
- `L_route(pdk, port1, port2, ...)` is useful for perpendicular or L-shaped connections.
- `smart_route(...)` can be useful when existing route helpers are insufficient, but it may be less predictable.
- When a route is created as a child, expose its useful ports if later code or labels need them: `route_ref = component << route; component.add_ports(route_ref.get_ports_list(), prefix="...")`.
- Avoid shorting bulk/source/gate/drain unless the schematic explicitly requires that short.
- If changing a route for DRC, preserve the same electrical net identity.

## Labels and Pins

- A robust label pattern is to create a small rectangle on a pin layer, add the text label to that rectangle, align it to a real port, and add the rectangle to the component.
- Example pattern: `pin = rectangle(layer=pdk.get_glayer("met2_pin"), size=(0.27, 0.27), centered=True).copy(); pin.add_label(text="VSS", layer=pdk.get_glayer("met2_label")); component.add(align_comp_to_port(pin, real_port))`.
- Another acceptable pattern is `component.add_label(text="NET", position=real_port.center, layer=pdk.get_glayer("met2_label"))` if the label lands on extracted metal.
- A label on the wrong layer or off-metal can create a top-level pin with zero fanout, which causes LVS pin mismatch.
- If the repair packet says a label is floating, move the label to the real routed conductor instead of changing net names.

## Netlist Syntax

- `Netlist(circuit_name="NAME", nodes=[...])` defines a hierarchical SPICE subckt interface.
- `netlist.connect_netlist(child.info["netlist"], [("child_pin", "TOP_NET"), ...])` adds a child and connects child nodes to top-level nets.
- Some files store the object as `component.info["netlist_obj"]`; use the helper pattern already present in that file if gdsfactory info validation is strict.
- `netlist.connect_subnets(child_a_ref, child_b_ref, [("A_NODE", "B_NODE"), ...])` connects two child sub-netlist nodes through an internal wire such as `wire0`.
- Primitive MOS netlists use nodes `D`, `G`, `S`, `B`.
- The physical route and netlist mapping must agree. If schematic says `CMIRROR.VOUT` connects to `DIFF_PAIR.VTAIL`, there must be a physical conductor connecting the corresponding ports.
- For strict LVS, top-level labels must match top-level `nodes`; topology-only "Netlists match" is not enough if pin matching fails.
- Store generated netlist strings consistently with the file style, often `component.info["netlist"] = netlist_obj.generate_netlist()`.

## From External Netlist to First gLayout Code

Use this recipe when the task starts from an outside schematic/netlist instead of an existing gLayout cell.

1. Identify top-level ports from the `.subckt` line. These become `Netlist(..., nodes=[...])` and top-level layout labels.
2. Classify devices by primitive type: NFET, PFET, resistor, capacitor, diode, BJT, or known subcell.
3. Group obvious matched devices before coding: differential pairs, current mirrors, stacked/cascode devices, transmission gates, source followers, load pairs, and bias branches.
4. Pick the closest clean archetype below. Reuse its generator shape instead of inventing geometry from scratch.
5. Instantiate primitives/composites with PDK-mapped generators.
6. Place references using symmetry and conservative spacing from `pdk.util_max_metal_seperation()`.
7. Route every schematic internal net physically. Internal schematic wires from `connect_subnets` must correspond to real metal.
8. Expose ports only when useful for top labels, parent cells, or downstream repair.
9. Add top-level labels for every `Netlist.nodes` entry on actual routed conductors.
10. Attach the hierarchical netlist and run DRC/LVS.

When converting netlists, do not try to perfectly reproduce arbitrary schematic instance order. LVS cares about equivalent topology, device classes, pins, and properties.

## Minimum Generator Skeleton

Use this as the first draft shape for a new cell:

```python
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from glayout.pdk.mappedpdk import MappedPDK
from glayout.provenance import tracked_generator
from glayout.spice.netlist import Netlist
from glayout.util.comp_utils import align_comp_to_port
from glayout.util.port_utils import rename_ports_by_orientation
from glayout.util.snap_to_grid import component_snap_to_grid


def add_example_labels(component: Component, pdk: MappedPDK) -> Component:
    component.unlock()
    for label, port_name, glayer in [
        ("VIN", "input_port_name", "met2"),
        ("VOUT", "output_port_name", "met2"),
        ("VSS", "ground_port_name", "met2"),
    ]:
        pin = rectangle(layer=pdk.get_glayer(f"{glayer}_pin"), size=(0.27, 0.27), centered=True).copy()
        pin.add_label(text=label, layer=pdk.get_glayer(f"{glayer}_label"))
        component.add(align_comp_to_port(pin, component.ports[port_name], alignment=("c", "b")))
    return component.flatten()


def example_netlist(child_a: Component, child_b: Component) -> Netlist:
    netlist = Netlist(circuit_name="EXAMPLE", nodes=["VIN", "VOUT", "VSS"])
    a_ref = netlist.connect_netlist(child_a.info["netlist"], [("D", "VOUT"), ("S", "VSS")])
    b_ref = netlist.connect_netlist(child_b.info["netlist"], [("D", "VOUT"), ("S", "VSS")])
    netlist.connect_subnets(a_ref, b_ref, [("G", "G")])
    return netlist


@tracked_generator("example_cell")
@cell
def example_cell(pdk: MappedPDK) -> Component:
    top = Component("example_cell")
    # Instantiate, place, route, expose ports, label, attach netlist.
    component = component_snap_to_grid(rename_ports_by_orientation(top))
    netlist_obj = Netlist(circuit_name="EXAMPLE", nodes=["VIN", "VOUT", "VSS"])
    component = add_example_labels(component, pdk)
    component.info["netlist"] = netlist_obj.generate_netlist()
    component.info["netlist_obj"] = netlist_obj
    component.info["netlist_data"] = {
        "circuit_name": netlist_obj.circuit_name,
        "nodes": netlist_obj.nodes,
        "source_netlist": netlist_obj.source_netlist,
    }
    return component
```

## Clean Archetype: Two-Device Current Mirror

Use this when an external netlist contains two matched transistors sharing source and bulk, with one reference device diode-connected.

Netlist intent:

```python
def current_mirror_style_netlist(pdk, width, length, fingers, multipliers, device="nfet"):
    netlist = Netlist(circuit_name="CMIRROR", nodes=["VREF", "VOUT", "VSS", "B"])
    netlist.connect_netlist(
        two_tran_interdigitized_netlist(
            pdk=pdk,
            width=width,
            length=length,
            fingers=fingers,
            multipliers=multipliers,
            with_dummy=True,
            n_or_p_fet=device,
        ),
        [
            ("VDD1", "VREF"),
            ("VG1", "VREF"),
            ("VDD2", "VOUT"),
            ("VG2", "VREF"),
            ("VSS1", "VSS"),
            ("VSS2", "VSS"),
            ("VB", "B"),
        ],
    )
    return netlist
```

Layout intent:

```python
top = Component("current_mirror")
fets = two_nfet_interdigitized(
    pdk,
    numcols=numcols,
    dummy=True,
    with_substrate_tap=False,
    with_tie=False,
    width=width,
    length=length,
    fingers=fingers,
)
top.add_ports(fets.get_ports_list(), prefix="fet_")
source_short = fets << c_route(pdk, fets.ports["A_source_E"], fets.ports["B_source_E"], viaoffset=False)
gate_short = fets << c_route(pdk, fets.ports["A_gate_W"], fets.ports["B_gate_W"], viaoffset=False)
fets << L_route(pdk, fets.ports["A_drain_W"], gate_short.ports["con_N"], viaoffset=False, fullbottom=False)
top << fets
top.add_ports(source_short.get_ports_list(), prefix="purposegndports")
```

Label intent:

- `VREF` on reference drain/gate conductor.
- `VOUT` on mirror drain.
- `VSS` on shared source.
- `B` on well/substrate tie.

Common repair mistakes:

- Mapping `B` to `VSS` in the netlist when the physical well tie is a separate top-level `B`.
- Placing `VSS` label on a route port that is not connected to transistor sources.
- Forgetting the diode connection from reference drain to shared gate.

## Clean Archetype: Transmission Gate

Use this when an external netlist has parallel NFET/PFET pass devices:

```python
def tg_style_netlist(nfet: Component, pfet: Component) -> Netlist:
    netlist = Netlist(circuit_name="Transmission_Gate", nodes=["VIN", "VSS", "VOUT", "VCC", "VGP", "VGN"])
    netlist.connect_netlist(nfet.info["netlist"], [("D", "VOUT"), ("G", "VGN"), ("S", "VIN"), ("B", "VSS")])
    netlist.connect_netlist(pfet.info["netlist"], [("D", "VOUT"), ("G", "VGP"), ("S", "VIN"), ("B", "VCC")])
    return netlist
```

Layout pattern:

```python
top = Component("transmission_gate")
nfet = nmos(pdk, width=nw, length=nl, fingers=nf, multipliers=nm, with_dummy=True, with_dnwell=False)
pfet = pmos(pdk, width=pw, length=pl, fingers=pf, multipliers=pm, with_dummy=True)
nfet_ref = top << nfet
pfet_ref = top << pfet
pfet_ref = rename_ports_by_orientation(pfet_ref.mirror_y())
pfet_ref.movey(nfet_ref.ymax + evaluate_bbox(pfet_ref)[1] / 2 + pdk.util_max_metal_seperation())
top << c_route(pdk, nfet_ref.ports["multiplier_0_source_E"], pfet_ref.ports["multiplier_0_source_E"])
top << c_route(pdk, nfet_ref.ports["multiplier_0_drain_W"], pfet_ref.ports["multiplier_0_drain_W"], viaoffset=False)
top.add_ports(nfet_ref.get_ports_list(), prefix="N_")
top.add_ports(pfet_ref.get_ports_list(), prefix="P_")
```

Label intent:

- `VIN` on the common source route.
- `VOUT` on the common drain route.
- `VGN` on NFET gate.
- `VGP` on PFET gate.
- `VSS` on NFET bulk/tap.
- `VCC` on PFET bulk/nwell/tap.

Common repair mistakes:

- Accidentally swapping `VIN` and `VOUT` labels; topology may still match if symmetric, but pin naming can fail.
- Labeling `VSS`/`VCC` on isolated taps that do not touch the device bulk in extraction.
- Using `c_route` after mirroring without refreshing port orientation names.

## Clean Archetype: Differential Pair With Tail Bias

Use this when a netlist has input pair devices sharing a tail node and a bias/current-mirror block driving that tail.

Netlist intent:

```python
def diff_pair_tail_bias_netlist(diff_pair_comp: Component, cmirror_comp: Component) -> Netlist:
    netlist = Netlist(circuit_name="DIFFPAIR_CMIRROR_BIAS", nodes=["VP", "VN", "VDD1", "VDD2", "IBIAS", "VSS", "B"])
    diff_ref = netlist.connect_netlist(diff_pair_comp.info["netlist"], [])
    cm_ref = netlist.connect_netlist(cmirror_comp.info["netlist"], [("VREF", "IBIAS"), ("B", "B"), ("VSS", "VSS")])
    netlist.connect_subnets(cm_ref, diff_ref, [("VOUT", "VTAIL")])
    return netlist
```

Layout repair pattern:

```python
route_ref = top << route_helper(
    pdk,
    top.ports["source_routeE_con_S"],  # DIFF_PAIR.VTAIL candidate
    top.ports["ibias_B_drain_N"],      # CMIRROR.VOUT candidate
)
top.add_ports(route_ref.get_ports_list(), prefix="tail_repair_")
```

Label intent:

- `VP` and `VN` on input gate conductors.
- `VDD1` and `VDD2` on pair drains.
- `IBIAS` on current mirror reference gate/drain.
- `VSS` on current mirror source rail.
- `B` on bulk/well tie.

Common repair mistakes:

- Adding labels without physically routing `CMIRROR.VOUT` to `DIFF_PAIR.VTAIL`.
- Shorting `B` to `VSS` to remove a mismatch when the layout intentionally keeps bulk on `B`.
- Mapping current mirror bulk to `VSS` while the physical layout labels bulk as `B`.

## Clean Archetype: Low-Voltage Current Mirror / FVF Composite

Use this when an external netlist contains flipped-voltage followers plus output NFET branches.

Netlist pattern:

```python
netlist = Netlist(circuit_name="Low_voltage_current_mirror", nodes=["IBIAS1", "IBIAS2", "GND", "IOUT1", "IOUT2"])
netlist.connect_netlist(get_component_netlist(bias_fvf), [("VIN", "IBIAS1"), ("VBULK", "GND"), ("Ib", "IBIAS1"), ("VOUT", "local_net_1")])
netlist.connect_netlist(get_component_netlist(cascode_fvf), [("VIN", "IBIAS1"), ("VBULK", "GND"), ("Ib", "IBIAS2"), ("VOUT", "local_net_2")])
fet_1a = netlist.connect_netlist(get_component_netlist(fet_2), [("D", "IOUT1"), ("G", "IBIAS1"), ("B", "GND")])
fet_1b = netlist.connect_netlist(get_component_netlist(fet_1), [("G", "IBIAS2"), ("S", "GND"), ("B", "GND")])
netlist.connect_subnets(fet_1a, fet_1b, [("S", "D")])
```

Layout pattern:

- Place FVF branches first, then mirrored/matched output transistors around them.
- Create vias to move gate/source/drain nets onto higher metal before long routes.
- Route local source/drain cascode nets physically for every `connect_subnets` internal connection.
- Expose child ports with stable prefixes such as `M_1_`, `M_2_`, `M_3_A_`, `M_3_B_`.
- Label `GND`, `IBIAS1`, `IBIAS2`, `IOUT1`, `IOUT2` on the actual branch conductors.

Common repair mistakes:

- Reusing `component.info["netlist"]` when a strict gdsfactory version converted it to a string; use `netlist_obj` or `netlist_data` helper pattern.
- Labeling bias nets on gate routes that are adjacent but not connected.
- Connecting output branch source/drain nets in layout but forgetting `connect_subnets` in the netlist.

## Clean Helper: Recover Netlist Across gdsfactory Versions

Some gdsfactory versions restrict `component.info`. Use this helper style when composing existing cells:

```python
def get_component_netlist(component):
    if "netlist_obj" in component.info:
        return component.info["netlist_obj"]
    if "netlist_data" in component.info:
        data = component.info["netlist_data"]
        netlist = Netlist(circuit_name=data["circuit_name"], nodes=data["nodes"])
        netlist.source_netlist = data["source_netlist"]
        return netlist
    return component.info["netlist"]
```

## Clean Helper: Robust Label Placement

Use this when the repair packet reports floating labels or when generating a first draft:

```python
def add_label_on_port(component: Component, pdk: MappedPDK, text: str, port_name: str, glayer: str | None = None, size: float = 0.27):
    component.unlock()
    port = component.ports[port_name]
    if glayer is None:
        glayer = pdk.layer_to_glayer(port.layer)
    pin = rectangle(layer=pdk.get_glayer(f"{glayer}_pin"), size=(size, size), centered=True).copy()
    pin.add_label(text=text, layer=pdk.get_glayer(f"{glayer}_label"))
    component.add(align_comp_to_port(pin, port, alignment=("c", "b")))
```

If `pdk.layer_to_glayer(port.layer)` fails, inspect nearby code and use the metal layer already used by that route.

## Clean Helper: Robust Route Attempt

Use this idea when a model-generated first draft does not know whether ports are parallel:

```python
def route_between(top: Component, pdk: MappedPDK, port1, port2, prefix: str):
    errors = []
    for builder in (smart_route, L_route, straight_route):
        try:
            route = builder(pdk, port1.copy(), port2.copy())
            ref = top << route
            top.add_ports(ref.get_ports_list(), prefix=prefix)
            return ref
        except Exception as exc:
            errors.append(f"{builder.__name__}: {exc}")
    if round(port1.orientation) % 360 == round(port2.orientation) % 360:
        route = c_route(pdk, port1.copy(), port2.copy(), viaoffset=False)
        ref = top << route
        top.add_ports(ref.get_ports_list(), prefix=prefix)
        return ref
    raise ValueError("No legal route found: " + " | ".join(errors))
```

For final production code, simplify the helper after the successful route type is known.

## First-Draft Coding Rules for External Netlists

- Start with a small set of known topology templates rather than one primitive per schematic transistor.
- For matched analog structures, use interdigitized placement helpers instead of separate unrelated `nmos`/`pmos` calls.
- Keep schematic net names readable in `Netlist.nodes` and `connect_netlist` mappings.
- Top-level supplies and biases should be labels, not only internal ports.
- Internal nets should not be labels unless the parent circuit needs them.
- Use conservative spacing first; optimize compactness only after DRC/LVS is clean.
- Create helper functions for labels and netlist recovery instead of duplicating fragile code.
- If LVS says device counts differ, check dummy settings, fingers, multipliers, and primitive netlist representation.
- If LVS says net counts differ, check missing routes and floating labels first.
- If LVS says pin matching failed but netlists matched, fix labels/pins first.
- If DRC appears after a topology fix, preserve topology and move/widen/space the specific route.

## Device and Port Glossary

Primitive MOS devices normally use schematic pin order:

- `D`: drain.
- `G`: gate.
- `S`: source.
- `B`: body/bulk.

Common layout port name fragments:

- `multiplier_0_gate_*`: gate route port for a transistor multiplier.
- `multiplier_0_source_*`: source route port.
- `multiplier_0_drain_*`: drain route port.
- `tie_*` or `welltie_*`: substrate/well tie ports.
- `tap_*`: guardring/tapring ports.
- `top_met_*` / `bottom_met_*`: ports on a via stack or vertical metal transition.
- `con_*`, `e1`, `e2`, `e3`: route helper output ports, often useful as intermediate connection points.
- Prefixes like `A_` and `B_` distinguish matched devices in interdigitized helpers.
- Prefixes like `tl_`, `tr_`, `bl_`, `br_` mean top-left, top-right, bottom-left, bottom-right in common-centroid structures.

Port orientation hints:

- `_N`, `_S`, `_E`, `_W` suffixes usually describe the physical side/direction after `rename_ports_by_orientation`.
- If you mirror a reference, port names may no longer describe physical direction until renamed.
- A route helper may require port orientations to match or oppose each other; do not ignore orientation errors.

## Choosing an Archetype From a Netlist

Use this map before writing code:

- Two same-type MOS, gates tied, source tied, one drain tied to gates: current mirror.
- NFET and PFET in parallel between the same input/output nodes with complementary gates: transmission gate.
- Two same-type input MOS with shared source/tail and two drain outputs: differential pair.
- Differential pair plus current mirror connected through tail or active load nets: composite differential/bias block.
- One transistor gate driven by input and another feeding back output/current: flipped voltage follower.
- Stacked devices sharing series source/drain nets: cascode or stacked current mirror.
- Matched four-device ABBA/BAAB arrangement: use common-centroid or interdigitized placement helper.
- Large system cell composed of known clean subcells: instantiate subcells first, then route top-level nets.

## External Netlist Translation Example

Given a schematic fragment:

```spice
.subckt SIMPLE_CM VREF VOUT VSS B
XM0 VREF VREF VSS B sky130_fd_pr__nfet_01v8 l=0.5 w=4
XM1 VOUT VREF VSS B sky130_fd_pr__nfet_01v8 l=0.5 w=4
.ends SIMPLE_CM
```

Translate it as a current mirror, not as two arbitrary devices:

```python
@tracked_generator("simple_cm")
@cell
def simple_cm(pdk: MappedPDK, width: float = 4, length: float = 0.5, fingers: int = 1, numcols: int = 2) -> Component:
    top = current_mirror(
        pdk,
        numcols=numcols,
        device="nfet",
        with_dummy=True,
        with_tie=True,
        width=width,
        length=length,
        fingers=fingers,
    )
    top.name = "simple_cm"
    return top
```

If the requested pin names differ from the clean cell, wrap and relabel:

```python
wrapper = Component("simple_cm")
cm = current_mirror(pdk, width=width, length=length, fingers=fingers, numcols=numcols)
cm_ref = wrapper << cm
wrapper.add_ports(cm_ref.get_ports_list(), prefix="cm_")
add_label_on_port(wrapper, pdk, "VREF", "cm_fet_A_drain_N")
add_label_on_port(wrapper, pdk, "VOUT", "cm_fet_B_drain_N")
add_label_on_port(wrapper, pdk, "VSS", "cm_fet_A_source_E")
add_label_on_port(wrapper, pdk, "B", "cm_welltie_S_top_met_S", size=0.5)
```

## External Netlist Diff Pair Example

Given:

```spice
.subckt DP VP VN VDD1 VDD2 VTAIL B
XM0 VDD1 VP VTAIL B sky130_fd_pr__nfet_01v8 l=1 w=6
XM1 VDD2 VN VTAIL B sky130_fd_pr__nfet_01v8 l=1 w=6
.ends DP
```

Prefer the existing differential-pair generator:

```python
dp = diff_pair(pdk, width=6, length=1, fingers=4, n_or_p_fet=True, substrate_tap=True)
```

The clean `diff_pair` implementation models four physical common-centroid device references for LVS even though the schematic has two conceptual sides. If an outside netlist has only two devices but the layout uses duplicated ABBA fingers, the reference netlist may need explicit repeated device instances so extracted device count/property comparison is clean.

Expected top labels:

- `VP`: plus input gate route.
- `VN`: minus input gate route.
- `VDD1`: left/output drain.
- `VDD2`: right/output drain.
- `VTAIL`: shared source route.
- `B`: substrate/well tie.

## External Netlist FVF Example

Given:

```spice
.subckt FVF VIN VBULK VOUT Ib
XM_IN Ib VIN VOUT VBULK sky130_fd_pr__nfet_01v8
XM_FB VOUT Ib VBULK VBULK sky130_fd_pr__nfet_01v8
.ends FVF
```

Use the FVF pattern:

```python
fvf = flipped_voltage_follower(
    pdk,
    device_type="nmos",
    placement="horizontal",
    width=(6.0, 3.0),
    length=(2.0, 2.0),
    fingers=(1, 1),
    multipliers=(2, 2),
    with_dnwell=False,
)
```

Expected top labels:

- `VIN` on input FET gate.
- `Ib` on input FET drain / feedback FET gate bias conductor.
- `VOUT` on input source / feedback drain conductor.
- `VBULK` on bulk/tie conductor.

## When to Write a Wrapper Instead of Modifying a Clean Cell

Write a wrapper around a clean existing cell when:

- The external netlist has the same topology but different pin names.
- The existing clean cell has more internal provenance and labels than a new generator would.
- You only need to add/remove top-level labels or expose selected ports.
- You are testing a repair-agent loop and want minimal risk.

Modify the underlying clean cell when:

- The topology is genuinely missing a route.
- The clean cell has a systematic netlist mapping bug.
- The primitive parameters or dummy/finger model are wrong for LVS.
- Multiple downstream cells fail from the same generator bug.

## Good Wrapper Pattern

```python
@tracked_generator("wrapped_cell")
@cell
def wrapped_cell(pdk: MappedPDK) -> Component:
    top = Component("wrapped_cell")
    child = existing_clean_cell(pdk)
    child_ref = top << child
    top.add_ports(child_ref.get_ports_list(), prefix="child_")

    # Add or adapt labels on real child conductors.
    add_label_on_port(top, pdk, "NEW_PIN", "child_existing_real_port")

    netlist_obj = Netlist(circuit_name="WRAPPED_CELL", nodes=["NEW_PIN", "VSS"])
    child_netlist = get_component_netlist(child)
    netlist_obj.connect_netlist(child_netlist, [("OLD_PIN", "NEW_PIN"), ("VSS", "VSS")])

    component = component_snap_to_grid(rename_ports_by_orientation(top))
    component.info["netlist"] = netlist_obj.generate_netlist()
    component.info["netlist_obj"] = netlist_obj
    component.info["netlist_data"] = {
        "circuit_name": netlist_obj.circuit_name,
        "nodes": netlist_obj.nodes,
        "source_netlist": netlist_obj.source_netlist,
    }
    return component
```

## DRC/LVS Debug Decision Tree

If verification fails:

1. Did Python fail before GDS? Fix imports, API signatures, route orientation, missing ports, or `component.info` handling.
2. Did GDS generate but DRC fails? Use DRC bboxes and layer hints. Fix geometry locally.
3. Did DRC pass but LVS device counts differ? Check primitive choice, dummy devices, fingers, multipliers, and explicit schematic device expansion.
4. Did device counts match but net counts differ? Check missing routes, unconnected labels, or accidental shorts.
5. Did topology match but pin matching fail? Move/add top-level labels on real conductors.
6. Did B/VSS or bulk/source mismatch appear? Check physical well tie and schematic bulk mapping.
7. Did only traced fail while baseline passes? The SMGR instrumentation or copied object handling may have changed port/object capture; preserve decorators and avoid mutating shared components unexpectedly.
8. Did both baseline and traced fail the same way? The generator itself is wrong, not the provenance runtime.

## Report-to-Code Mapping Hints

Repair packet fields and how to use them:

- `source_spans`: edit here first.
- `component_port_manifest`: use exact port names here for routes and labels.
- `unmatched_net_fingerprints.layout_fingerprint.pins`: tells which extracted devices touched a layout net.
- `unmatched_net_fingerprints.schematic_fingerprint.pins`: tells which schematic subblocks should share a net.
- `floating_label_candidates`: label exists but is not connected to real extracted devices.
- `drc_repair_hints.sample_bboxes`: local geometry bboxes, not a reason to refactor whole cell.
- `primary_hint_types`: high-level failure type; choose the matching archetype.

## Clean-Code Style for Repair Agents

- Use explicit helper names: `add_*_labels`, `*_netlist`, `get_component_netlist`.
- Keep top-level generator readable: instantiate, place, route, expose ports, attach netlist, label, return.
- Avoid hidden global state.
- Avoid broad `try/except` unless the existing code uses it for optional dummy ports.
- When trying multiple routes, record errors during experimentation, then simplify once a route is known.
- Do not leave generated GDS paths, temporary debug files, or print-heavy debugging in production patches.
- Do not rename a public generator or case id unless the task explicitly asks for a candidate script.

## Clean Cell Call Library

These are useful starting points when generating a first draft. Exact parameter names may vary by file, so inspect the imported generator before patching.

Current mirror:

```python
from glayout.cells.elementary.current_mirror import current_mirror

cm = current_mirror(
    pdk,
    numcols=3,
    device="nfet",
    with_dummy=True,
    with_tie=True,
    with_substrate_tap=False,
    width=3,
    length=0.5,
    fingers=1,
)
```

Transmission gate:

```python
from glayout.cells.elementary.transmission_gate import transmission_gate

tg = transmission_gate(
    pdk,
    width=(1.0, 1.0),
    length=(None, None),
    fingers=(1, 1),
    multipliers=(1, 1),
    substrate_tap=False,
)
```

Differential pair:

```python
from glayout.cells.elementary.diff_pair import diff_pair

dp = diff_pair(
    pdk,
    width=6.0,
    length=1.0,
    fingers=4,
    n_or_p_fet=True,
    rmult=2,
    substrate_tap=True,
)
```

Flipped voltage follower:

```python
from glayout.cells.elementary.FVF import flipped_voltage_follower

fvf = flipped_voltage_follower(
    pdk,
    device_type="nmos",
    placement="horizontal",
    width=(6.0, 3.0),
    length=(2.0, 2.0),
    fingers=(1, 1),
    multipliers=(2, 2),
    with_dnwell=False,
)
```

Low-voltage current mirror:

```python
from glayout.cells.composite.fvf_based_ota.low_voltage_cmirror import low_voltage_cmirror

lvcm = low_voltage_cmirror(
    pdk,
    width=(4.15, 1.42),
    length=2,
    fingers=(2, 1),
    multipliers=(1, 1),
)
```

Diff pair with current mirror bias:

```python
from glayout.cells.composite.diffpair_cmirror_bias import diff_pair_ibias

cell = diff_pair_ibias(
    pdk,
    half_diffpair_params=(6.0, 1.0, 4),
    diffpair_bias=(6.0, 2.0, 4),
    rmult=2,
    with_antenna_diode_on_diffinputs=0,
)
```

## Expected Pin Sets by Archetype

Use these expected top-level pins to catch missing labels before running LVS:

- Current mirror: `VREF`, `VOUT`, `VSS`, `B`.
- Transmission gate: `VIN`, `VOUT`, `VGN`, `VGP`, `VSS`, `VCC`.
- Differential pair: `VP`, `VN`, `VDD1`, `VDD2`, `VTAIL`, `B`.
- Diff pair plus bias: `VP`, `VN`, `VDD1`, `VDD2`, `IBIAS`, `VSS`, `B`.
- FVF: `VIN`, `VBULK`, `VOUT`, `Ib`.
- Low-voltage current mirror: `IBIAS1`, `IBIAS2`, `GND`, `IOUT1`, `IOUT2`.
- OTA-style system cells: usually supplies, inputs, output, and bias pins such as `AVDD`, `AVSS`, `INP`, `INM`, `VOUT`, `NB_10U`, `NBC_10U`.

If a generated component's `Netlist.nodes` contains a pin that is not labeled on a real conductor, strict LVS will fail even when internal topology is correct.

## Port Selection Heuristics

When the repair packet gives many candidate ports:

- For a top-level input gate, prefer ports containing `gate` and matching the intended side.
- For a drain output, prefer ports containing `drain` on the output-side device.
- For a source/common tail, prefer ports containing `source`, `source_route`, or route helper ports connected to source nets.
- For a current mirror reference, the reference net often touches both `A_drain` and `A_gate`.
- For a current mirror output, the output net often touches `B_drain`.
- For bulk/body, prefer `tie`, `welltie`, `tap`, `substrate`, or `well` ports, not source ports unless schematic says body is tied to source.
- For long route endpoints, use route-created ports such as `con_N`, `top_met_N`, or `bottom_met_E` if those are known to be on the desired net.

## Device Property and Count Heuristics

If LVS property mismatch appears:

- Check whether schematic uses one device with `m=N` while layout extraction emits N parallel devices.
- Check whether dummy devices are included in the schematic reference; extracted dummies may appear as devices tied to bulk.
- Check `width`, `length`, `fingers`, and `multipliers` values passed to primitive generators.
- Check if the primitive netlist emits explicit `XMAIN`/`XDUMMY` devices or relies on `m`.
- If a clean primitive already fixed property mismatch, reuse it rather than hand-writing SPICE.

If LVS device count mismatch appears:

- Compare extracted class counts before and after series/parallel merging.
- If layout has more devices, dummies or physical finger expansion may be missing from schematic.
- If schematic has more devices, the generator may not instantiate every branch or may short away a series device.
- If counts differ only after merging, topology may have an accidental short or missing internal route.

## Topology Construction Patterns

Diode-connected MOS:

```python
gate_short = top << c_route(pdk, fet_ref.ports["multiplier_0_gate_W"], fet_ref.ports["multiplier_0_drain_W"], viaoffset=False)
```

Shared source rail:

```python
source_short = top << c_route(pdk, left_ref.ports["multiplier_0_source_E"], right_ref.ports["multiplier_0_source_E"], viaoffset=False)
top.add_ports(source_short.get_ports_list(), prefix="source_rail_")
```

Gate bus through vias:

```python
via = via_stack(pdk, "met2", "met3", centered=True)
via_ref = top << via
via_ref.move(fet_ref.ports["multiplier_0_gate_W"].center)
top << straight_route(pdk, fet_ref.ports["multiplier_0_gate_W"], via_ref.ports["bottom_met_E"])
```

Common-centroid input pair:

```python
# Prefer existing `diff_pair` or `common_centroid_ab_ba` rather than hand-placing four devices.
dp = diff_pair(pdk, width=width, length=length, fingers=fingers, substrate_tap=True)
```

Substrate/well tie:

```python
tap = top << tapring(pdk, enclosed_rectangle=evaluate_bbox(top, padding=1), sdlayer="p+s/d", horizontal_glayer="met1")
top.add_ports(tap.get_ports_list(), prefix="tap_")
```

## Label Construction Patterns

Direct label on known metal:

```python
component.add_label(text="VOUT", position=component.ports["out_port"].center, layer=pdk.get_glayer("met2_label"))
```

Pin rectangle plus label:

```python
pin = rectangle(layer=pdk.get_glayer("met2_pin"), size=(0.27, 0.27), centered=True).copy()
pin.add_label(text="VOUT", layer=pdk.get_glayer("met2_label"))
component.add(align_comp_to_port(pin, component.ports["out_port"], alignment=("c", "b")))
```

Layer inferred from port:

```python
glayer = pdk.layer_to_glayer(component.ports["out_port"].layer)
pin = rectangle(layer=pdk.get_glayer(f"{glayer}_pin"), size=(0.27, 0.27), centered=True).copy()
pin.add_label(text="VOUT", layer=pdk.get_glayer(f"{glayer}_label"))
```

Avoid:

- Labels on `met2_label` when the actual conductor is `met1` and no via connects them.
- Labels centered in empty whitespace near a route.
- Labels on route helper geometry that later gets flattened away incorrectly.
- Duplicate top-level labels with the same name on disconnected nets.

## Netlist Construction Patterns

Primitive wrapper:

```python
netlist = Netlist(circuit_name="MY_CELL", nodes=["D", "G", "S", "B"])
netlist.connect_netlist(fet.info["netlist"], [("D", "D"), ("G", "G"), ("S", "S"), ("B", "B")])
```

Two children sharing an internal wire:

```python
top_netlist = Netlist(circuit_name="PAIR", nodes=["IN1", "IN2", "OUT", "VSS", "B"])
a = top_netlist.connect_netlist(child_a.info["netlist"], [("G", "IN1"), ("D", "OUT"), ("B", "B")])
b = top_netlist.connect_netlist(child_b.info["netlist"], [("G", "IN2"), ("D", "OUT"), ("B", "B")])
top_netlist.connect_subnets(a, b, [("S", "S")])
top_netlist.connect_node(a, [("S", "VSS")])
```

Child pin renamed to top pin:

```python
top_netlist.connect_netlist(child.info["netlist"], [("VIN", "INP"), ("VOUT", "OUT"), ("VBULK", "AVSS")])
```

Generated netlist attachment:

```python
component.info["netlist"] = netlist_obj.generate_netlist()
component.info["netlist_obj"] = netlist_obj
component.info["netlist_data"] = {
    "circuit_name": netlist_obj.circuit_name,
    "nodes": netlist_obj.nodes,
    "source_netlist": netlist_obj.source_netlist,
}
```

## Prompt Use Guidance for Qwen-Scale Models

For a 32K context model:

- Include this compact skill.
- Include only the current `repair_packet`, not the full locator.
- Include source spans from the packet.
- Include the verifier log tail.
- Ask for unified diff only.
- Apply one patch, rerun verification, then provide the next compact packet.

Do not ask the model to inspect the whole repo in one prompt. The loop should make the repo available in the workspace and only place the relevant source spans in context.

## Anti-Patterns

- Writing raw GDS or editing generated files instead of generator Python.
- Using hard-coded Sky130 layer tuples in new generic code.
- Removing SMGR provenance decorators to avoid trace issues.
- Treating `Netlists match uniquely` as strict success when the final result says top-level pin matching failed.
- Fixing pin mismatch by deleting labels.
- Fixing B/VSS mismatch by shorting all bulk/source nets blindly.
- Creating a new primitive for a standard topology that already has a clean archetype.
- Making a large refactor when one route or label is missing.

## LVS Repair Pattern: Missing Internal Net

Repair packet signal:

- `missing_route_for_schematic_internal_net`
- schematic fingerprint shows one net connecting pins across multiple subblocks.
- layout fingerprints show separate internal nets with partial fanout.

Repair approach:

- Identify the source code span building the relevant subblocks.
- Find matching child ports in `component_port_manifest`.
- Add a physical route between the child ports that correspond to the schematic net.
- Ensure no new top label is needed unless the schematic net is top-level.
- Re-run LVS after the route.

## LVS Repair Pattern: Floating or Misplaced Top Label

Repair packet signal:

- `floating_or_misplaced_top_label`
- layout top node exists but has zero extracted device fanout.
- schematic fingerprint for the same net has device pins.

Repair approach:

- Move the label to an actual conductor on the intended net.
- Prefer label position from a real port center on that net.
- Use the corresponding port layer via `pdk.layer_to_glayer(...)`.
- Do not merely rename the label if the conductor is wrong or floating.

## LVS Repair Pattern: Bulk/Source Mapping

Repair packet signal:

- `possible_bulk_source_net_mapping_mismatch`
- mismatches mention `B`, `VSS`, source pins, bulk pins, or substrate ties.

Repair approach:

- Check whether the schematic expects bulk on `B` or tied to source/VSS.
- Check whether physical well/substrate tie is routed to `B` or `VSS`.
- Do not short B to VSS just to satisfy one mismatch unless that is physically intended.
- If the layout already has the correct bulk tie, fix the netlist connection mapping.
- If the netlist is correct but layout is floating, move/add the physical tie.

## LVS Repair Pattern: Pin Naming Only

Repair packet signal:

- `status=top_level_pin_mismatch`
- `netlists_matched=true`
- topology matches, but top-level pins are absent or mislabeled.

Repair approach:

- Add or move labels/pins to the real top-level conductors.
- Do not change transistor topology.
- Verify final result says `Circuits match uniquely`, not just `Netlists match uniquely`.

## DRC Repair Pattern

Repair packet signal:

- `drc_repair_hints` with rule, layer hint, and sample bboxes.

Repair approach:

- Use the source span and candidate calls to find the local route/geometry.
- Fix the smallest spacing/width/enclosure issue near the layer and bbox.
- Preserve LVS connectivity while moving routes.
- If DRC and LVS conflict, prefer a route movement that preserves net identity.

## Reading the Repair Packet

- `status`, `matched`, and `netlists_matched` describe strict LVS state.
- `repair_hints` are the highest-value topology or label clues.
- `drc_repair_hints` group DRC markers by rule/layer.
- `unmatched_net_fingerprints` compare layout and schematic fanout for mismatched nets.
- `floating_label_candidates` identify labels extracted with no device fanout.
- `component_port_manifest` lists relevant provenance calls and ports.
- `source_spans` are the source locations most likely to need edits.
- Trust `source_spans` before searching broadly. Search the repo only if the packet lacks the needed function.

## Patch Quality Checklist

- Does every new route connect two ports that should share one schematic net?
- Does every top-level label sit on a real conductor with device fanout?
- Did you avoid accidental shorts between bulk, source, gate, and drain?
- Did you keep generated cell names and public APIs stable?
- Did you keep the patch small enough to understand from the next repair packet?

## If Unsure

- Prefer one targeted route or label correction over a broad refactor.
- Prefer using existing helper functions already present in the same file.
- Prefer changing the candidate cell script over core library behavior unless many cells share the same proven bug.
- Never silence verification or relax LVS/DRC criteria as a repair.
