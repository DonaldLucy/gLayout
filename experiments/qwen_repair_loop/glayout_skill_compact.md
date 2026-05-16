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
