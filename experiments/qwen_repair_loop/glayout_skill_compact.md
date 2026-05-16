# Compact gLayout Repair Skill

This is the stable repair context for Qwen-style iterative verification repair. Keep patches small, generator-local, and easy to re-run through DRC/LVS.

## Goal

Repair one gLayout cell so both baseline and traced outputs are DRC-clean and LVS-clean, without weakening verification criteria. Prefer preserving the intended schematic over making cosmetic layout-only changes.

## Output Contract

- Return only a unified diff patch.
- Do not include Markdown fences or commentary in the model response.
- Touch the smallest number of files needed.
- Preserve existing public function names unless the task explicitly asks for a new candidate generator.
- Preserve SMGR/provenance behavior: do not remove decorators, netlist metadata, component names, labels, or ports that verification depends on.

## gLayout Mental Model

- A generator returns a `gdsfactory.Component`.
- Subgenerators are inserted with `component << child_component`.
- References expose ports with names such as `source_E`, `drain_N`, `multiplier_0_gate_W`, or generator-specific aliases.
- Routing functions create geometry between compatible ports. The route must physically connect the same nets that the schematic/netlist says are connected.
- Labels are electrical only when placed on real extracted conductor geometry. A label can exist in the extracted top node list but still have zero device fanout if it is misplaced or floating.
- The SPICE/netlist side must match the layout topology. Fixing a layout route without updating the netlist mapping, or updating the netlist without the route, often leaves LVS mismatched.

## Common APIs

- `Component()` creates a new component.
- `component.add_ports(ref.get_ports_list(), prefix="...")` exposes child ports.
- `component.add_label(text=..., position=..., layer=...)` adds a layout label.
- Use `pdk.layer_to_glayer(port.layer)` when converting an existing port layer into a glayer for labels/pins.
- `straight_route(pdk, port1, port2, ...)` is best for aligned parallel ports.
- `c_route(pdk, port1, port2, ...)` requires ports to be parallel and compatible in orientation.
- If `c_route` raises orientation errors, use an existing route helper with correct orientation, create intermediate ports, or choose a compatible pair of ports.
- `L_route`, `smart_route`, or manual metal/via helpers may be better when ports are perpendicular or offset.
- Avoid shorting bulk/source/gate/drain unless the schematic explicitly requires that short.

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
