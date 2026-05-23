from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

from gdsfactory.component import Component, ComponentReference
from gdsfactory.components import rectangle

from glayout.pdk.mappedpdk import MappedPDK
from glayout.primitives.fet import nmos, pmos
from glayout.primitives.mimcap import mimcap
from glayout.primitives.via_gen import via_stack
from glayout.routing.L_route import L_route
from glayout.routing.c_route import c_route
from glayout.routing.smart_route import smart_route
from glayout.routing.straight_route import straight_route
from glayout.spice.netlist import Netlist
from glayout.util.comp_utils import align_comp_to_port


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    kind: str
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MoveSpec:
    name: str
    relation: str
    anchor: str


@dataclass(frozen=True)
class RouteSpec:
    port1: str
    port2: str
    kind: str = "smart_route"
    params: Mapping[str, object] = field(default_factory=dict)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}

    def add(self, item: tuple[str, str]) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: tuple[str, str]) -> tuple[str, str]:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: tuple[str, str], right: tuple[str, str]) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left

    def groups(self) -> dict[tuple[str, str], list[tuple[str, str]]]:
        result: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for item in sorted(self.parent):
            result.setdefault(self.find(item), []).append(item)
        return result


def build_netlisted_layout(
    pdk: MappedPDK,
    circuit_name: str,
    devices: list[DeviceSpec],
    moves: list[MoveSpec],
    routes: list[RouteSpec],
) -> Component:
    """Build a small prompt-derived layout with a matching SPICE netlist."""
    pdk.activate()
    top = Component(circuit_name)
    refs: dict[str, ComponentReference] = {}
    children: dict[str, Component] = {}
    device_kinds = {spec.name: spec.kind.lower() for spec in devices}

    for spec in devices:
        child = _make_device(pdk, spec)
        children[spec.name] = child
        refs[spec.name] = top << child

    for move in moves:
        _apply_move(pdk, refs[move.name], refs[move.anchor], move.relation)

    for name, ref in refs.items():
        top.add_ports(ref.get_ports_list(), prefix=f"{name}_")

    uf = _UnionFind()
    for spec in devices:
        for pin in _logical_pins(spec.kind):
            uf.add((spec.name, pin))
    nmos_bulk_nodes = [(spec.name, "B") for spec in devices if spec.kind.lower() == "nmos"]
    for bulk_node in nmos_bulk_nodes[1:]:
        uf.union(nmos_bulk_nodes[0], bulk_node)

    for index, route in enumerate(routes):
        node1 = _strict_port_to_node(route.port1, device_kinds)
        node2 = _strict_port_to_node(route.port2, device_kinds)
        uf.union(node1, node2)
        if route.port1 == route.port2:
            continue
        route_component = _make_route(pdk, top.ports[route.port1], top.ports[route.port2], route)
        if route_component is not None:
            route_ref = top << route_component
            top.add_ports(route_ref.get_ports_list(), prefix=f"route{index}_")

    net_names = _assign_net_names(uf, devices)
    hidden_net_names = _apply_implicit_bulk_nets(uf, devices, net_names)
    netlist = Netlist(
        circuit_name=circuit_name,
        nodes=[name for name in dict.fromkeys(net_names.values()) if name not in hidden_net_names],
    )
    for spec in devices:
        child_netlist = children[spec.name].info["netlist"]
        netlist.connect_netlist(
            child_netlist,
            [(pin, net_names[uf.find((spec.name, pin))]) for pin in _logical_pins(spec.kind)],
        )
    top.info["netlist"] = netlist

    for root, members in uf.groups().items():
        net_name = net_names[root]
        if net_name in hidden_net_names:
            continue
        anchor = _anchor_port(top, members, device_kinds)
        _add_net_label(pdk, top, net_name, anchor)

    return top


def _make_device(pdk: MappedPDK, spec: DeviceSpec) -> Component:
    kind = spec.kind.lower()
    params = dict(spec.params)
    if kind == "nmos":
        if "dnwell" in params and "with_dnwell" not in params:
            params["with_dnwell"] = params.pop("dnwell")
        params.setdefault("width", 3.0)
        params.setdefault("length", 0.5)
        params.setdefault("fingers", 2)
        params.setdefault("multipliers", 1)
        params.setdefault("with_substrate_tap", False)
        params.setdefault("with_tie", True)
        params.setdefault("with_dummy", True)
        params.setdefault("with_dnwell", False)
        params.setdefault("rmult", 1)
        return nmos(pdk, **params)
    if kind == "pmos":
        if "with_dnwell" in params and "dnwell" not in params:
            params["dnwell"] = params.pop("with_dnwell")
        params.setdefault("width", 3.0)
        params.setdefault("length", 0.5)
        params.setdefault("fingers", 2)
        params.setdefault("multipliers", 1)
        params.setdefault("with_substrate_tap", False)
        params.setdefault("with_tie", True)
        params.setdefault("with_dummy", True)
        params.setdefault("dnwell", False)
        params.setdefault("rmult", 1)
        return pmos(pdk, **params)
    if kind == "mimcap":
        params.setdefault("size", (5.0, 5.0))
        return mimcap(pdk, **params)
    raise ValueError(f"Unsupported device kind: {spec.kind}")


def _apply_move(
    pdk: MappedPDK,
    ref: ComponentReference,
    anchor: ComponentReference,
    relation: str,
) -> None:
    sep = float(pdk.util_max_metal_seperation()) + 2.0
    normalized = relation.replace("to the ", "").replace(" of", "").strip().lower()
    if normalized in {"right", "right next to"}:
        ref.movey(float(anchor.center[1]) - float(ref.center[1]))
        ref.movex(float(anchor.xmax) - float(ref.xmin) + sep)
    elif normalized == "left":
        ref.movey(float(anchor.center[1]) - float(ref.center[1]))
        ref.movex(float(anchor.xmin) - float(ref.xmax) - sep)
    elif normalized == "above":
        ref.movex(float(anchor.center[0]) - float(ref.center[0]))
        ref.movey(float(anchor.ymax) - float(ref.ymin) + sep)
    elif normalized == "below":
        ref.movex(float(anchor.center[0]) - float(ref.center[0]))
        ref.movey(float(anchor.ymin) - float(ref.ymax) - sep)
    else:
        raise ValueError(f"Unsupported move relation: {relation}")


def _make_route(pdk: MappedPDK, port1, port2, route: RouteSpec):
    kwargs = dict(route.params)
    route_kind = route.kind.lower()
    if route_kind == "smart_route":
        return smart_route(pdk, port1, port2, **kwargs)
    if route_kind == "straight_route":
        return straight_route(pdk, port1, port2, **kwargs)
    if route_kind == "c_route":
        return c_route(pdk, port1, port2, **kwargs)
    if route_kind == "l_route":
        return L_route(pdk, port1, port2, **kwargs)
    if route_kind == "highway_route":
        return _highway_route(pdk, port1, port2, route)
    if route_kind == "logical":
        return None
    raise ValueError(f"Unsupported route kind: {route.kind}")


def _highway_route(pdk: MappedPDK, port1, port2, route: RouteSpec) -> Component:
    """Connect two local ports with short endpoint vias and a high-metal path."""
    params = dict(route.params)
    glayer = str(params.get("glayer", "met4"))
    width = float(params.get("width", pdk.get_grule(glayer)["min_width"]))
    layer = pdk.get_glayer(glayer)
    escape = float(params.get("escape", pdk.util_max_metal_seperation() + 1.0))

    route_comp = Component()
    x1, y1 = _add_escape_via(route_comp, pdk, port1, glayer, escape)
    x2, y2 = _add_escape_via(route_comp, pdk, port2, glayer, escape)
    if "track_y" in params:
        track_y = float(params["track_y"])
        points = [(x1, y1), (x1, track_y), (x2, track_y), (x2, y2)]
    elif "track_x" in params:
        track_x = float(params["track_x"])
        points = [(x1, y1), (track_x, y1), (track_x, y2), (x2, y2)]
    else:
        points = [(x1, y1), (x2, y1), (x2, y2)]

    for start, end in zip(points, points[1:]):
        _add_high_metal_segment(route_comp, layer, start, end, width)
    return route_comp


def _add_escape_via(
    component: Component,
    pdk: MappedPDK,
    port,
    target_glayer: str,
    escape: float,
) -> tuple[float, float]:
    source_glayer = pdk.layer_to_glayer(port.layer)
    dx, dy = _port_direction(port)
    x0, y0 = map(float, port.center)
    x1 = x0 + dx * escape
    y1 = y0 + dy * escape
    stub_width = max(float(port.width), float(pdk.get_grule(source_glayer)["min_width"]))
    if dx:
        size = (escape + stub_width, stub_width)
    else:
        size = (stub_width, escape + stub_width)
    stub = component << rectangle(size=size, layer=port.layer, centered=True)
    stub.move(destination=((x0 + x1) / 2, (y0 + y1) / 2))
    via = component << via_stack(pdk, source_glayer, target_glayer, fullbottom=True, fulltop=True)
    via.move(destination=(x1, y1))
    return x1, y1


def _port_direction(port) -> tuple[int, int]:
    orientation = round(float(port.orientation)) % 360
    if orientation == 0:
        return (1, 0)
    if orientation == 180:
        return (-1, 0)
    if orientation == 90:
        return (0, 1)
    if orientation == 270:
        return (0, -1)
    return (0, 0)


def _add_high_metal_segment(
    component: Component,
    layer: tuple[int, int],
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
        return
    if abs(y2 - y1) < 1e-6:
        size = (max(abs(x2 - x1), width) + width, width)
        center = ((x1 + x2) / 2, y1)
    elif abs(x2 - x1) < 1e-6:
        size = (width, max(abs(y2 - y1), width) + width)
        center = (x1, (y1 + y2) / 2)
    else:
        raise ValueError("Highway route segments must be Manhattan")
    ref = component << rectangle(size=size, layer=layer, centered=True)
    ref.move(destination=center)


def _logical_pins(kind: str) -> list[str]:
    if kind.lower() in {"nmos", "pmos"}:
        return ["D", "G", "S", "B"]
    if kind.lower() == "mimcap":
        return ["V1", "V2"]
    raise ValueError(f"Unsupported device kind: {kind}")


def _strict_port_to_node(port_name: str, device_kinds: Mapping[str, str]) -> tuple[str, str]:
    device = _match_device_prefix(port_name, device_kinds)
    suffix = port_name[len(device) + 1 :]
    kind = device_kinds[device]
    if kind in {"nmos", "pmos"}:
        for token, pin in (("drain", "D"), ("gate", "G"), ("source", "S")):
            if suffix.startswith(token + "_"):
                return (device, pin)
    if kind == "mimcap":
        if suffix.startswith("top_met_"):
            return (device, "V1")
        if suffix.startswith("bottom_met_"):
            return (device, "V2")
    raise ValueError(f"Cannot map strict port to netlist node: {port_name}")


def _match_device_prefix(port_name: str, device_kinds: Mapping[str, str]) -> str:
    for name in sorted(device_kinds, key=len, reverse=True):
        if port_name.startswith(f"{name}_"):
            return name
    raise ValueError(f"Unknown device prefix in port: {port_name}")


def _assign_net_names(
    uf: _UnionFind,
    devices: list[DeviceSpec],
) -> dict[tuple[str, str], str]:
    order: list[tuple[str, str]] = []
    for spec in devices:
        for pin in _logical_pins(spec.kind):
            order.append((spec.name, pin))
    names: dict[tuple[str, str], str] = {}
    for root, members in uf.groups().items():
        first = min(members, key=order.index)
        if len(members) == 1:
            raw = f"{first[0]}_{_pin_label(first[1])}"
        else:
            raw = "_".join(f"{dev}_{_pin_label(pin)}" for dev, pin in members)
        names[root] = _sanitize_net_name(raw)
    return names


def _apply_implicit_bulk_nets(
    uf: _UnionFind,
    devices: list[DeviceSpec],
    net_names: dict[tuple[str, str], str],
) -> set[str]:
    by_name = {spec.name: spec for spec in devices}
    hidden_net_names: set[str] = set()
    for root, members in uf.groups().items():
        if not members or any(pin != "B" for _, pin in members):
            continue
        specs = [by_name[device] for device, _ in members]
        if specs and all(spec.kind.lower() in {"nmos", "pmos"} and not _has_body_tie(spec) for spec in specs):
            net_names[root] = _sanitize_net_name(f"{net_names[root]}_internal")
            hidden_net_names.add(net_names[root])
    return hidden_net_names


def _has_body_tie(spec: DeviceSpec) -> bool:
    return bool(spec.params.get("with_tie", True) or spec.params.get("with_substrate_tap", False))


def _pin_label(pin: str) -> str:
    return {"D": "drain", "G": "gate", "S": "source", "B": "bulk", "V1": "top", "V2": "bottom"}[pin]


def _sanitize_net_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", raw).upper()


def _anchor_port(
    top: Component,
    members: list[tuple[str, str]],
    device_kinds: Mapping[str, str],
):
    for device, pin in members:
        candidates = _anchor_candidates(device, pin, device_kinds[device])
        for candidate in candidates:
            if candidate in top.ports:
                return top.ports[candidate]
    raise ValueError(f"Could not find a label anchor for net members {members}")


def _anchor_candidates(device: str, pin: str, kind: str) -> list[str]:
    if kind in {"nmos", "pmos"}:
        if pin == "B":
            return [
                f"{device}_tie_E_top_met_E",
                f"{device}_tie_W_top_met_W",
                f"{device}_tie_N_top_met_N",
                f"{device}_tie_S_top_met_S",
                f"{device}_well_N",
                f"{device}_well_E",
                f"{device}_well_W",
                f"{device}_well_S",
            ]
        base = {"D": "drain", "G": "gate", "S": "source"}[pin]
        return [f"{device}_{base}_{direction}" for direction in ("E", "W", "N", "S")]
    if kind == "mimcap":
        base = "top_met" if pin == "V1" else "bottom_met"
        return [f"{device}_{base}_{direction}" for direction in ("E", "W", "N", "S")]
    return []


def _add_net_label(pdk: MappedPDK, top: Component, net_name: str, port) -> None:
    glayer = pdk.layer_to_glayer(port.layer)
    pin_layer = f"{glayer}_pin"
    label_layer = f"{glayer}_label"
    if pin_layer not in pdk.glayers or label_layer not in pdk.glayers:
        pin_layer = "met2_pin"
        label_layer = "met2_label"
    size = max(0.27, float(port.width))
    label = rectangle(layer=pdk.get_glayer(pin_layer), size=(size, size), centered=True).copy()
    label.add_label(text=net_name, layer=pdk.get_glayer(label_layer))
    top.add(align_comp_to_port(label, port, alignment=("c", "c")))
