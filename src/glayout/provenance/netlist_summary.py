from __future__ import annotations

import hashlib
import re
from typing import Any, Optional


DEFAULT_MAX_INSTANCES = 96
DEFAULT_MAX_NETS = 160
DEFAULT_MAX_PINS_PER_INSTANCE = 24
DEFAULT_MAX_PARAMS = 24


_MOS_MODEL_RE = re.compile(r"(?:^|__)n?pfet|(?:^|__)nfet|(?:^|__)pfet", re.IGNORECASE)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _compact_list(values: list[Any], max_items: int) -> tuple[list[Any], int, bool]:
    total = len(values)
    return values[:max_items], total, total > max_items


def _compact_parameters(parameters: Any, max_items: int = DEFAULT_MAX_PARAMS) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {}
    payload: dict[str, Any] = {}
    for key in sorted(parameters.keys(), key=str)[:max_items]:
        value = parameters[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[str(key)] = value
        elif isinstance(value, (list, tuple)):
            payload[str(key)] = list(value[:8])
        else:
            payload[str(key)] = repr(value)
    if len(parameters) > max_items:
        payload["_truncated"] = True
        payload["_total"] = len(parameters)
    return payload


def _build_fanout(
    instances: list[dict[str, Any]],
    max_nets: int,
    max_pins_per_net: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    fanout: dict[str, list[dict[str, Any]]] = {}
    for instance in instances:
        for pin in instance.get("pin_connections", []):
            net = _safe_str(pin.get("net"))
            if not net:
                continue
            fanout.setdefault(net, []).append(
                {
                    "instance": instance.get("name"),
                    "circuit_name": instance.get("circuit_name"),
                    "pin": pin.get("pin"),
                }
            )

    rows: list[dict[str, Any]] = []
    for net in sorted(fanout.keys())[:max_nets]:
        pins, pin_count, pins_truncated = _compact_list(fanout[net], max_pins_per_net)
        rows.append(
            {
                "net": net,
                "pin_count": pin_count,
                "pins_truncated": pins_truncated,
                "pins": pins,
            }
        )
    return rows, len(fanout), len(fanout) > max_nets


def _instance_from_tokens(
    name: str,
    pins: list[str],
    circuit_name: str,
    subckt_pin_map: dict[str, list[str]],
    max_pins_per_instance: int,
) -> dict[str, Any]:
    pin_names = subckt_pin_map.get(circuit_name)
    if pin_names is None and len(pins) == 4 and _MOS_MODEL_RE.search(circuit_name):
        pin_names = ["D", "G", "S", "B"]
    if pin_names is None:
        pin_names = [f"pin_{index}" for index in range(len(pins))]
    connections: list[dict[str, str]] = []
    for pin_name, net_name in list(zip(pin_names, pins))[:max_pins_per_instance]:
        connections.append({"pin": pin_name, "net": net_name})
    return {
        "name": name,
        "circuit_name": circuit_name,
        "pin_count": len(pins),
        "pins_truncated": len(pins) > max_pins_per_instance,
        "pin_connections": connections,
    }


def summarize_netlist(
    netlist: Any,
    *,
    source: str = "netlist_obj",
    max_instances: int = DEFAULT_MAX_INSTANCES,
    max_nets: int = DEFAULT_MAX_NETS,
    max_pins_per_instance: int = DEFAULT_MAX_PINS_PER_INSTANCE,
) -> Optional[dict[str, Any]]:
    if netlist is None:
        return None

    nodes = list(getattr(netlist, "nodes", []) or [])
    node_rows, node_count, nodes_truncated = _compact_list(nodes, max_nets)
    sub_netlists = list(getattr(netlist, "sub_netlists", []) or [])
    netlist_connections = list(getattr(netlist, "netlist_connections", []) or [])
    instance_rows: list[dict[str, Any]] = []

    for index, sub_netlist in enumerate(sub_netlists[:max_instances]):
        sub_nodes = list(getattr(sub_netlist, "nodes", []) or [])
        connection_row = (
            list(netlist_connections[index])
            if index < len(netlist_connections)
            else list(sub_nodes)
        )
        pin_connections: list[dict[str, str]] = []
        for pin_name, net_name in list(zip(sub_nodes, connection_row))[:max_pins_per_instance]:
            pin_connections.append({"pin": _safe_str(pin_name), "net": _safe_str(net_name)})
        instance_rows.append(
            {
                "name": str(index),
                "circuit_name": _safe_str(getattr(sub_netlist, "circuit_name", None)),
                "node_count": len(sub_nodes),
                "pin_count": len(connection_row),
                "pins_truncated": len(connection_row) > max_pins_per_instance,
                "pin_connections": pin_connections,
                "parameters": _compact_parameters(getattr(sub_netlist, "parameters", {})),
            }
        )

    net_fanout, net_count, nets_truncated = _build_fanout(
        instance_rows,
        max_nets=max_nets,
        max_pins_per_net=max_pins_per_instance,
    )

    source_netlist = getattr(netlist, "source_netlist", "") or ""
    return {
        "source": source,
        "circuit_name": _safe_str(getattr(netlist, "circuit_name", None)),
        "nodes": node_rows,
        "node_count": node_count,
        "nodes_truncated": nodes_truncated,
        "instance_count": len(sub_netlists),
        "instances": instance_rows,
        "instances_truncated": len(sub_netlists) > max_instances,
        "net_count": net_count,
        "net_fanout": net_fanout,
        "nets_truncated": nets_truncated,
        "parameters": _compact_parameters(getattr(netlist, "parameters", {})),
        "source_netlist_sha256": (
            hashlib.sha256(source_netlist.encode("utf-8")).hexdigest()
            if source_netlist
            else None
        ),
    }


def summarize_component_netlist(component: Any) -> Optional[dict[str, Any]]:
    info = getattr(component, "info", {}) or {}
    parent = getattr(component, "parent", None)
    parent_info = getattr(parent, "info", {}) if parent is not None else {}

    netlist = info.get("netlist_obj") or parent_info.get("netlist_obj")
    if netlist is not None:
        return summarize_netlist(netlist, source="component.info.netlist_obj")

    netlist = info.get("netlist") or parent_info.get("netlist")
    if netlist is not None and hasattr(netlist, "nodes"):
        return summarize_netlist(netlist, source="component.info.netlist")
    if isinstance(netlist, str):
        summary = parse_spice_netlist_summary(netlist)
        if summary is not None:
            summary["source"] = "component.info.netlist_spice"
        return summary

    data = info.get("netlist_data") or parent_info.get("netlist_data")
    if isinstance(data, dict):
        return {
            "source": "component.info.netlist_data",
            "circuit_name": _safe_str(data.get("circuit_name")),
            "nodes": list(data.get("nodes", [])[:DEFAULT_MAX_NETS]),
            "node_count": len(data.get("nodes", []) or []),
            "nodes_truncated": len(data.get("nodes", []) or []) > DEFAULT_MAX_NETS,
            "instance_count": 0,
            "instances": [],
            "instances_truncated": False,
            "net_count": 0,
            "net_fanout": [],
            "nets_truncated": False,
            "parameters": _compact_parameters(data.get("parameters", {})),
            "source_netlist_sha256": (
                hashlib.sha256(str(data.get("source_netlist", "")).encode("utf-8")).hexdigest()
                if data.get("source_netlist")
                else None
            ),
        }

    return None


def parse_spice_netlist_summary(
    spice_text: str,
    *,
    circuit_name: Optional[str] = None,
    max_instances: int = DEFAULT_MAX_INSTANCES,
    max_nets: int = DEFAULT_MAX_NETS,
    max_pins_per_instance: int = DEFAULT_MAX_PINS_PER_INSTANCE,
) -> Optional[dict[str, Any]]:
    subckts: dict[str, dict[str, Any]] = {}
    current: Optional[dict[str, Any]] = None
    continuation = ""

    for raw_line in spice_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+"):
            continuation += " " + line[1:].strip()
            continue
        if continuation:
            line = continuation + " " + line
            continuation = ""
        lower = line.lower()
        if lower.startswith(".include") or lower.startswith(".global"):
            continue
        if lower.startswith(".subckt"):
            parts = line.split()
            if len(parts) >= 2:
                name = parts[1]
                pins = [part for part in parts[2:] if "=" not in part]
                current = {"circuit_name": name, "nodes": pins, "instances": []}
                subckts[name] = current
            continue
        if lower.startswith(".ends"):
            current = None
            continue
        if current is None:
            continue

        parts = line.split()
        if not parts:
            continue
        token = parts[0]
        if not token or token[0].upper() not in {"X", "M", "R", "C", "D"}:
            continue
        non_param = [part for part in parts[1:] if "=" not in part]
        if len(non_param) < 2:
            continue
        if token[0].upper() == "X":
            inst_circuit = non_param[-1]
            pins = non_param[:-1]
        elif token[0].upper() == "M" and len(non_param) >= 5:
            inst_circuit = non_param[4]
            pins = non_param[:4]
        else:
            inst_circuit = token[0].upper()
            pins = non_param[:2]
        current["instances"].append(
            {
                "name": token,
                "circuit_name": inst_circuit,
                "pins": pins,
            }
        )

    if not subckts:
        return None
    selected = subckts.get(circuit_name) if circuit_name else None
    if selected is None:
        selected = list(subckts.values())[-1]
    subckt_pin_map = {name: payload.get("nodes", []) for name, payload in subckts.items()}

    instances = [
        _instance_from_tokens(
            instance["name"],
            instance["pins"],
            instance["circuit_name"],
            subckt_pin_map,
            max_pins_per_instance,
        )
        for instance in selected.get("instances", [])[:max_instances]
    ]
    net_fanout, net_count, nets_truncated = _build_fanout(
        instances,
        max_nets=max_nets,
        max_pins_per_net=max_pins_per_instance,
    )
    nodes = list(selected.get("nodes", []))
    node_rows, node_count, nodes_truncated = _compact_list(nodes, max_nets)
    return {
        "source": "spice_text",
        "circuit_name": selected.get("circuit_name"),
        "nodes": node_rows,
        "node_count": node_count,
        "nodes_truncated": nodes_truncated,
        "instance_count": len(selected.get("instances", [])),
        "instances": instances,
        "instances_truncated": len(selected.get("instances", [])) > max_instances,
        "net_count": net_count,
        "net_fanout": net_fanout,
        "nets_truncated": nets_truncated,
        "parameters": {},
        "source_netlist_sha256": hashlib.sha256(spice_text.encode("utf-8")).hexdigest(),
    }
