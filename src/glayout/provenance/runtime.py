from __future__ import annotations

import functools
import hashlib
import inspect
import itertools
import json
import math
import os
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

try:
    from gdsfactory.component import Component
except Exception:  # pragma: no cover - exercised only in minimal environments
    Component = None  # type: ignore[assignment]


_CALL_STACK: ContextVar[tuple[str, ...]] = ContextVar("glayout_smgr_call_stack", default=())
_SOURCE_MAP_VERSION = "smgr.v1"
_ENV_ENABLE_KEYS = ("GLAYOUT_SMGR", "GLAYOUT_ENABLE_SMGR", "GLAYOUT_PROVENANCE")
_ENV_CAPTURE_POLYGONS = "GLAYOUT_SMGR_CAPTURE_POLYGONS"
_ENV_CAPTURE_PORT_OBJECTS = "GLAYOUT_SMGR_CAPTURE_PORT_OBJECTS"
_ENV_CAPTURE_LIVE_REFS = "GLAYOUT_SMGR_CAPTURE_LIVE_REFS"


def _normalize_bool_env(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _safe_round(value: Any, digits: int = 6) -> Optional[float]:
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _bbox_from_points(points: Iterable[Any]) -> Optional[list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except Exception:
            continue
    if not xs or not ys:
        return None
    return [
        round(min(xs), 6),
        round(min(ys), 6),
        round(max(xs), 6),
        round(max(ys), 6),
    ]


def _bbox_from_object(obj: Any) -> Optional[list[float]]:
    bbox = getattr(obj, "bbox", None)
    if bbox is None:
        return None
    try:
        return [
            round(float(bbox[0][0]), 6),
            round(float(bbox[0][1]), 6),
            round(float(bbox[1][0]), 6),
            round(float(bbox[1][1]), 6),
        ]
    except Exception:
        return None


def _bbox_to_size(bbox: Optional[list[float]]) -> Optional[list[float]]:
    if not bbox:
        return None
    return [round(bbox[2] - bbox[0], 6), round(bbox[3] - bbox[1], 6)]


def _serialize_port(port: Any) -> dict[str, Any]:
    center = getattr(port, "center", None)
    layer = getattr(port, "layer", None)
    return {
        "name": getattr(port, "name", None),
        "center": [
            _safe_round(center[0]) if center is not None else None,
            _safe_round(center[1]) if center is not None else None,
        ],
        "width": _safe_round(getattr(port, "width", None)),
        "orientation": _safe_round(getattr(port, "orientation", None)),
        "layer": _serialize_layer(layer),
        "port_type": getattr(port, "port_type", None),
    }


def _port_priority(name: str) -> tuple[int, int, str]:
    penalty = 0
    if "array_" in name:
        penalty += 20
    if "private" in name:
        penalty += 10
    if name.startswith("_"):
        penalty += 5
    return (penalty, len(name), name)


def _serialize_layer(layer: Any) -> Any:
    if layer is None:
        return None
    if isinstance(layer, (tuple, list)) and len(layer) == 2:
        try:
            return [int(layer[0]), int(layer[1])]
        except Exception:
            return [layer[0], layer[1]]
    return layer


def _hash_file(path: str) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None


def _hash_payload(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _serialize_value(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return repr(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return repr(value)
        return round(value, 6)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _serialize_value(val, depth + 1)
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item, depth + 1) for item in value]
    if hasattr(value, "name") and hasattr(value, "grules") and hasattr(value, "glayers"):
        return {
            "kind": value.__class__.__name__,
            "name": getattr(value, "name", None),
        }
    if hasattr(value, "name") and hasattr(value, "ports"):
        return {
            "kind": value.__class__.__name__,
            "name": getattr(value, "name", None),
            "bbox": _bbox_from_object(value),
        }
    if hasattr(value, "name") and hasattr(value, "center") and hasattr(value, "width"):
        return _serialize_port(value)
    return repr(value)


def _coerce_component_like(value: Any) -> Any:
    if value is not None and hasattr(value, "bbox") and hasattr(value, "ports"):
        return value
    if isinstance(value, tuple) and value:
        head = value[0]
        if hasattr(head, "bbox") and hasattr(head, "ports"):
            return head
    return None


def _unwrap_callable(func: Callable[..., Any]) -> Callable[..., Any]:
    try:
        return inspect.unwrap(func)
    except Exception:
        return func


def _infer_edit_handles(generator_id: str, params: dict[str, Any]) -> list[str]:
    candidate_keys = [
        "glayer1",
        "glayer2",
        "e1glayer",
        "e2glayer",
        "cglayer",
        "horizontal_glayer",
        "vertical_glayer",
        "route_layer",
        "width",
        "width1",
        "width2",
        "cwidth",
        "extension",
        "spacing",
        "margin",
        "pin_side",
        "side",
        "sides",
        "with_dummy",
        "dummy",
    ]
    handles = [key for key in candidate_keys if key in params]
    if "route" in generator_id and "extension" not in handles:
        handles.append("extension")
    return handles


def _infer_object_type(generator_id: str) -> str:
    lowered = generator_id.lower()
    if "via" in lowered:
        return "via"
    if "route" in lowered:
        return "route"
    if "guard" in lowered or "tapring" in lowered:
        return "guardring"
    if "port" in lowered:
        return "port"
    return "component"


def _infer_object_layer(generator_id: str, params: dict[str, Any]) -> Any:
    lowered = generator_id.lower()
    if "via" in lowered:
        return (
            params.get("glayer2")
            or params.get("glayer1")
            or params.get("top_layer")
            or params.get("bottom_layer")
        )
    if "straight_route" in lowered:
        return params.get("glayer1") or params.get("glayer2")
    if "c_route" in lowered:
        return params.get("cglayer") or params.get("e1glayer") or params.get("e2glayer")
    if "l_route" in lowered:
        return params.get("hglayer") or params.get("vglayer")
    if "guard" in lowered or "tapring" in lowered:
        return params.get("horizontal_glayer") or params.get("vertical_glayer") or params.get("sdlayer")
    return None


def _summarize_component_netlist(component: Any) -> Optional[dict[str, Any]]:
    try:
        from glayout.provenance.netlist_summary import summarize_component_netlist
    except Exception:
        return None
    try:
        return summarize_component_netlist(component)
    except Exception:
        return None


def _intersects(lhs: Iterable[float], rhs: Iterable[float]) -> bool:
    l = list(lhs)
    r = list(rhs)
    return not (l[2] < r[0] or l[0] > r[2] or l[3] < r[1] or l[1] > r[3])


def _intersection_area(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    l = list(lhs)
    r = list(rhs)
    x0 = max(l[0], r[0])
    y0 = max(l[1], r[1])
    x1 = min(l[2], r[2])
    y1 = min(l[3], r[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


class ProvenanceSnapshot:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.calls = payload.get("calls", {})
        self.objects = payload.get("objects", {})
        self.artifacts = payload.get("artifacts", {})
        self.pdk = payload.get("pdk", {})
        self.source_hashes = payload.get("source_hashes", {})

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "ProvenanceSnapshot":
        return cls(json.loads(Path(path).read_text()))

    def get_call(self, call_id: str) -> Optional[dict[str, Any]]:
        return self.calls.get(call_id)

    def get_objects_by_call(self, call_id: str) -> list[dict[str, Any]]:
        return [
            obj for obj in self.objects.values()
            if obj.get("generated_by", {}).get("call_id") == call_id
        ]

    def query_objects_by_bbox(
        self,
        bbox: Iterable[float],
        layer_hint: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query_bbox = list(bbox)
        matches: list[dict[str, Any]] = []
        for obj in self.objects.values():
            object_bbox = obj.get("bbox")
            if not object_bbox or not _intersects(object_bbox, query_bbox):
                continue
            if layer_hint is not None:
                layer_value = obj.get("layer")
                if layer_value != layer_hint and layer_value != [layer_hint]:
                    continue
            score = _intersection_area(object_bbox, query_bbox)
            record = dict(obj)
            record["_match_score"] = round(score, 6)
            matches.append(record)
        matches.sort(key=lambda item: item.get("_match_score", 0.0), reverse=True)
        return matches

    def rank_candidate_calls(
        self,
        marker_bbox: Iterable[float],
        rule_name: Optional[str] = None,
        layer_hint: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}
        for obj in self.query_objects_by_bbox(marker_bbox, layer_hint=layer_hint):
            call_id = obj.get("generated_by", {}).get("call_id")
            if not call_id:
                continue
            call = self.calls.get(call_id, {})
            current = scores.setdefault(
                call_id,
                {
                    "call_id": call_id,
                    "generator_id": call.get("generator_id"),
                    "score": 0.0,
                    "object_ids": [],
                    "rule_name": rule_name,
                    "callsite": call.get("callsite"),
                },
            )
            boost = 1.0
            if obj.get("object_type") in {"route", "via", "polygon"}:
                boost += 0.5
            if call.get("parent_call_id"):
                boost += 0.1
            current["score"] += obj.get("_match_score", 0.0) * boost
            current["object_ids"].append(obj.get("object_id"))
        ranked = list(scores.values())
        ranked.sort(key=lambda item: item["score"], reverse=True)
        for entry in ranked:
            entry["score"] = round(entry["score"], 6)
        return ranked


class SourceMappedGeneratorRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._installed = False
        self.enabled = False
        self.auto_emit_sidecar = True
        self.capture_polygon_events = False
        self.capture_port_events = False
        self.capture_live_reference_events = False
        self.max_ports_per_component_record = 256
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.session_id = time.strftime("%Y%m%d_%H%M%S")
            self.call_records: dict[str, dict[str, Any]] = {}
            self.object_records: dict[str, dict[str, Any]] = {}
            self.source_hashes: dict[str, str] = {}
            self.artifacts: dict[str, Any] = {}
            self.errors: list[str] = []
            self._call_counter = itertools.count(1)
            self._object_counter = itertools.count(1)
            self._component_counter = itertools.count(1)
            self._root_calls: list[str] = []

    def enable(
        self,
        *,
        reset: bool = True,
        auto_emit_sidecar: bool = True,
        capture_polygon_events: bool = False,
        capture_port_events: bool = False,
        capture_live_reference_events: bool = False,
        max_ports_per_component_record: int = 256,
    ) -> None:
        self.install_component_hooks()
        with self._lock:
            if reset:
                self.reset()
            self.enabled = True
            self.auto_emit_sidecar = auto_emit_sidecar
            self.capture_polygon_events = capture_polygon_events
            self.capture_port_events = capture_port_events
            self.capture_live_reference_events = capture_live_reference_events
            self.max_ports_per_component_record = max(1, int(max_ports_per_component_record))

    def disable(self) -> None:
        with self._lock:
            self.enabled = False

    def install_component_hooks(self) -> None:
        if self._installed or Component is None:
            return

        runtime = self

        def _guarded(recorder: Callable[[], None]) -> None:
            try:
                recorder()
            except Exception as exc:  # pragma: no cover - no gdsfactory locally
                runtime.errors.append(str(exc))

        original_add_polygon = Component.add_polygon
        original_add_port = Component.add_port
        original_add_ports = getattr(Component, "add_ports", None)
        original_add_ref = getattr(Component, "add_ref", None)
        original_add = getattr(Component, "add", None)
        original_write_gds = getattr(Component, "write_gds", None)

        @functools.wraps(original_add_polygon)
        def add_polygon_wrapper(component: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_add_polygon(component, *args, **kwargs)
            if runtime.enabled and runtime.capture_polygon_events:
                _guarded(lambda: runtime._record_polygon(component, args, kwargs))
            return result

        @functools.wraps(original_add_port)
        def add_port_wrapper(component: Any, *args: Any, **kwargs: Any) -> Any:
            before = set(getattr(component, "ports", {}).keys()) if runtime.enabled and runtime.capture_port_events else None
            result = original_add_port(component, *args, **kwargs)
            if before is not None:
                _guarded(lambda: runtime._record_new_ports(component, before))
            return result

        if original_add_ports is not None:
            @functools.wraps(original_add_ports)
            def add_ports_wrapper(component: Any, *args: Any, **kwargs: Any) -> Any:
                before = set(getattr(component, "ports", {}).keys()) if runtime.enabled and runtime.capture_port_events else None
                result = original_add_ports(component, *args, **kwargs)
                if before is not None:
                    _guarded(lambda: runtime._record_new_ports(component, before))
                return result

            Component.add_ports = add_ports_wrapper

        if original_add_ref is not None:
            @functools.wraps(original_add_ref)
            def add_ref_wrapper(component: Any, *args: Any, **kwargs: Any) -> Any:
                result = original_add_ref(component, *args, **kwargs)
                if runtime.enabled and runtime.capture_live_reference_events:
                    _guarded(lambda: runtime._tag_reference(component, result))
                return result

            Component.add_ref = add_ref_wrapper

        if original_add is not None:
            @functools.wraps(original_add)
            def add_wrapper(component: Any, *args: Any, **kwargs: Any) -> Any:
                result = original_add(component, *args, **kwargs)
                if runtime.enabled and runtime.capture_live_reference_events:
                    _guarded(lambda: runtime._tag_added_items(component, args))
                return result

            Component.add = add_wrapper

        if original_write_gds is not None:
            @functools.wraps(original_write_gds)
            def write_gds_wrapper(component: Any, *args: Any, **kwargs: Any) -> Any:
                path = original_write_gds(component, *args, **kwargs)
                if runtime.enabled and runtime.auto_emit_sidecar:
                    _guarded(lambda: runtime.emit_sidecar(component, path))
                return path

            Component.write_gds = write_gds_wrapper

        Component.add_polygon = add_polygon_wrapper
        Component.add_port = add_port_wrapper
        self._installed = True

    def _next_call_id(self) -> str:
        return f"call_{next(self._call_counter):06d}"

    def _next_object_id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._object_counter):06d}"

    def _component_uid(self, component: Any) -> str:
        info = getattr(component, "info", None)
        if info is None:
            info = {}
            setattr(component, "info", info)
        uid = info.get("_smgr_component_uid")
        if uid is None:
            uid = f"comp_{next(self._component_counter):06d}"
            info["_smgr_component_uid"] = uid
        return uid

    def _current_call_id(self) -> Optional[str]:
        stack = _CALL_STACK.get()
        return stack[-1] if stack else None

    def _current_ancestor_ids(self) -> list[str]:
        return list(_CALL_STACK.get())

    def _ensure_source_hash(self, file_path: Optional[str]) -> None:
        if not file_path or file_path in self.source_hashes:
            return
        digest = _hash_file(file_path)
        if digest:
            self.source_hashes[file_path] = digest

    def _capture_callsite(self) -> dict[str, Any]:
        current_file = Path(__file__).resolve()
        for frame_info in inspect.stack()[2:]:
            try:
                frame_file = Path(frame_info.filename).resolve()
            except Exception:
                frame_file = Path(frame_info.filename)
            if frame_file == current_file:
                continue
            return {
                "file": str(frame_file),
                "line": frame_info.lineno,
                "function": frame_info.function,
            }
        return {"file": None, "line": None, "function": None}

    def _serialize_params(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Optional[Any]]:
        signature = inspect.signature(_unwrap_callable(func))
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        params: dict[str, Any] = {}
        pdk = None
        for name, value in bound.arguments.items():
            if pdk is None and hasattr(value, "name") and hasattr(value, "glayers"):
                pdk = value
                continue
            params[name] = _serialize_value(value)
        return params, pdk

    def _serialize_pdk(self, pdk: Any) -> dict[str, Any]:
        if pdk is None:
            return {}
        payload = {
            "name": getattr(pdk, "name", None),
            "glayers": _serialize_value(getattr(pdk, "glayers", {})),
            "models": _serialize_value(getattr(pdk, "models", {})),
            "pdk_files": _serialize_value(getattr(pdk, "pdk_files", {})),
        }
        payload["fingerprint"] = _hash_payload(payload)[:16]
        return payload

    def start_call(
        self,
        func: Callable[..., Any],
        generator_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        params, pdk = self._serialize_params(func, args, kwargs)
        parent_call_id = self._current_call_id()
        call_id = self._next_call_id()
        callsite = self._capture_callsite()
        source_func = _unwrap_callable(func)
        definition_file = inspect.getsourcefile(source_func)
        try:
            definition_line = inspect.getsourcelines(source_func)[1]
        except Exception:
            definition_line = None
        call_record = {
            "call_id": call_id,
            "parent_call_id": parent_call_id,
            "children_call_ids": [],
            "generator_id": generator_id,
            "function_name": func.__name__,
            "qualname": getattr(func, "__qualname__", func.__name__),
            "module": getattr(func, "__module__", None),
            "callsite": callsite,
            "definition": {
                "file": definition_file,
                "line": definition_line,
            },
            "params": params,
            "pdk": self._serialize_pdk(pdk),
            "created_object_ids": [],
            "status": "running",
            "started_at": time.time(),
            "output_component_name": None,
            "output_component_uid": None,
            "output_bbox": None,
            "ports": [],
        }
        with self._lock:
            self.call_records[call_id] = call_record
            if parent_call_id:
                self.call_records[parent_call_id]["children_call_ids"].append(call_id)
            else:
                self._root_calls.append(call_id)
        self._ensure_source_hash(callsite.get("file"))
        self._ensure_source_hash(definition_file)
        return call_record

    def push_call(self, call_id: str) -> Any:
        stack = _CALL_STACK.get()
        return _CALL_STACK.set(stack + (call_id,))

    def pop_call(self, token: Any) -> None:
        _CALL_STACK.reset(token)

    def _record_object(self, record: dict[str, Any]) -> None:
        object_id = record["object_id"]
        self.object_records[object_id] = record
        call_id = record.get("generated_by", {}).get("call_id")
        if call_id and call_id in self.call_records:
            self.call_records[call_id]["created_object_ids"].append(object_id)

    def _record_polygon(self, component: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        call_id = self._current_call_id()
        if not call_id:
            return
        layer = kwargs.get("layer")
        points = args[0] if args else kwargs.get("points")
        bbox = None
        if hasattr(points, "points"):
            bbox = _bbox_from_points(getattr(points, "points"))
            if layer is None:
                layer = getattr(points, "layer", None)
        elif points is not None:
            bbox = _bbox_from_points(points)
        record = {
            "object_id": self._next_object_id("poly"),
            "object_type": "polygon",
            "component_uid": self._component_uid(component),
            "component_name": getattr(component, "name", None),
            "layer": _serialize_layer(layer),
            "bbox": bbox,
            "net": None,
            "cell_path": [getattr(component, "name", None)],
            "mutable": False,
            "edit_handles": [],
            "generated_by": self._generated_by(call_id),
            "generator_context": self._generator_context(call_id),
            "parent_calls": self._ancestor_generator_ids(call_id),
            "children": [],
        }
        self._record_object(record)

    def _record_new_ports(self, component: Any, before_keys: set[str]) -> None:
        call_id = self._current_call_id()
        if not call_id:
            return
        ports = getattr(component, "ports", {})
        for name in sorted(set(ports.keys()) - before_keys):
            port = ports[name]
            port_payload = _serialize_port(port)
            center = port_payload["center"]
            width = port_payload["width"] or 0.0
            bbox = None
            if center[0] is not None and center[1] is not None:
                half = width / 2.0
                bbox = [
                    round(center[0] - half, 6),
                    round(center[1] - half, 6),
                    round(center[0] + half, 6),
                    round(center[1] + half, 6),
                ]
            record = {
                "object_id": self._next_object_id("port"),
                "object_type": "port",
                "component_uid": self._component_uid(component),
                "component_name": getattr(component, "name", None),
                "layer": port_payload["layer"],
                "bbox": bbox,
                "net": name,
                "cell_path": [getattr(component, "name", None)],
                "mutable": True,
                "edit_handles": ["center", "width", "orientation"],
                "generated_by": self._generated_by(call_id),
                "generator_context": self._generator_context(call_id),
                "parent_calls": self._ancestor_generator_ids(call_id),
                "children": [],
                "port": port_payload,
            }
            self._record_object(record)

    def _tag_reference(self, component: Any, ref: Any) -> None:
        if ref is None:
            return
        if getattr(ref, "_smgr_object_id", None) is None:
            setattr(ref, "_smgr_object_id", self._next_object_id("ref"))
        setattr(ref, "_smgr_parent_component_uid", self._component_uid(component))
        target = getattr(ref, "parent", None) or getattr(ref, "ref_cell", None)
        if target is not None:
            setattr(ref, "_smgr_target_component_uid", self._component_uid(target))
            target_info = getattr(target, "info", {})
            latest_call_id = target_info.get("_smgr_latest_call_id")
            if latest_call_id:
                setattr(ref, "_smgr_target_call_id", latest_call_id)

    def _tag_added_items(self, component: Any, args: tuple[Any, ...]) -> None:
        items = args[0] if len(args) == 1 else args
        if not isinstance(items, (list, tuple)):
            items = [items]
        for item in items:
            if hasattr(item, "bbox") and hasattr(item, "ports"):
                self._tag_reference(component, item)

    def _generated_by(self, call_id: str) -> dict[str, Any]:
        call = self.call_records[call_id]
        return {
            "file": call["callsite"]["file"],
            "line": call["callsite"]["line"],
            "function": call["callsite"]["function"],
            "call_id": call_id,
        }

    def _generator_context(self, call_id: str) -> dict[str, Any]:
        call = self.call_records[call_id]
        return {
            "generator_id": call["generator_id"],
            "pdk": call.get("pdk", {}).get("name"),
            "pdk_fingerprint": call.get("pdk", {}).get("fingerprint"),
            "params": call["params"],
        }

    def _ancestor_generator_ids(self, call_id: str) -> list[str]:
        chain: list[str] = []
        current = self.call_records.get(call_id)
        while current and current.get("parent_call_id"):
            parent_id = current["parent_call_id"]
            parent = self.call_records.get(parent_id)
            if parent is None:
                break
            chain.append(parent["generator_id"])
            current = parent
        chain.reverse()
        return chain

    def _serialize_component_ports(self, component: Any) -> tuple[list[dict[str, Any]], int, bool]:
        parent = getattr(component, "parent", None) or getattr(component, "ref_cell", None)
        if parent is not None and not (Component is not None and isinstance(component, Component)):
            try:
                parent_port_names = sorted(getattr(parent, "ports", {}).keys(), key=_port_priority)
            except Exception:
                parent_port_names = []
            parent_total = len(parent_port_names)
            if parent_total > self.max_ports_per_component_record:
                return ([], parent_total, True)

        try:
            port_names = sorted(getattr(component, "ports", {}).keys(), key=_port_priority)
        except Exception:
            return ([], 0, False)
        total = len(port_names)
        truncated = total > self.max_ports_per_component_record
        selected_names = port_names[: self.max_ports_per_component_record]
        ports: list[dict[str, Any]] = []
        for port_name in selected_names:
            ports.append(_serialize_port(component.ports[port_name]))
        return (ports, total, truncated)

    def _record_output_component(
        self,
        call_id: str,
        generator_id: str,
        component: Any,
        params: dict[str, Any],
    ) -> None:
        bbox = _bbox_from_object(component)
        uid = self._component_uid(component)
        ports, port_count_total, ports_truncated = self._serialize_component_ports(component)
        netlist_summary = _summarize_component_netlist(component)
        component_info = getattr(component, "info", None)
        if component_info is None:
            component_info = {}
            setattr(component, "info", component_info)
        component_info["_smgr_latest_call_id"] = call_id
        if component_info.get("_smgr_root_call_id") is None:
            component_info["_smgr_root_call_id"] = call_id
        component_info.setdefault("_smgr_call_ids", []).append(call_id)

        self.call_records[call_id]["output_component_name"] = getattr(component, "name", None)
        self.call_records[call_id]["output_component_uid"] = uid
        self.call_records[call_id]["output_bbox"] = bbox
        self.call_records[call_id]["ports"] = ports
        self.call_records[call_id]["port_count_total"] = port_count_total
        self.call_records[call_id]["ports_truncated"] = ports_truncated
        if netlist_summary is not None:
            self.call_records[call_id]["netlist_summary"] = netlist_summary
        self.call_records[call_id]["component_summary"] = {
            "bbox": bbox,
            "size": _bbox_to_size(bbox),
            "port_count": port_count_total,
            "ports_truncated": ports_truncated,
            "reference_count": len(getattr(component, "references", [])) if hasattr(component, "references") else None,
        }

        output_object = {
            "object_id": self._next_object_id("comp"),
            "object_type": _infer_object_type(generator_id),
            "component_uid": uid,
            "component_name": getattr(component, "name", None),
            "layer": _infer_object_layer(generator_id, params),
            "bbox": bbox,
            "net": None,
            "cell_path": [getattr(component, "name", None)],
            "mutable": True,
            "edit_handles": _infer_edit_handles(generator_id, params),
            "generated_by": self._generated_by(call_id),
            "generator_context": self._generator_context(call_id),
            "parent_calls": self._ancestor_generator_ids(call_id),
            "children": [],
            "port_count": port_count_total,
            "ports_truncated": ports_truncated,
        }
        if netlist_summary is not None:
            output_object["netlist_summary"] = {
                "circuit_name": netlist_summary.get("circuit_name"),
                "nodes": netlist_summary.get("nodes"),
                "node_count": netlist_summary.get("node_count"),
                "instance_count": netlist_summary.get("instance_count"),
                "net_count": netlist_summary.get("net_count"),
            }
        self._record_object(output_object)

        references = getattr(component, "references", None)
        if references:
            for ref in references:
                object_id = getattr(ref, "_smgr_object_id", None) or self._next_object_id("ref")
                target = getattr(ref, "parent", None) or getattr(ref, "ref_cell", None)
                target_info = getattr(target, "info", {}) if target is not None else {}
                record = {
                    "object_id": object_id,
                    "object_type": "instance",
                    "component_uid": uid,
                    "component_name": getattr(component, "name", None),
                    "layer": _infer_object_layer(generator_id, params),
                    "bbox": _bbox_from_object(ref),
                    "net": None,
                    "cell_path": [
                        getattr(component, "name", None),
                        getattr(target, "name", None) if target is not None else None,
                    ],
                    "mutable": True,
                    "edit_handles": ["translate", "rotate", "mirror"],
                    "generated_by": self._generated_by(call_id),
                    "generator_context": self._generator_context(call_id),
                    "parent_calls": self._ancestor_generator_ids(call_id),
                    "children": [],
                    "target_component_uid": getattr(ref, "_smgr_target_component_uid", None) or target_info.get("_smgr_component_uid"),
                    "target_call_id": getattr(ref, "_smgr_target_call_id", None) or target_info.get("_smgr_latest_call_id"),
                }
                self.object_records[object_id] = record
                if object_id not in self.call_records[call_id]["created_object_ids"]:
                    self.call_records[call_id]["created_object_ids"].append(object_id)

    def finish_call_success(self, call_id: str, generator_id: str, result: Any) -> None:
        with self._lock:
            call = self.call_records[call_id]
            call["status"] = "completed"
            call["finished_at"] = time.time()
            call["duration_s"] = round(call["finished_at"] - call["started_at"], 6)
            call["return_summary"] = _serialize_value(result)
            output_component = _coerce_component_like(result)
            if output_component is not None:
                self._record_output_component(call_id, generator_id, output_component, call["params"])

    def finish_call_error(self, call_id: str, exc: BaseException) -> None:
        with self._lock:
            call = self.call_records[call_id]
            call["status"] = "error"
            call["error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            call["finished_at"] = time.time()
            call["duration_s"] = round(call["finished_at"] - call["started_at"], 6)

    def _call_subtree(self, root_call_id: str) -> set[str]:
        pending = [root_call_id]
        visited: set[str] = set()
        while pending:
            call_id = pending.pop()
            if call_id in visited:
                continue
            visited.add(call_id)
            pending.extend(self.call_records.get(call_id, {}).get("children_call_ids", []))
        return visited

    def snapshot_for_component(self, component: Any) -> dict[str, Any]:
        info = getattr(component, "info", {})
        root_call_id = info.get("_smgr_latest_call_id") or info.get("_smgr_root_call_id")
        if root_call_id is None:
            related_call_ids = set(self.call_records.keys())
        else:
            related_call_ids = self._call_subtree(root_call_id)
            related_call_ids.add(root_call_id)
        calls = {
            call_id: self.call_records[call_id]
            for call_id in sorted(related_call_ids)
            if call_id in self.call_records
        }
        objects = {
            object_id: payload
            for object_id, payload in self.object_records.items()
            if payload.get("generated_by", {}).get("call_id") in related_call_ids
        }
        pdk_entries = [call.get("pdk", {}) for call in calls.values() if call.get("pdk")]
        pdk_summary = pdk_entries[0] if pdk_entries else {}
        return {
            "format": _SOURCE_MAP_VERSION,
            "session_id": self.session_id,
            "calls": calls,
            "objects": objects,
            "artifacts": dict(self.artifacts),
            "pdk": pdk_summary,
            "source_hashes": dict(self.source_hashes),
            "runtime_errors": list(self.errors),
        }

    def snapshot(self) -> dict[str, Any]:
        pdk_entries = [call.get("pdk", {}) for call in self.call_records.values() if call.get("pdk")]
        pdk_summary = pdk_entries[0] if pdk_entries else {}
        return {
            "format": _SOURCE_MAP_VERSION,
            "session_id": self.session_id,
            "calls": dict(self.call_records),
            "objects": dict(self.object_records),
            "artifacts": dict(self.artifacts),
            "pdk": pdk_summary,
            "source_hashes": dict(self.source_hashes),
            "runtime_errors": list(self.errors),
        }

    def emit_sidecar(self, component: Any, gds_path: Any) -> Optional[str]:
        if gds_path is None:
            return None
        resolved = Path(gds_path).resolve()
        if resolved.suffix.lower() == ".gds":
            sidecar_path = resolved.with_suffix(".provenance.json")
        else:
            sidecar_path = resolved.parent / f"{resolved.name}.provenance.json"
        payload = self.snapshot_for_component(component)
        payload["artifacts"].update(
            {
                "top_component_name": getattr(component, "name", None),
                "top_component_uid": self._component_uid(component),
                "gds": str(resolved),
                "provenance_json": str(sidecar_path),
            }
        )
        self.artifacts = payload["artifacts"]
        sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return str(sidecar_path)


_RUNTIME = SourceMappedGeneratorRuntime()


def get_runtime() -> SourceMappedGeneratorRuntime:
    return _RUNTIME


def enable_source_mapping(
    *,
    reset: bool = True,
    auto_emit_sidecar: bool = True,
    capture_polygon_events: bool = False,
    capture_port_events: bool = False,
    capture_live_reference_events: bool = False,
    max_ports_per_component_record: int = 256,
) -> None:
    _RUNTIME.enable(
        reset=reset,
        auto_emit_sidecar=auto_emit_sidecar,
        capture_polygon_events=capture_polygon_events,
        capture_port_events=capture_port_events,
        capture_live_reference_events=capture_live_reference_events,
        max_ports_per_component_record=max_ports_per_component_record,
    )


def disable_source_mapping() -> None:
    _RUNTIME.disable()


def reset_source_mapping() -> None:
    _RUNTIME.reset()


def tracked_generator(generator_id: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tracked_id = generator_id or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            runtime = get_runtime()
            if not runtime.enabled:
                return func(*args, **kwargs)
            call_record = runtime.start_call(func, tracked_id, args, kwargs)
            token = runtime.push_call(call_record["call_id"])
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                runtime.finish_call_error(call_record["call_id"], exc)
                raise
            finally:
                runtime.pop_call(token)
            runtime.finish_call_success(call_record["call_id"], tracked_id, result)
            return result

        return wrapper

    return decorator


def load_provenance(path: str | os.PathLike[str]) -> ProvenanceSnapshot:
    return ProvenanceSnapshot.from_file(path)


def get_call(call_id: str, snapshot: Optional[ProvenanceSnapshot] = None) -> Optional[dict[str, Any]]:
    if snapshot is None:
        snapshot = ProvenanceSnapshot(get_runtime().snapshot())
    return snapshot.get_call(call_id)


def get_objects_by_call(
    call_id: str,
    snapshot: Optional[ProvenanceSnapshot] = None,
) -> list[dict[str, Any]]:
    if snapshot is None:
        snapshot = ProvenanceSnapshot(get_runtime().snapshot())
    return snapshot.get_objects_by_call(call_id)


def query_objects_by_bbox(
    bbox: Iterable[float],
    layer_hint: Optional[str] = None,
    snapshot: Optional[ProvenanceSnapshot] = None,
) -> list[dict[str, Any]]:
    if snapshot is None:
        snapshot = ProvenanceSnapshot(get_runtime().snapshot())
    return snapshot.query_objects_by_bbox(bbox, layer_hint=layer_hint)


def rank_candidate_calls(
    marker_bbox: Iterable[float],
    rule_name: Optional[str] = None,
    layer_hint: Optional[str] = None,
    snapshot: Optional[ProvenanceSnapshot] = None,
) -> list[dict[str, Any]]:
    if snapshot is None:
        snapshot = ProvenanceSnapshot(get_runtime().snapshot())
    return snapshot.rank_candidate_calls(marker_bbox, rule_name=rule_name, layer_hint=layer_hint)


def auto_enable_from_env() -> None:
    for key in _ENV_ENABLE_KEYS:
        if _normalize_bool_env(os.getenv(key)):
            enable_source_mapping(
                reset=True,
                auto_emit_sidecar=True,
                capture_polygon_events=_normalize_bool_env(os.getenv(_ENV_CAPTURE_POLYGONS)),
                capture_port_events=_normalize_bool_env(os.getenv(_ENV_CAPTURE_PORT_OBJECTS)),
                capture_live_reference_events=_normalize_bool_env(os.getenv(_ENV_CAPTURE_LIVE_REFS)),
            )
            break
