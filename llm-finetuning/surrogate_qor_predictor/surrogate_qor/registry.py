from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


Builder = Callable[[Any, dict[str, Any]], Any]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    default: Any
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] = ()
    target: str | None = None
    index: int | None = None

    def with_prefix(self, prefix: str, index: int | None = None) -> "ParamSpec":
        name = f"{prefix}__{index}" if index is not None else prefix
        return ParamSpec(
            name=name,
            kind=self.kind,
            default=self.default,
            low=self.low,
            high=self.high,
            choices=self.choices,
            target=prefix,
            index=index,
        )


@dataclass(frozen=True)
class GeneratorSpec:
    generator_id: str
    corpus: str
    family: str
    builder: Builder
    params: tuple[ParamSpec, ...] = ()
    description: str = ""
    source_ref: str = ""
    fixed: bool = False
    cost: str = "medium"
    code_features: dict[str, float] = field(default_factory=dict)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def add_repo_import_paths() -> None:
    root = repo_root()
    for path in (
        root / "src",
        root / "tests",
        root / "llm-finetuning" / "verified_convo_samples",
        root / "llm-finetuning" / "surrogate_qor_predictor",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def stable_sample_id(generator_id: str, params: dict[str, Any], ordinal: int) -> str:
    payload = json.dumps([generator_id, ordinal, params], sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def reconstruct_kwargs(param_specs: tuple[ParamSpec, ...], sampled: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    tuple_parts: dict[str, dict[int, Any]] = {}
    for spec in param_specs:
        value = sampled.get(spec.name, spec.default)
        target = spec.target or spec.name
        if spec.index is None:
            kwargs[target] = value
        else:
            tuple_parts.setdefault(target, {})[spec.index] = value
    for target, parts in tuple_parts.items():
        kwargs[target] = tuple(parts[index] for index in sorted(parts))
    return kwargs


def _numeric_range(name: str, default: float) -> tuple[float, float]:
    lname = name.lower()
    if "length" in lname:
        lo = 0.15 if default in (0, None) else max(0.15, float(default) * 0.5)
        hi = max(lo + 0.1, min(5.0, float(default or 0.5) * 3.0))
    elif "via_x" in lname or "track" in lname or "separation" in lname:
        lo, hi = -5.0, 5.0
    elif "size" in lname or "cap" in lname:
        lo = max(0.5, float(default or 1.0) * 0.5)
        hi = max(lo + 0.5, min(25.0, float(default or 1.0) * 4.0))
    else:
        lo = max(0.5, float(default or 1.0) * 0.5)
        hi = max(lo + 0.5, min(20.0, float(default or 1.0) * 4.0))
    return lo, hi


def _int_choices(name: str, default: int) -> tuple[int, ...]:
    lname = name.lower()
    if "finger" in lname or "col" in lname or "row" in lname:
        base = [1, 2, 3, 4, 6, 8, 10, 12]
    elif "mult" in lname or "rmult" in lname:
        base = [1, 2, 3, 4, 5, 6]
    else:
        base = [max(0, default - 1), default, default + 1, default + 2]
    return tuple(sorted({int(x) for x in base if int(x) >= 0}))


def _param_from_default(name: str, default: Any) -> list[ParamSpec]:
    if isinstance(default, bool):
        return [ParamSpec(name=name, kind="categorical", default=default, choices=(False, True))]
    if isinstance(default, int) and not isinstance(default, bool):
        return [ParamSpec(name=name, kind="categorical", default=default, choices=_int_choices(name, default))]
    if isinstance(default, float):
        lo, hi = _numeric_range(name, default)
        return [ParamSpec(name=name, kind="float", default=default, low=lo, high=hi)]
    if isinstance(default, str):
        return [ParamSpec(name=name, kind="categorical", default=default, choices=(default,))]
    if isinstance(default, tuple) and default and all(isinstance(v, (int, float, bool)) or v is None for v in default):
        specs: list[ParamSpec] = []
        for index, value in enumerate(default):
            scalar = 0.5 if value is None else value
            for child in _param_from_default(f"{name}_{index}", scalar):
                specs.append(child.with_prefix(name, index))
        return specs
    return []


def _auto_param_specs(func: Callable[..., Any]) -> tuple[ParamSpec, ...]:
    specs: list[ParamSpec] = []
    for name, param in inspect.signature(func).parameters.items():
        if name in {"pdk", "kwargs"} or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        if param.default is inspect.Parameter.empty:
            continue
        specs.extend(_param_from_default(name, param.default))
    return tuple(specs)


def _code_features(func: Callable[..., Any]) -> dict[str, float]:
    try:
        source = inspect.getsource(func)
    except Exception:
        source = ""
    return {
        "code_line_count": float(source.count("\n") + 1 if source else 0),
        "code_route_mentions": float(source.count("route") + source.count("RouteSpec")),
        "code_device_mentions": float(source.count("DeviceSpec") + source.count("nmos") + source.count("pmos")),
        "code_move_mentions": float(source.count("MoveSpec") + source.count("move")),
        "code_label_mentions": float(source.count("label") + source.count("add_label")),
    }


def _wrap_signature_builder(func: Callable[..., Any], param_specs: tuple[ParamSpec, ...]) -> Builder:
    def _builder(pdk: Any, sampled: dict[str, Any]) -> Any:
        return func(pdk, **reconstruct_kwargs(param_specs, sampled))

    return _builder


def _manual_float(name: str, default: float, low: float, high: float) -> ParamSpec:
    return ParamSpec(name=name, kind="float", default=default, low=low, high=high)


def _manual_cat(name: str, default: Any, choices: list[Any] | tuple[Any, ...]) -> ParamSpec:
    return ParamSpec(name=name, kind="categorical", default=default, choices=tuple(choices))


def _load_openfasoc_specs() -> list[GeneratorSpec]:
    add_repo_import_paths()
    generated = importlib.import_module("convo_layouts.generated")
    manifest_path = repo_root() / "llm-finetuning" / "verified_convo_samples" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    specs: list[GeneratorSpec] = []
    for prompt_name, meta in sorted(manifest["generated"].items()):
        module_name, func_name = meta["builder"].rsplit(".", 1)
        if module_name != "convo_layouts.generated":
            continue
        func = getattr(generated, func_name)
        params = _auto_param_specs(func)
        specs.append(
            GeneratorSpec(
                generator_id=f"openfasoc.{prompt_name}",
                corpus="openfasoc_convo",
                family=prompt_name,
                builder=_wrap_signature_builder(func, params),
                params=params,
                description=meta.get("repair_note", ""),
                source_ref=f"{module_name}.{func_name}",
                fixed=not bool(params),
                cost="low" if len(params) <= 4 else "medium",
                code_features=_code_features(func),
            )
        )
    return specs


def _load_glayout_specs() -> list[GeneratorSpec]:
    add_repo_import_paths()
    from smgr_cases import SMGR_CASES

    specs: list[GeneratorSpec] = []

    def fixed_case_builder(case_id: str) -> Builder:
        case_map = {case.case_id: case for case in SMGR_CASES}

        def _builder(_pdk: Any, _sampled: dict[str, Any]) -> Any:
            return case_map[case_id].builder()

        return _builder

    manual: dict[str, tuple[Builder, tuple[ParamSpec, ...], dict[str, float], str]] = {}

    from glayout.cells.elementary.diff_pair import diff_pair, diff_pair_generic
    from glayout.cells.elementary.diff_pair.diff_pair import add_df_labels
    from glayout.cells.elementary.current_mirror import current_mirror
    from glayout.cells.elementary.transmission_gate import transmission_gate
    from glayout.cells.elementary.FVF import flipped_voltage_follower
    from glayout.cells.composite.differential_to_single_ended_converter import differential_to_single_ended_converter

    diff_params = (
        _manual_float("width", 3.0, 1.0, 10.0),
        _manual_float("length", 0.5, 0.15, 3.0),
        _manual_cat("fingers", 4, [2, 3, 4, 6, 8, 10]),
        _manual_cat("rmult", 1, [1, 2, 3, 4]),
        _manual_float("plus_minus_seperation", 0.0, 0.0, 4.0),
    )

    def _diff_pair(n_or_p: bool) -> Builder:
        def _builder(pdk: Any, sampled: dict[str, Any]) -> Any:
            return add_df_labels(diff_pair(pdk, n_or_p_fet=n_or_p, dummy=True, substrate_tap=True, **sampled), pdk)

        return _builder

    manual["diff_pair_default"] = (_diff_pair(True), diff_params, _code_features(diff_pair), "parameterized")
    manual["diff_pair_pmos"] = (_diff_pair(False), diff_params, _code_features(diff_pair), "parameterized")

    generic_params = diff_params + (_manual_cat("n_or_p_fet", True, [True, False]),)

    def _diff_pair_generic(pdk: Any, sampled: dict[str, Any]) -> Any:
        return diff_pair_generic(pdk, dummy=True, substrate_tap=True, **sampled)

    manual["diff_pair_generic"] = (_diff_pair_generic, generic_params, _code_features(diff_pair_generic), "parameterized")

    cm_params = (
        _manual_cat("numcols", 2, [2, 3, 4, 5, 6]),
        _manual_float("width", 3.0, 1.0, 12.0),
        _manual_float("length", 0.5, 0.15, 3.0),
        _manual_cat("fingers", 2, [1, 2, 3, 4, 6, 8]),
        _manual_cat("with_dummy", True, [True, False]),
    )

    def _cm(device: str) -> Builder:
        def _builder(pdk: Any, sampled: dict[str, Any]) -> Any:
            return current_mirror(pdk, device=device, with_labels=True, with_substrate_tap=False, with_tie=True, **sampled)

        return _builder

    manual["current_mirror_nfet"] = (_cm("nfet"), cm_params, _code_features(current_mirror), "parameterized")
    manual["current_mirror_pfet"] = (_cm("pfet"), cm_params, _code_features(current_mirror), "parameterized")

    tg_params = (
        ParamSpec("width__0", "float", 2.0, 1.0, 8.0, target="width", index=0),
        ParamSpec("width__1", "float", 2.0, 1.0, 8.0, target="width", index=1),
        ParamSpec("length__0", "float", 0.5, 0.15, 2.0, target="length", index=0),
        ParamSpec("length__1", "float", 0.5, 0.15, 2.0, target="length", index=1),
        ParamSpec("fingers__0", "categorical", 1, choices=(1, 2, 3, 4, 6), target="fingers", index=0),
        ParamSpec("fingers__1", "categorical", 1, choices=(1, 2, 3, 4, 6), target="fingers", index=1),
        ParamSpec("multipliers__0", "categorical", 1, choices=(1, 2, 3, 4), target="multipliers", index=0),
        ParamSpec("multipliers__1", "categorical", 1, choices=(1, 2, 3, 4), target="multipliers", index=1),
    )

    def _tg(pdk: Any, sampled: dict[str, Any]) -> Any:
        return transmission_gate(pdk, substrate_tap=False, with_labels=True, **reconstruct_kwargs(tg_params, sampled))

    manual["transmission_gate"] = (_tg, tg_params, _code_features(transmission_gate), "parameterized")

    fvf_params = (
        ParamSpec("width__0", "float", 4.15, 1.0, 10.0, target="width", index=0),
        ParamSpec("width__1", "float", 4.15, 1.0, 10.0, target="width", index=1),
        ParamSpec("length__0", "float", 2.0, 0.3, 4.0, target="length", index=0),
        ParamSpec("length__1", "float", 2.0, 0.3, 4.0, target="length", index=1),
        ParamSpec("fingers__0", "categorical", 2, choices=(1, 2, 3, 4, 6), target="fingers", index=0),
        ParamSpec("fingers__1", "categorical", 2, choices=(1, 2, 3, 4, 6), target="fingers", index=1),
        ParamSpec("multipliers__0", "categorical", 1, choices=(1, 2, 3), target="multipliers", index=0),
        ParamSpec("multipliers__1", "categorical", 1, choices=(1, 2, 3), target="multipliers", index=1),
        _manual_cat("sd_rmult", 1, [1, 2, 3]),
    )

    def _fvf(pdk: Any, sampled: dict[str, Any]) -> Any:
        return flipped_voltage_follower(pdk, with_dnwell=False, with_labels=True, **reconstruct_kwargs(fvf_params, sampled))

    manual["flipped_voltage_follower"] = (_fvf, fvf_params, _code_features(flipped_voltage_follower), "parameterized")

    d2s_params = (
        _manual_cat("rmult", 2, [1, 2, 3, 4]),
        ParamSpec("half_pload__0", "float", 6.0, 2.0, 12.0, target="half_pload", index=0),
        ParamSpec("half_pload__1", "float", 1.0, 0.3, 4.0, target="half_pload", index=1),
        ParamSpec("half_pload__2", "categorical", 4, choices=(2, 3, 4, 6, 8), target="half_pload", index=2),
        _manual_float("via_xlocation", 0.0, -4.0, 4.0),
    )

    def _d2s(pdk: Any, sampled: dict[str, Any]) -> Any:
        return differential_to_single_ended_converter(pdk, **reconstruct_kwargs(d2s_params, sampled))

    manual["differential_to_single_ended_converter"] = (
        _d2s,
        d2s_params,
        _code_features(differential_to_single_ended_converter),
        "parameterized",
    )

    for case in SMGR_CASES:
        if case.case_id in manual:
            builder, params, code_features, cost = manual[case.case_id]
            specs.append(
                GeneratorSpec(
                    generator_id=f"glayout.{case.case_id}",
                    corpus="glayout_smgr",
                    family=case.case_id,
                    builder=builder,
                    params=params,
                    description=case.description,
                    source_ref=f"tests.smgr_cases.{case.builder.__name__}",
                    fixed=False,
                    cost=cost,
                    code_features=code_features,
                )
            )
        else:
            specs.append(
                GeneratorSpec(
                    generator_id=f"glayout.{case.case_id}",
                    corpus="glayout_smgr",
                    family=case.case_id,
                    builder=fixed_case_builder(case.case_id),
                    params=(),
                    description=case.description,
                    source_ref=f"tests.smgr_cases.{case.builder.__name__}",
                    fixed=True,
                    cost="high",
                    code_features=_code_features(case.builder),
                )
            )
    return specs


def load_generator_specs(selected: set[str] | None = None) -> list[GeneratorSpec]:
    specs = _load_openfasoc_specs() + _load_glayout_specs()
    if selected:
        specs = [spec for spec in specs if spec.generator_id in selected or spec.family in selected]
    return specs


def specs_as_manifest(specs: list[GeneratorSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            {
                "generator_id": spec.generator_id,
                "corpus": spec.corpus,
                "family": spec.family,
                "description": spec.description,
                "source_ref": spec.source_ref,
                "fixed": spec.fixed,
                "cost": spec.cost,
                "params": [param.__dict__ for param in spec.params],
                "code_features": spec.code_features,
            }
        )
    return rows

