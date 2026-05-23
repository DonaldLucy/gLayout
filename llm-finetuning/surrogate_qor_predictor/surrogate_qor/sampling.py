from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from .registry import GeneratorSpec, ParamSpec, stable_sample_id


def _lhs_unit(n: int, k: int, rng: random.Random) -> list[list[float]]:
    columns: list[list[float]] = []
    for _ in range(k):
        values = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(values)
        columns.append(values)
    return [[columns[j][i] for j in range(k)] for i in range(n)]


def _min_pairwise_distance(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    best = float("inf")
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
            best = min(best, dist)
    return best


def maximin_lhs(n: int, k: int, seed: int, candidates: int = 16) -> list[list[float]]:
    if n <= 0 or k <= 0:
        return []
    best_points: list[list[float]] = []
    best_score = -1.0
    for offset in range(max(1, candidates)):
        rng = random.Random(seed + offset * 1009)
        points = _lhs_unit(n, k, rng)
        score = _min_pairwise_distance(points)
        if score > best_score:
            best_score = score
            best_points = points
    return best_points


def _coerce_value(spec: ParamSpec, unit: float) -> Any:
    if spec.kind == "float":
        assert spec.low is not None and spec.high is not None
        value = spec.low + unit * (spec.high - spec.low)
        return round(float(value), 6)
    if spec.kind == "int":
        assert spec.low is not None and spec.high is not None
        value = int(round(spec.low + unit * (spec.high - spec.low)))
        return value
    if spec.kind == "categorical":
        if not spec.choices:
            return spec.default
        index = min(len(spec.choices) - 1, int(unit * len(spec.choices)))
        return spec.choices[index]
    if spec.kind == "bool":
        return bool(unit >= 0.5)
    return spec.default


def default_params(spec: GeneratorSpec) -> dict[str, Any]:
    return {param.name: param.default for param in spec.params}


def sample_params(spec: GeneratorSpec, count: int, seed: int) -> list[dict[str, Any]]:
    if not spec.params:
        return [{}]
    count = max(1, count)
    rows: list[dict[str, Any]] = [default_params(spec)]
    if count == 1:
        return rows
    points = maximin_lhs(count - 1, len(spec.params), seed=seed)
    seen = {json.dumps(rows[0], sort_keys=True, default=str)}
    for point in points:
        row = {param.name: _coerce_value(param, point[index]) for index, param in enumerate(spec.params)}
        key = json.dumps(row, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def generate_sample_plan(
    specs: list[GeneratorSpec],
    samples_per_parameterized_cell: int,
    seed: int,
    include_fixed: bool = True,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        if spec.fixed and not include_fixed:
            continue
        count = 1 if spec.fixed or not spec.params else samples_per_parameterized_cell
        for ordinal, params in enumerate(sample_params(spec, count=count, seed=seed + spec_index * 7919)):
            sample_id = stable_sample_id(spec.generator_id, params, ordinal)
            plan.append(
                {
                    "sample_id": sample_id,
                    "ordinal": ordinal,
                    "generator_id": spec.generator_id,
                    "corpus": spec.corpus,
                    "family": spec.family,
                    "source_ref": spec.source_ref,
                    "description": spec.description,
                    "fixed": spec.fixed,
                    "cost": spec.cost,
                    "params": params,
                    "param_count": len(spec.params),
                    "code_features": spec.code_features,
                }
            )
    return plan


def shard_plan(plan: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return plan
    return [row for index, row in enumerate(plan) if index % num_shards == shard_index]


def write_plan(path: Path, plan: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in plan:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

