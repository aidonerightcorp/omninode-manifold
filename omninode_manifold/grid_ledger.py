"""Declarative NodeGrid, 7-stage GridLedger, Boundary Auto-Expansion, grid/1 Recorder, and FactsLedger."""

from __future__ import annotations

import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def canonical_json(obj: Any) -> str:
    """Deterministic sorted-key compact JSON representation."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_digest_hex(payload: str | bytes) -> str:
    """Compute hex content digest over UTF-8 text or raw bytes."""
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(raw).hexdigest()


def utc_now_iso() -> str:
    """UTC ISO-8601 timestamp string."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_fsync_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL record under exclusive file lock with flush and os.fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class ParameterBounds:
    """Clamping bounds and auto-expansion increment for a declarative NodeParameter axis."""

    min_value: float
    max_value: float
    expansion_step: float

    def clamp(self, value: int | float) -> int | float:
        if value < self.min_value:
            return type(value)(self.min_value)
        if value > self.max_value:
            return type(value)(self.max_value)
        return value


@dataclass(frozen=True)
class NodeParameter:
    """Single tunable parameter axis inside NodeGrid."""

    axis_id: str
    kind: str
    bounds: ParameterBounds
    default: Any
    choices: tuple[Any, ...] = ()
    low: int | None = None
    high: int | None = None
    step: int = 1

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> NodeParameter:
        b_spec = spec["bounds"]
        bounds = ParameterBounds(
            min_value=float(b_spec["min_value"]),
            max_value=float(b_spec["max_value"]),
            expansion_step=float(b_spec["expansion_step"]),
        )
        kind = str(spec["kind"])
        if kind == "choice":
            return cls(
                axis_id=str(spec["axis_id"]),
                kind=kind,
                bounds=bounds,
                default=spec["default"],
                choices=tuple(spec["choices"]),
            )
        if kind == "integer_range":
            return cls(
                axis_id=str(spec["axis_id"]),
                kind=kind,
                bounds=bounds,
                default=int(spec["default"]),
                low=int(spec["low"]),
                high=int(spec["high"]),
                step=int(spec["step"]),
            )
        if kind == "boolean":
            return cls(
                axis_id=str(spec["axis_id"]),
                kind=kind,
                bounds=bounds,
                default=bool(spec["default"]),
            )
        raise ValueError(f"Unsupported NodeParameter kind: {kind}")

    def validate_and_clamp(self, raw_value: Any) -> Any:
        if self.kind == "boolean":
            if not isinstance(raw_value, bool):
                raise ValueError(f"Axis {self.axis_id} requires bool")
            return raw_value
        if self.kind == "choice":
            if self.choices and isinstance(self.choices[0], (int, float)) and not isinstance(raw_value, bool):
                if isinstance(raw_value, (int, float)):
                    clamped = self.bounds.clamp(raw_value)
                    if clamped in self.choices:
                        return clamped
            if raw_value not in self.choices:
                raise ValueError(f"Value {raw_value!r} invalid for choice axis {self.axis_id}")
            return raw_value
        if self.kind == "integer_range":
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise ValueError(f"Axis {self.axis_id} requires integer")
            clamped_i = int(self.bounds.clamp(raw_value))
            assert self.low is not None and self.high is not None
            if clamped_i < self.low or clamped_i > self.high:
                raise ValueError(f"Clamped value {clamped_i} outside [{self.low}, {self.high}]")
            return clamped_i
        raise ValueError(f"Unknown parameter kind {self.kind}")

    def is_upper_saturated(self, value: Any) -> bool:
        if self.bounds.expansion_step <= 0.0:
            return False
        if self.kind == "integer_range" and self.high is not None:
            return int(value) >= self.high
        if self.kind == "choice" and self.choices and isinstance(self.choices[-1], (int, float)):
            return float(value) >= float(self.choices[-1])
        return False

    def expand_upper_boundary(self) -> NodeParameter:
        step_f = float(self.bounds.expansion_step)
        if step_f <= 0.0:
            return self
        new_bounds = ParameterBounds(
            min_value=self.bounds.min_value,
            max_value=self.bounds.max_value + step_f,
            expansion_step=self.bounds.expansion_step,
        )
        if self.kind == "integer_range" and self.high is not None:
            step_i = int(step_f)
            return NodeParameter(
                axis_id=self.axis_id,
                kind=self.kind,
                bounds=new_bounds,
                default=self.default,
                low=self.low,
                high=self.high + step_i,
                step=self.step,
            )
        if self.kind == "choice" and self.choices and isinstance(self.choices[-1], (int, float)):
            last_val = self.choices[-1]
            new_top: int | float = int(last_val) + int(step_f) if isinstance(last_val, int) else float(last_val) + step_f
            return NodeParameter(
                axis_id=self.axis_id,
                kind=self.kind,
                bounds=new_bounds,
                default=self.default,
                choices=(*self.choices, new_top),
            )
        return self


@dataclass
class NodeGrid:
    """Multi-dimensional configuration settings grid with cross-axis constraints and boundary auto-expansion."""

    grid_id: str
    generation: int
    parameters: dict[str, NodeParameter]
    constraints: list[dict[str, Any]]

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> NodeGrid:
        params = {p["axis_id"]: NodeParameter.from_spec(p) for p in spec["axes"]}
        return cls(
            grid_id=str(spec["grid_id"]),
            generation=int(spec["generation"]),
            parameters=params,
            constraints=list(spec.get("constraints", [])),
        )

    def default_cell(self) -> dict[str, Any]:
        return {axis_id: param.default for axis_id, param in self.parameters.items()}

    def normalize_and_partition(
        self, raw_config: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        """Normalize known parameters, returning (normalized_cell, unknown_keys, invalid_keys)."""
        normalized = self.default_cell()
        unknown: list[str] = []
        invalid: list[str] = []
        for key in sorted(raw_config.keys()):
            if key not in self.parameters:
                unknown.append(key)
                continue
            param = self.parameters[key]
            try:
                normalized[key] = param.validate_and_clamp(raw_config[key])
            except ValueError:
                invalid.append(key)

        for rule in self.constraints:
            r_kind = str(rule["kind"])
            left_k = str(rule["left_axis"])
            right_k = str(rule["right_axis"])
            if r_kind == "less_or_equal":
                if float(normalized[left_k]) > float(normalized[right_k]):
                    invalid.append(left_k)
                    normalized[left_k] = self.parameters[left_k].default
            elif r_kind == "less_or_equal_scaled":
                scale = float(rule["right_scale"])
                if float(normalized[left_k]) > float(normalized[right_k]) * scale:
                    invalid.append(left_k)
                    normalized[left_k] = self.parameters[left_k].default
        return normalized, sorted(set(unknown)), sorted(set(invalid))

    def cell_digest(self, normalized_cell: dict[str, Any]) -> str:
        return content_digest_hex(canonical_json(normalized_cell))

    def saturated_axes(self, normalized_cell: dict[str, Any]) -> list[str]:
        return [
            axis_id
            for axis_id, param in self.parameters.items()
            if param.is_upper_saturated(normalized_cell.get(axis_id, param.default))
        ]

    def auto_expand_boundaries(self, saturated_axis_ids: list[str]) -> NodeGrid:
        new_params: dict[str, NodeParameter] = {}
        expanded_any = False
        target_set = set(saturated_axis_ids)
        for axis_id, param in self.parameters.items():
            if axis_id in target_set:
                expanded = param.expand_upper_boundary()
                if expanded != param:
                    expanded_any = True
                new_params[axis_id] = expanded
            else:
                new_params[axis_id] = param
        next_gen = self.generation + (1 if expanded_any else 0)
        return NodeGrid(
            grid_id=self.grid_id,
            generation=next_gen,
            parameters=new_params,
            constraints=copy.deepcopy(self.constraints),
        )


@dataclass
class OperationCostLedger:
    """Empirical cost and quality ledger backing contextual regime and CARD search routing."""

    observations: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        regime_id: str,
        component_name: str,
        cell_digest: str,
        passed: bool,
        quality_score: float,
        latency_ms: float,
    ) -> None:
        self.observations.append(
            {
                "regime_id": regime_id,
                "component_name": component_name,
                "cell_digest": cell_digest,
                "passed": bool(passed),
                "quality_score": float(quality_score),
                "latency_ms": float(latency_ms),
            }
        )

    def component_summary(self, component_name: str) -> dict[str, float]:
        matching = [row for row in self.observations if row["component_name"] == component_name]
        if not matching:
            return {"count": 0.0, "pass_rate": 0.0, "mean_score": 0.0, "mean_latency_ms": 0.0}
        cnt = float(len(matching))
        passes = sum(1.0 for row in matching if row["passed"])
        tot_score = sum(float(row["quality_score"]) for row in matching)
        tot_lat = sum(float(row["latency_ms"]) for row in matching)
        return {
            "count": cnt,
            "pass_rate": passes / cnt,
            "mean_score": tot_score / cnt,
            "mean_latency_ms": tot_lat / cnt,
        }


@dataclass
class GridLedger:
    """7-stage configuration cell lifecycle state machine enforcing disjoint maker_id != checker_id."""

    stages: tuple[str, ...]
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)

    def register_cell(self, cell_id: str, regime_id: str, values: dict[str, Any], maker_id: str) -> None:
        self.cells[cell_id] = {
            "cell_id": cell_id,
            "regime_id": regime_id,
            "values": copy.deepcopy(values),
            "maker_id": maker_id,
            "checker_id": None,
            "stage_index": 0,
            "stage": self.stages[0],
            "history": [self.stages[0]],
            "metrics": {},
        }

    def advance(
        self,
        cell_id: str,
        target_stage: str,
        *,
        actor_id: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if cell_id not in self.cells:
            raise KeyError(f"Unknown cell_id {cell_id}")
        rec = self.cells[cell_id]
        expected_next = int(rec["stage_index"]) + 1
        if expected_next >= len(self.stages) or self.stages[expected_next] != target_stage:
            raise ValueError(f"Illegal GridLedger transition {rec['stage']} -> {target_stage}")
        checker_start = len(self.stages) - 1 - 1
        if expected_next >= checker_start:
            if actor_id == rec["maker_id"]:
                raise PermissionError("Disjoint maker_id != checker_id required for cell promotion")
            rec["checker_id"] = actor_id
        rec["stage_index"] = expected_next
        rec["stage"] = target_stage
        rec["history"].append(target_stage)
        if metrics:
            rec["metrics"].update(metrics)

    def promote_full_lifecycle(
        self,
        cell_id: str,
        regime_id: str,
        values: dict[str, Any],
        *,
        maker_id: str,
        checker_id: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        self.register_cell(cell_id, regime_id, values, maker_id=maker_id)
        promote_idx = len(self.stages) - 1
        gate_idx = promote_idx - 1
        checker_stages = {self.stages[gate_idx], self.stages[promote_idx]}
        for stg in self.stages[1:]:
            actor = checker_id if stg in checker_stages else maker_id
            stg_metrics = metrics if stg == self.stages[gate_idx] else None
            self.advance(cell_id, stg, actor_id=actor, metrics=stg_metrics)
        return copy.deepcopy(self.cells[cell_id])


class Grid1Recorder:
    """Append-only fsynced grid/1 cell outcome recorder writing grid-records.jsonl."""

    def __init__(self, path: Path, format_version: str) -> None:
        self.path = path
        self.format_version = format_version

    def record_run(
        self,
        *,
        run_id: str,
        kind: str,
        source: str,
        grid: NodeGrid,
        raw_config: dict[str, Any],
        passed: bool,
        score: float | None,
        cost: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        failure: str | None = None,
    ) -> dict[str, Any]:
        normalized, unknown, invalid = grid.normalize_and_partition(raw_config)
        cell = grid.cell_digest(normalized)
        row = {
            "format": self.format_version,
            "run_id": str(run_id),
            "kind": str(kind),
            "source": str(source),
            "at": utc_now_iso(),
            "cell": cell,
            "config": normalized,
            "unknown": unknown,
            "invalid": invalid,
            "outcome": {
                "passed": bool(passed),
                "score": float(score) if score is not None else None,
                "cost": float(cost),
                "tokens_in": int(tokens_in),
                "tokens_out": int(tokens_out),
                "latency_ms": float(latency_ms),
                "failure": str(failure) if failure is not None else None,
            },
        }
        append_fsync_jsonl(self.path, row)
        return row

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


class FactsLedger:
    """Append-only fsynced OmniNode observation ledger writing .omni/facts.jsonl."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def emit(self, kind: str, **payload: Any) -> dict[str, Any]:
        event = {"at": utc_now_iso(), "kind": str(kind), **payload}
        append_fsync_jsonl(self.path, event)
        return event

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


class OmniNodeGridRegistry:
    """Central registry holding declarative NodeGrid, promoted regime cells, and protocol/numerical specs."""

    def __init__(self, config_path: Path | None = None) -> None:
        default_path = Path(__file__).resolve().parent.parent / "config" / "omninode_universal_grid.json"
        self.config_path = config_path or default_path
        self.raw_spec: dict[str, Any] = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.grid = NodeGrid.from_spec(self.raw_spec)
        self.ledger = GridLedger(stages=tuple(self.raw_spec["lifecycle_stages"]))
        self.cost_ledger = OperationCostLedger()
        self.protocol = dict(self.raw_spec["protocol_constants"])
        self.num = dict(self.raw_spec["numerical_constants"])
        self.local_runner = dict(self.raw_spec["local_runner_config"])
        self.compiler_grid = dict(self.raw_spec["nine_runtime_compiler_grid"])
        self.thresholds = dict(self.raw_spec["test_thresholds"])
        self.failure_kinds: tuple[str, ...] = tuple(self.raw_spec["failure_kinds"])
        self.rerank_failure_kinds: frozenset[str] = frozenset(self.raw_spec["rerank_failure_kinds"])
        self.halt_failure_kinds: frozenset[str] = frozenset(self.raw_spec["halt_failure_kinds"])
        self.regime_cells: dict[str, dict[str, Any]] = {}
        self._bootstrap_regimes()

    def _bootstrap_regimes(self) -> None:
        regimes = self.raw_spec.get("regimes", {})
        for regime_id, r_spec in regimes.items():
            normalized, _, invalid = self.grid.normalize_and_partition(dict(r_spec["values"]))
            if invalid:
                raise ValueError(f"Invalid bootstrap regime {regime_id}: {invalid}")
            cell_id = f"gen{self.grid.generation}_{regime_id}"
            self.ledger.promote_full_lifecycle(
                cell_id=cell_id,
                regime_id=regime_id,
                values=normalized,
                maker_id="grid_architect_maker",
                checker_id="independent_quorum_checker",
                metrics={"bootstrap_valid": 1.0},
            )
            self.regime_cells[regime_id] = normalized

    def resolve_regime_cell(
        self,
        *,
        participation_ratio: float | None = None,
        anisotropy_ratio: float | None = None,
        heterogeneous_dims: bool = False,
        high_capacity: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        base = self.grid.default_cell()
        pr_low = float(base["orient_participation_low_threshold"])
        aniso_high = float(base["orient_anisotropy_high_threshold"])
        if high_capacity:
            regime_id = "saturated_high_capacity_manifold"
        elif heterogeneous_dims:
            regime_id = "cross_space_heterogeneous_bridge"
        elif anisotropy_ratio is not None and anisotropy_ratio >= aniso_high:
            regime_id = "anisotropic_concept_erasure"
        elif participation_ratio is not None and participation_ratio <= pr_low:
            regime_id = "low_rank_fast_stream"
        else:
            regime_id = "anisotropic_concept_erasure"

        merged = dict(self.regime_cells[regime_id])
        if overrides:
            merged.update(overrides)
        normalized, _, _ = self.grid.normalize_and_partition(merged)
        return regime_id, normalized, self.grid.cell_digest(normalized)

    def evaluate_and_auto_expand(
        self,
        regime_id: str,
        *,
        observed_score: float,
        maker_id: str = "explorer_maker",
        checker_id: str = "independent_quorum_checker",
    ) -> dict[str, Any]:
        cell_values = dict(self.regime_cells[regime_id])
        saturated = self.grid.saturated_axes(cell_values)
        prev_gen = self.grid.generation
        if not saturated:
            return {
                "expanded": False,
                "previous_generation": prev_gen,
                "new_generation": prev_gen,
                "saturated_axes": [],
            }
        self.grid = self.grid.auto_expand_boundaries(saturated)
        upgraded = dict(cell_values)
        for axis_id in saturated:
            param = self.grid.parameters[axis_id]
            if param.kind == "integer_range" and param.high is not None:
                upgraded[axis_id] = param.high
            elif param.kind == "choice" and param.choices:
                upgraded[axis_id] = param.choices[-1]
        normalized, _, _ = self.grid.normalize_and_partition(upgraded)
        new_cell_id = f"gen{self.grid.generation}_{regime_id}_expanded"
        self.ledger.promote_full_lifecycle(
            cell_id=new_cell_id,
            regime_id=regime_id,
            values=normalized,
            maker_id=maker_id,
            checker_id=checker_id,
            metrics={"observed_score": float(observed_score)},
        )
        self.regime_cells[regime_id] = normalized
        return {
            "expanded": self.grid.generation > prev_gen,
            "previous_generation": prev_gen,
            "new_generation": self.grid.generation,
            "saturated_axes": saturated,
            "promoted_cell_id": new_cell_id,
            "promoted_digest": self.grid.cell_digest(normalized),
        }
