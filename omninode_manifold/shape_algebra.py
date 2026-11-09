"""Recursive parser and evaluator for OmniNode shape expressions (>> pipe, | fan-out, // fallback, @ check)."""

from __future__ import annotations

import copy
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np


class OmniNodeFailure(RuntimeError):
    """Typed failure envelope raised by OmniNode components and shells."""

    def __init__(self, message: str, *, kind: str = "capability", details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.kind = str(kind)
        self.details = dict(details) if details else {}


class OmniNodeContractPermissionError(PermissionError, OmniNodeFailure):
    """Raised when maker_id == checker_id on an @ verification handoff (both PermissionError and contract failure)."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        PermissionError.__init__(self, message)
        OmniNodeFailure.__init__(self, message, kind="contract", details=details)


SHAPE_TOKEN_PATTERN = re.compile(r"\s*(>>|//|\||@|\(|\)|[A-Za-z_][A-Za-z0-9_.\-]*)")


@dataclass(frozen=True)
class ShapeLeaf:
    name: str


@dataclass(frozen=True)
class ShapePipe:
    left: ShapeExpr
    right: ShapeExpr


@dataclass(frozen=True)
class ShapeParallel:
    branches: tuple[ShapeExpr, ...]


@dataclass(frozen=True)
class ShapeFallback:
    primary: ShapeExpr
    backup: ShapeExpr


@dataclass(frozen=True)
class ShapeVerify:
    body: ShapeExpr
    checker: ShapeExpr


ShapeExpr = ShapeLeaf | ShapePipe | ShapeParallel | ShapeFallback | ShapeVerify


def tokenize_shape(expr_text: str) -> list[str]:
    pos = 0
    tokens: list[str] = []
    text = expr_text.strip()
    while pos < len(text):
        match = SHAPE_TOKEN_PATTERN.match(text, pos)
        if not match:
            raise ValueError(f"Unexpected token in shape expression at offset {pos}: {text[pos:]!r}")
        tokens.append(match.group(1))
        pos = match.end()
    return tokens


class _ShapeParser:
    """Recursive-descent parser with precedence: >> (pipe) > | (parallel) > // (fallback) > @ (verify)."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.idx = 0

    def peek(self) -> str | None:
        return self.tokens[self.idx] if self.idx < len(self.tokens) else None

    def consume(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of shape expression")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r} in shape expression, got {tok!r}")
        self.idx += 1
        return tok

    def parse(self) -> ShapeExpr:
        node = self.parse_verify()
        if self.peek() is not None:
            raise ValueError(f"Trailing token in shape expression: {self.peek()!r}")
        return node

    def parse_verify(self) -> ShapeExpr:
        node = self.parse_fallback()
        while self.peek() == "@":
            self.consume("@")
            checker = self.parse_fallback()
            node = ShapeVerify(body=node, checker=checker)
        return node

    def parse_fallback(self) -> ShapeExpr:
        node = self.parse_parallel()
        while self.peek() == "//":
            self.consume("//")
            backup = self.parse_parallel()
            node = ShapeFallback(primary=node, backup=backup)
        return node

    def parse_parallel(self) -> ShapeExpr:
        first = self.parse_pipe()
        branches = [first]
        while self.peek() == "|":
            self.consume("|")
            branches.append(self.parse_pipe())
        if len(branches) == 1:
            return first
        return ShapeParallel(branches=tuple(branches))

    def parse_pipe(self) -> ShapeExpr:
        node = self.parse_atom()
        while self.peek() == ">>":
            self.consume(">>")
            rhs = self.parse_atom()
            node = ShapePipe(left=node, right=rhs)
        return node

    def parse_atom(self) -> ShapeExpr:
        tok = self.peek()
        if tok is None:
            raise ValueError("Empty atom in shape expression")
        if tok == "(":
            self.consume("(")
            inner = self.parse_verify()
            self.consume(")")
            return inner
        if tok in (">>", "//", "|", "@", ")"):
            raise ValueError(f"Unexpected operator {tok!r} where node identifier expected")
        return ShapeLeaf(name=self.consume())


def parse_shape_expression(expr_text: str) -> ShapeExpr:
    """Parse algebraic OmniNode shape expression into an immutable AST."""
    tokens = tokenize_shape(expr_text)
    if not tokens:
        raise ValueError("Shape expression must not be empty")
    return _ShapeParser(tokens).parse()


def _fuse_parallel_outputs(
    branch_outputs: list[dict[str, Any]],
    *,
    consensus_mode: str,
    eps: float,
    ridge_lambda: float,
) -> dict[str, Any]:
    """Fuse sibling OmniNode vector states via sheaf inverse-variance, ridge, or uniform consensus."""
    arrays: list[np.ndarray] = []
    for out in branch_outputs:
        candidate = out.get("activations")
        if candidate is None:
            candidate = out.get("predictions")
        if candidate is not None:
            arr = np.asarray(candidate, dtype=np.float64)
            arrays.append(arr)

    merged: dict[str, Any] = {}
    for out in branch_outputs:
        for k, v in out.items():
            if k not in ("activations", "predictions", "shape_trace"):
                merged[k] = copy.deepcopy(v)

    if not arrays:
        merged["parallel_branch_count"] = len(branch_outputs)
        merged["sheaf_disagreement_energy"] = 0.0
        merged["consensus_weights"] = [1.0 / float(len(branch_outputs)) for _ in branch_outputs]
        return merged

    min_cols = min(arr.shape[-1] for arr in arrays)
    aligned = [arr[..., :min_cols] for arr in arrays]
    n_b = len(aligned)
    mean_stack = np.mean(np.stack(aligned, axis=0), axis=0)

    if consensus_mode == "uniform_mean":
        weights = np.full(n_b, 1.0 / float(n_b), dtype=np.float64)
    elif consensus_mode == "ridge_weighted":
        norms = np.array([float(np.mean(a * a)) + ridge_lambda for a in aligned], dtype=np.float64)
        inv = 1.0 / (norms + eps)
        weights = inv / float(np.sum(inv))
    else:
        disagreements = np.array(
            [float(np.mean((a - mean_stack) * (a - mean_stack))) + eps for a in aligned],
            dtype=np.float64,
        )
        inv_var = 1.0 / disagreements
        weights = inv_var / float(np.sum(inv_var))

    fused = np.zeros_like(aligned[0], dtype=np.float64)
    for idx, arr in enumerate(aligned):
        fused = fused + weights[idx] * arr

    sheaf_energy = 0.0
    pair_count = 0
    normed = [
        arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + eps)
        for arr in aligned
    ]
    for i in range(n_b):
        for j in range(i + 1, n_b):
            diff = normed[i] - normed[j]
            sheaf_energy += float(np.mean(diff * diff))
            pair_count += 1
    if pair_count > 0:
        sheaf_energy /= float(pair_count)

    merged["activations"] = fused.tolist()
    merged["parallel_branch_count"] = n_b
    merged["consensus_mode"] = consensus_mode
    merged["consensus_weights"] = [float(w) for w in weights]
    merged["sheaf_disagreement_energy"] = float(sheaf_energy)
    return merged


def run_shape_expression(
    expr: ShapeExpr | str,
    job: dict[str, Any],
    *,
    resolve_child: Callable[[str], Any],
    grid_cell: dict[str, Any],
    rerank_kinds: frozenset[str],
    halt_kinds: frozenset[str],
    eps: float,
) -> dict[str, Any]:
    """Evaluate a parsed ShapeExpr across child OmniNode shells."""
    ast_node = parse_shape_expression(expr) if isinstance(expr, str) else expr

    if isinstance(ast_node, ShapeLeaf):
        child = resolve_child(ast_node.name)
        res = child.handle(copy.deepcopy(job))
        out = dict(res) if isinstance(res, dict) else {"value": res}
        actor = str(out.get("actor_id", getattr(child, "actor_id", ast_node.name)))
        out["actor_id"] = actor
        trace = list(out.get("shape_trace", []))
        trace.append({"op": "leaf", "node": ast_node.name, "actor_id": actor})
        out["shape_trace"] = trace
        return out

    if isinstance(ast_node, ShapePipe):
        left_out = run_shape_expression(
            ast_node.left,
            job,
            resolve_child=resolve_child,
            grid_cell=grid_cell,
            rerank_kinds=rerank_kinds,
            halt_kinds=halt_kinds,
            eps=eps,
        )
        next_job = copy.deepcopy(job)
        for k, v in left_out.items():
            if k != "shape_trace":
                next_job[k] = copy.deepcopy(v)
        right_out = run_shape_expression(
            ast_node.right,
            next_job,
            resolve_child=resolve_child,
            grid_cell=grid_cell,
            rerank_kinds=rerank_kinds,
            halt_kinds=halt_kinds,
            eps=eps,
        )
        combined_trace = list(left_out.get("shape_trace", [])) + list(right_out.get("shape_trace", []))
        merged_pipe = dict(left_out)
        merged_pipe.update(right_out)
        merged_pipe["shape_trace"] = combined_trace
        return merged_pipe

    if isinstance(ast_node, ShapeFallback):
        try:
            return run_shape_expression(
                ast_node.primary,
                job,
                resolve_child=resolve_child,
                grid_cell=grid_cell,
                rerank_kinds=rerank_kinds,
                halt_kinds=halt_kinds,
                eps=eps,
            )
        except OmniNodeFailure as err:
            if err.kind in halt_kinds or err.kind not in rerank_kinds:
                raise
            backup_job = copy.deepcopy(job)
            fallbacks = list(backup_job.get("fallback_events", []))
            fallbacks.append({"failed_kind": err.kind, "reason": str(err)})
            backup_job["fallback_events"] = fallbacks
            backup_out = run_shape_expression(
                ast_node.backup,
                backup_job,
                resolve_child=resolve_child,
                grid_cell=grid_cell,
                rerank_kinds=rerank_kinds,
                halt_kinds=halt_kinds,
                eps=eps,
            )
            backup_out["fallback_events"] = fallbacks
            trace = list(backup_out.get("shape_trace", []))
            trace.append({"op": "fallback_recovered", "failed_kind": err.kind})
            backup_out["shape_trace"] = trace
            return backup_out

    if isinstance(ast_node, ShapeParallel):
        branch_results: list[dict[str, Any]] = []
        combined_trace = []
        for branch in ast_node.branches:
            b_res = run_shape_expression(
                branch,
                job,
                resolve_child=resolve_child,
                grid_cell=grid_cell,
                rerank_kinds=rerank_kinds,
                halt_kinds=halt_kinds,
                eps=eps,
            )
            branch_results.append(b_res)
            combined_trace.extend(b_res.get("shape_trace", []))
        consensus_mode = str(grid_cell.get("shell_parallel_consensus_mode", "sheaf_inverse_variance"))
        ridge_lam = float(grid_cell.get("woodbury_ridge_lambda", 1.0))
        fused = _fuse_parallel_outputs(
            branch_results,
            consensus_mode=consensus_mode,
            eps=eps,
            ridge_lambda=ridge_lam,
        )
        combined_trace.append({"op": "parallel_fuse", "branches": len(branch_results), "mode": consensus_mode})
        fused["shape_trace"] = combined_trace
        return fused

    if isinstance(ast_node, ShapeVerify):
        body_out = run_shape_expression(
            ast_node.body,
            job,
            resolve_child=resolve_child,
            grid_cell=grid_cell,
            rerank_kinds=rerank_kinds,
            halt_kinds=halt_kinds,
            eps=eps,
        )
        maker_id = str(body_out.get("actor_id", job.get("maker_id", "body_maker")))
        if isinstance(ast_node.checker, ShapeLeaf):
            checker_shell = resolve_child(ast_node.checker.name)
            pre_checker_id = str(getattr(checker_shell, "actor_id", ast_node.checker.name))
            if pre_checker_id == maker_id:
                raise OmniNodeContractPermissionError(
                    f"Independent check handoff violated: maker_id ({maker_id}) == checker_id ({pre_checker_id})",
                    details={"maker_id": maker_id, "checker_id": pre_checker_id},
                )

        check_job = copy.deepcopy(job)
        check_job["candidate_output"] = copy.deepcopy(body_out)
        check_job["maker_id"] = maker_id
        for k, v in body_out.items():
            if k not in ("shape_trace", "actor_id"):
                check_job[k] = copy.deepcopy(v)

        check_out = run_shape_expression(
            ast_node.checker,
            check_job,
            resolve_child=resolve_child,
            grid_cell=grid_cell,
            rerank_kinds=rerank_kinds,
            halt_kinds=halt_kinds,
            eps=eps,
        )
        checker_id = str(check_out.get("actor_id", "independent_checker"))
        if checker_id == maker_id:
            raise OmniNodeContractPermissionError(
                f"Independent check handoff violated: maker_id ({maker_id}) == checker_id ({checker_id})",
                details={"maker_id": maker_id, "checker_id": checker_id},
            )
        if not bool(check_out.get("check_passed", True)):
            raise OmniNodeFailure(
                f"Check node rejected candidate output: {check_out.get('check_reason', 'failed')}",
                kind="contract",
                details={"maker_id": maker_id, "checker_id": checker_id, "check_output": check_out},
            )
        merged_verify = dict(body_out)
        merged_verify["verification"] = {
            "passed": True,
            "maker_id": maker_id,
            "checker_id": checker_id,
            "metrics": check_out.get("check_metrics", {}),
        }
        trace = list(body_out.get("shape_trace", [])) + list(check_out.get("shape_trace", []))
        trace.append({"op": "verify_handoff", "maker_id": maker_id, "checker_id": checker_id})
        merged_verify["shape_trace"] = trace
        return merged_verify

    raise ValueError(f"Unhandled ShapeExpr node: {ast_node!r}")
