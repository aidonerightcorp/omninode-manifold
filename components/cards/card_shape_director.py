"""Director CARD parsing and executing arbitrary nested >> // | @ OmniNode shape expressions."""

from __future__ import annotations

from typing import Any

from omninode_manifold.shape_algebra import OmniNodeFailure

CARD = {
    "name": "card_shape_director",
    "role": "director",
    "summary": "Parses and evaluates recursive OmniNode shape algebra expressions (>> pipe, | parallel, // fallback, @ check).",
    "capabilities": ["director", "shape_algebra", "pipe_parallel_fallback_verify"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    expr = job.get("shape") or shell.cfg.get("shape")
    if not expr:
        raise OmniNodeFailure("card_shape_director requires a 'shape' expression in cfg or job", kind="bad_input")
    res = shell.execute_shape(str(expr), job)
    res.setdefault("quality_score", 1.0)
    return res
