"""Input-conditioned Kronecker nullspace HyperLoRA and Green's-function LiquidStateSpace worker CARD."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.math_kernels import apply_hyperlora_liquid_operator

CARD = {
    "name": "card_hyperlora_liquid_operator",
    "role": "worker",
    "summary": "Applies input-conditioned Kronecker nullspace HyperLoRA weight modulation followed by causal Green's-function LiquidStateSpace integration.",
    "capabilities": ["hyperlora", "kronecker_nullspace", "liquid_state_space", "sequence_worker"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    cell = dict(job.get("selected_grid_cell") or shell.grid_cell)
    seq = np.asarray(job["activations"], dtype=np.float64)
    ctx_raw = job.get("context_vector")
    ctx = np.asarray(ctx_raw, dtype=np.float64) if ctx_raw is not None else np.mean(seq, axis=0)
    anc_raw = job.get("anchor_subspace")
    anchors = np.asarray(anc_raw, dtype=np.float64) if anc_raw is not None else None

    res = apply_hyperlora_liquid_operator(seq, ctx, anchors, cell, shell.registry.num)
    return {
        "activations": res["activations"].tolist(),
        "hyperlora_metrics": {
            "kronecker_factors": res["kronecker_factors"],
            "kronecker_rank": res["kronecker_rank"],
            "anchor_nullspace_drift": res["anchor_nullspace_drift"],
        },
        "quality_score": 1.0,
    }
