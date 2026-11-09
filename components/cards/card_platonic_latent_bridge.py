"""Zero-retraining Orthogonal Procrustes and Ridge cross-space latent bridge CARD."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.math_kernels import apply_platonic_latent_bridge

CARD = {
    "name": "card_platonic_latent_bridge",
    "role": "transform",
    "summary": "Aligns heterogeneous embedding dimensions via Economy-SVD subspace, Orthogonal Procrustes, and Ridge refinement.",
    "capabilities": ["platonic_bridge", "procrustes_alignment", "cross_space_transfer"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    cell = dict(job.get("selected_grid_cell") or shell.grid_cell)
    x_src = np.asarray(job["activations"], dtype=np.float64)
    raw_tgt = job.get("target_activations")
    if raw_tgt is None:
        x_tgt = x_src
    else:
        x_tgt = np.asarray(raw_tgt, dtype=np.float64)

    default_split = int(shell.registry.local_runner["train_split"])
    train_split = int(job.get("train_split", min(default_split, max(int(shell.registry.num["two"]), x_src.shape[0] - int(shell.registry.num["two"])))))
    res = apply_platonic_latent_bridge(x_src, x_tgt, train_split, cell, shell.registry.num)
    return {
        "activations": res["mapped_activations"].tolist(),
        "bridge_metrics": {
            "held_out_top1_retrieval": res["held_out_top1_retrieval"],
            "held_out_cosine_alignment": res["held_out_cosine_alignment"],
            "bridge_rank": res["bridge_rank"],
            "source_dim": res["source_dim"],
            "target_dim": res["target_dim"],
        },
        "quality_score": float(res["held_out_top1_retrieval"]),
    }
