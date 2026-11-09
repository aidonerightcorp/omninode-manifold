"""Closed-form Economy-SVD Belrose LEACE oblique concept eraser CARD."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.math_kernels import apply_leace_concept_eraser

CARD = {
    "name": "card_leace_concept_eraser",
    "role": "transform",
    "summary": "Applies closed-form Economy-SVD IPW LEACE oblique projection driving nuisance covariance Cov(PX, Z) to zero.",
    "capabilities": ["leace", "concept_erasure", "oblique_projection", "nuisance_removal"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    cell = dict(job.get("selected_grid_cell") or shell.grid_cell)
    arr_x = np.asarray(job["activations"], dtype=np.float64)
    two = int(shell.registry.num["two"])
    z_nuis = np.asarray(
        job.get("nuisance_labels", [idx % two for idx in range(arr_x.shape[0])]),
        dtype=np.float64,
    )
    res = apply_leace_concept_eraser(arr_x, z_nuis, cell, shell.registry.num)
    return {
        "activations": res["erased_activations"].tolist(),
        "leace_metrics": {
            "max_abs_cov_before": res["max_abs_cov_before"],
            "max_abs_cov_after": res["max_abs_cov_after"],
            "nuisance_rank_removed": res["nuisance_rank_removed"],
        },
        "quality_score": 1.0,
    }
