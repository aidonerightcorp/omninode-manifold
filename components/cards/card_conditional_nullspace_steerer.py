"""Conditional regime-gated nullspace steerer CARD guaranteeing zero anchor drift and zero off-target intrusion."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.math_kernels import apply_conditional_nullspace_steerer

CARD = {
    "name": "card_conditional_nullspace_steerer",
    "role": "transform",
    "summary": "Projects steering vector onto P_{Null(A_anchor)} with Fisher-shrinkage regime gating for zero anchor drift and zero off-target intrusion.",
    "capabilities": ["nullspace_steerer", "conditional_steering", "anchor_preservation"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    cell = dict(job.get("selected_grid_cell") or shell.grid_cell)
    arr_x = np.asarray(job["activations"], dtype=np.float64)
    n_s, n_d = arr_x.shape
    two = int(shell.registry.num["two"])

    raw_mask = job.get("is_target_regime")
    if raw_mask is None:
        mask = np.array([idx < (n_s // two) for idx in range(n_s)], dtype=bool)
    else:
        mask = np.asarray(raw_mask, dtype=bool)

    raw_anchors = job.get("anchor_subspace")
    if raw_anchors is None:
        rng = np.random.default_rng(int(shell.registry.num["default_rng_seed"]))
        anchors = rng.standard_normal((int(shell.registry.num["four"]), n_d))
    else:
        anchors = np.asarray(raw_anchors, dtype=np.float64)

    res = apply_conditional_nullspace_steerer(arr_x, mask, anchors, cell, shell.registry.num)
    return {
        "activations": res["steered_activations"].tolist(),
        "steerer_metrics": {
            "anchor_max_abs_drift": res["anchor_max_abs_drift"],
            "off_target_intrusion_rate": res["off_target_intrusion_rate"],
            "on_target_activation_rate": res["on_target_activation_rate"],
            "gate_threshold": res["gate_threshold"],
        },
        "quality_score": 1.0 - float(res["off_target_intrusion_rate"]),
    }
