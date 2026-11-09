"""Woodbury ridge, (epsilon, delta)-DP RLS, Sinkhorn-Knopp balanced routing, and CAS lease_epoch solver CARD."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.math_kernels import solve_woodbury_sinkhorn_attributed

CARD = {
    "name": "card_woodbury_sinkhorn_attributed_solver",
    "role": "worker",
    "summary": "Solves primal/dual Woodbury ridge with differentially private RLS statistics, Sinkhorn-Knopp routing, and bitemporal lease_epoch reuse.",
    "capabilities": ["woodbury_solver", "dp_rls", "sinkhorn_knopp", "bitemporal_lease_cache"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    cell = dict(job.get("selected_grid_cell") or shell.grid_cell)
    arr_x = np.asarray(job["activations"], dtype=np.float64)
    raw_y = job.get("targets")
    if raw_y is None:
        out_cols = min(int(shell.registry.num["four"]), arr_x.shape[1])
        arr_y = arr_x[:, :out_cols]
    else:
        arr_y = np.asarray(raw_y, dtype=np.float64)
    lease_epoch = str(job.get("lease_epoch", "epoch_v1"))
    cache_bucket = shell.library.runtime_cache.setdefault("woodbury_lease_cache", {})

    res = solve_woodbury_sinkhorn_attributed(
        arr_x,
        arr_y,
        lease_epoch,
        cache_bucket,
        cell,
        shell.registry.num,
    )
    return {
        "activations": res["predictions"].tolist(),
        "fitted_weights": {"w1": res["weights"].tolist()},
        "solver_metrics": {
            "woodbury_r2": res["woodbury_r2"],
            "sinkhorn_row_marginal_error": res["sinkhorn_row_marginal_error"],
            "lease_epoch": res["lease_epoch"],
            "cache_hit": res["cache_hit"],
        },
        "quality_score": float(res["woodbury_r2"]),
    }
