"""Frenet-Serret curvature, Participation Ratio, and Koopman eigenvalue trajectory loop-breaker CARD."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.math_kernels import analyze_trajectory_loop

CARD = {
    "name": "card_trajectory_loop_breaker",
    "role": "check",
    "summary": "Monitors state trajectories using Frenet-Serret curvature, Participation Ratio, and Koopman spectral modulus to halt repetitive loops early.",
    "capabilities": ["trajectory_loop_breaker", "frenet_serret", "koopman_halt", "check_gate"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    cell = dict(job.get("selected_grid_cell") or shell.grid_cell)
    raw_traj = job.get("trajectory", job.get("activations"))
    traj = np.asarray(raw_traj, dtype=np.float64)
    res = analyze_trajectory_loop(traj, cell, shell.registry.num)
    allow_loop = bool(job.get("halt_is_success", True))
    passed = True if allow_loop else (not bool(res["loop_detected"]))
    return {
        "check_passed": passed,
        "check_metrics": res,
        "trajectory_metrics": res,
        "quality_score": 1.0,
    }
