"""Orient-stage spectral profiler CARD computing Participation Ratio, anisotropy, and RMT signal count."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.math_kernels import orient_spectral_profile

CARD = {
    "name": "card_orient_spectral_profiler",
    "role": "stage",
    "stage": "orient",
    "summary": "Computes Participation Ratio, top eigenvalue anisotropy, and Marchenko-Pastur RMT signal dimensions.",
    "capabilities": ["orient", "spectral_profile", "participation_ratio", "rmt_concentration"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    raw_x = job.get("activations")
    if raw_x is None:
        rng = np.random.default_rng(int(shell.registry.num["default_rng_seed"]))
        raw_x = rng.standard_normal(
            (int(shell.registry.num["sixteen"]), int(shell.registry.num["thirty_two"]))
        )
    arr_x = np.asarray(raw_x, dtype=np.float64)
    profile = orient_spectral_profile(arr_x, shell.registry.num)
    target_x = job.get("target_activations")
    hetero = target_x is not None and np.asarray(target_x).shape[-1] != arr_x.shape[-1]
    return {
        "activations": arr_x.tolist(),
        "spectral_profile": profile,
        "heterogeneous_dims": bool(hetero),
        "quality_score": 1.0,
    }
