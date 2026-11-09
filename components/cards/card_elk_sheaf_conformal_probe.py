"""ELK truth-vs-sycophancy probe, cellular sheaf coboundary, and ACI conformal prediction check CARD."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.math_kernels import evaluate_elk_sheaf_conformal

CARD = {
    "name": "card_elk_sheaf_conformal_probe",
    "role": "check",
    "summary": "Checks internal latent consistency via ELK contrast direction, sheaf coboundary energy, and ACI conformal coverage.",
    "capabilities": ["elk_probe", "sheaf_coboundary", "conformal_aci", "check_gate"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    cell = dict(job.get("selected_grid_cell") or shell.grid_cell)
    arr = np.asarray(job["activations"], dtype=np.float64)
    n_s = arr.shape[0]
    two = int(shell.registry.num["two"])

    true_y = np.asarray(
        job.get("true_labels", [idx % two for idx in range(n_s)]),
        dtype=np.float64,
    )
    syco_y = np.asarray(
        job.get("sycophantic_labels", [(idx + 1) % two for idx in range(n_s)]),
        dtype=np.float64,
    )
    raw_sibs = job.get("sibling_views")
    if raw_sibs is None:
        hundredth = float(shell.registry.num["hundredth"])
        sibs = [arr, arr * (1.0 - hundredth)]
    else:
        sibs = [np.asarray(s, dtype=np.float64) for s in raw_sibs]

    res = evaluate_elk_sheaf_conformal(arr, true_y, syco_y, sibs, cell, shell.registry.num)
    return {
        "check_passed": bool(res["check_passed"]),
        "check_metrics": res,
        "quality_score": float(res["conformal_empirical_coverage"]),
    }
