"""Independent maker != checker quorum verification gate CARD."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.shape_algebra import OmniNodeContractPermissionError

CARD = {
    "name": "card_independent_quorum_gate",
    "role": "check",
    "summary": "Enforces disjoint maker_id != checker_id and checks finite non-degenerate output invariants.",
    "capabilities": ["independent_quorum_gate", "maker_checker_separation", "check_gate"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    maker_id = str(job.get("maker_id", "unknown_maker"))
    checker_id = str(shell.actor_id)
    if maker_id == checker_id:
        raise OmniNodeContractPermissionError(
            f"Quorum gate rejected self-check: maker_id ({maker_id}) == checker_id ({checker_id})",
            details={"maker_id": maker_id, "checker_id": checker_id},
        )

    eps = float(shell.registry.num["eps"])
    act_raw = job.get("activations")
    finite_ok = True
    energy_ok = True
    norm_val = 1.0
    if act_raw is not None:
        arr = np.asarray(act_raw, dtype=np.float64)
        finite_ok = bool(np.all(np.isfinite(arr)))
        norm_val = float(np.linalg.norm(arr))
        energy_ok = norm_val > eps

    sheaf_energy = float(job.get("sheaf_disagreement_energy", 0.0))
    sheaf_ceiling = float(shell.grid_cell["elk_sheaf_energy_ceiling"])
    sheaf_ok = sheaf_energy <= sheaf_ceiling

    passed = finite_ok and energy_ok and sheaf_ok
    return {
        "actor_id": checker_id,
        "check_passed": bool(passed),
        "check_reason": "ok" if passed else "invariant_violation",
        "check_metrics": {
            "maker_id": maker_id,
            "checker_id": checker_id,
            "finite_ok": finite_ok,
            "norm": norm_val,
            "sheaf_disagreement_energy": sheaf_energy,
            "sheaf_ok": sheaf_ok,
        },
        "quality_score": 1.0 if passed else 0.0,
    }
