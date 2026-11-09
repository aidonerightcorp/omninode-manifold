"""Compose-stage contextual NodeGrid selector CARD clamping ParameterBounds and resolving regime cells."""

from __future__ import annotations

from typing import Any

CARD = {
    "name": "card_compose_grid_selector",
    "role": "stage",
    "stage": "compose",
    "summary": "Selects contextual promoted NodeGrid cell from spectral signature and clamps ParameterBounds.",
    "capabilities": ["compose", "grid_selector", "regime_router", "parameter_bounds"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    prof = dict(job.get("spectral_profile") or {})
    pr_val = prof.get("participation_ratio")
    aniso_val = prof.get("anisotropy_ratio")
    hetero = bool(job.get("heterogeneous_dims", False))
    high_cap = bool(job.get("high_capacity", False))
    overrides = dict(job.get("grid_overrides") or {})

    regime_id, cell_vals, cell_digest = shell.registry.resolve_regime_cell(
        participation_ratio=float(pr_val) if pr_val is not None else None,
        anisotropy_ratio=float(aniso_val) if aniso_val is not None else None,
        heterogeneous_dims=hetero,
        high_capacity=high_cap,
        overrides=overrides,
    )
    expansion = {}
    if bool(job.get("auto_expand_if_saturated", False)):
        expansion = shell.registry.evaluate_and_auto_expand(
            regime_id,
            observed_score=float(job.get("observed_score", 1.0)),
        )

    out: dict[str, Any] = {
        "selected_regime_id": regime_id,
        "selected_grid_cell": cell_vals,
        "selected_cell_digest": cell_digest,
        "boundary_expansion": expansion,
        "quality_score": 1.0,
    }
    if "activations" in job:
        out["activations"] = job["activations"]
    return out
