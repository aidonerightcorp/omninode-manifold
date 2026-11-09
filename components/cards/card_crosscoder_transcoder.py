"""Multi-layer SparseCrosscoder and causal MlpTranscoder circuit CARD."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.math_kernels import fit_transform_crosscoder_transcoder

CARD = {
    "name": "card_crosscoder_transcoder",
    "role": "transform",
    "summary": "Learns shared multi-layer SparseCrosscoder atoms and causal MlpTranscoder pre-to-post MLP readouts.",
    "capabilities": ["crosscoder", "mlp_transcoder", "circuit_discovery"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    cell = dict(job.get("selected_grid_cell") or shell.grid_cell)
    raw_views = job.get("layer_views")
    if raw_views is None:
        base_x = np.asarray(job["activations"], dtype=np.float64)
        half = float(shell.registry.num["half"])
        raw_views = [base_x, np.tanh(base_x) * half + base_x * half]
    views = [np.asarray(v, dtype=np.float64) for v in raw_views]
    post_target = np.asarray(job.get("post_mlp_target", np.maximum(views[0], 0.0)), dtype=np.float64)
    res = fit_transform_crosscoder_transcoder(views, post_target, cell, shell.registry.num)
    return {
        "activations": res["transcoder_predictions"].tolist(),
        "crosscoder_metrics": {
            "crosscoder_r2": res["crosscoder_r2"],
            "transcoder_r2": res["transcoder_r2"],
            "shared_atom_count": res["shared_atom_count"],
        },
        "quality_score": float(res["crosscoder_r2"]),
    }
