"""Parameterized Sparse Autoencoder CARD supporting TopK, BatchTopK, JumpReLU, GatedSAE, and MatryoshkaNested."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.math_kernels import fit_transform_sae

CARD = {
    "name": "card_sae_dictionary",
    "role": "transform",
    "summary": "Parameterized sparse dictionary operator across TopK, BatchTopK, JumpReLU, GatedSAE, and MatryoshkaNested with ZCA or Mahalanobis whitening.",
    "capabilities": ["sae_dictionary", "sparse_autoencoder", "zca_whitening", "matryoshka_sae"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    cell = dict(job.get("selected_grid_cell") or shell.grid_cell)
    if "sae_activation_family" in job:
        cell["sae_activation_family"] = str(job["sae_activation_family"])
    if "sae_whitening_metric" in job:
        cell["sae_whitening_metric"] = str(job["sae_whitening_metric"])

    x_in = np.asarray(job["activations"], dtype=np.float64)
    labels_raw = job.get("concept_labels")
    labels = np.asarray(labels_raw, dtype=np.int64) if labels_raw is not None else None
    res = fit_transform_sae(x_in, labels, cell, shell.registry.num)
    return {
        "activations": res["reconstructed"].tolist(),
        "sae_metrics": {
            "r2_score": res["r2_score"],
            "mean_l0": res["mean_l0"],
            "concept_selectivity_f1": res["concept_selectivity_f1"],
            "n_atoms": res["n_atoms"],
            "activation_family": res["activation_family"],
            "whitening_metric": res["whitening_metric"],
        },
        "quality_score": float(res["r2_score"]),
    }
