"""Portable 9-runtime compiler CARD lowering fitted OmniNode weights with in-memory C99 and WebAssembly parity."""

from __future__ import annotations

from typing import Any

import numpy as np

from omninode_manifold.compiler_9runtime import NineRuntimeOmniCompiler

CARD = {
    "name": "card_portable_9runtime_compiler",
    "role": "transform",
    "summary": "Lowers fitted float32 OmniNode weights into .wasm, Rust #![no_std], C99, WGSL, GLSL, SQL, JS, Python, and .safetensors with zero binary residue.",
    "capabilities": ["portable_9runtime_compiler", "wasm_lowering", "c99_parity", "rust_nostd", "safetensors"],
}


def execute(job: dict[str, Any], shell: Any) -> dict[str, Any]:
    num = shell.registry.num
    compiler = NineRuntimeOmniCompiler(shell.registry)

    raw_weights = job.get("fitted_weights")
    if raw_weights is None:
        rng = np.random.default_rng(int(num["default_rng_seed"]))
        in_d = int(num["probe_synth_dim_in"])
        hid_d = int(num["probe_synth_dim_hid"])
        out_d = int(num["probe_synth_dim_out"])
        raw_weights = {
            "w1": rng.standard_normal((hid_d, in_d)).astype(np.float32),
            "b1": rng.standard_normal(hid_d).astype(np.float32),
            "w2": rng.standard_normal((out_d, hid_d)).astype(np.float32),
            "b2": rng.standard_normal(out_d).astype(np.float32),
        }

    in_dim = int(np.asarray(raw_weights["w1"]).shape[1])
    raw_probe = job.get("probe_inputs")
    if raw_probe is None:
        rng_p = np.random.default_rng(int(num["alt_rng_seed"]))
        batch_n = int(shell.registry.compiler_grid["compiler_limits"]["parity_probe_batch"])
        x_probe = rng_p.standard_normal((batch_n, in_dim)).astype(np.float32)
    else:
        x_probe = np.asarray(raw_probe, dtype=np.float32)[:, :in_dim]

    report = compiler.lower_and_verify(raw_weights, x_probe)
    out: dict[str, Any] = {
        "compiler_report": report,
        "quality_score": 1.0 if bool(report["parity_passed"]) else 0.0,
    }
    if "activations" in job:
        out["activations"] = job["activations"]
    return out
