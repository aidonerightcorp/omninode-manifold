"""Universal parameterized latent, spectral, sheaf, conformal, steering, and solver kernels (zero magic literals)."""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np

from omninode_manifold.grid_ledger import OmniNodeGridRegistry, content_digest_hex

UNIVERSAL_SYSTEMS_STRATA: tuple[str, ...] = (
    "spectral_algebra",
    "content_addressed_state",
    "sheaf_quorum_routing",
    "dynamical_trajectory_lowering",
)

UNIVERSAL_SYSTEMS_CORPUS: tuple[tuple[str, str, int], ...] = (
    ("Singular value decomposition extracts principal orthogonal axes from dense activation matrices.", "spectral_algebra", 0),
    ("Immutable content stores address Python operator modules by deterministic byte digests.", "content_addressed_state", 1),
    ("Cellular sheaf coboundary operators measure vector disagreement across sibling graph edges.", "sheaf_quorum_routing", 0),
    ("WebAssembly binary emitters encode IEEE float32 matrix kernels into linear memory pages.", "dynamical_trajectory_lowering", 1),
    ("Economy singular spectrum factorization isolates dominant variance modes in high dimensions.", "spectral_algebra", 1),
    ("Content addressed module caches load pinned component files without directory scanning.", "content_addressed_state", 0),
    ("Sheaf Laplacian quadratic forms aggregate pairwise residual energy over parallel branches.", "sheaf_quorum_routing", 1),
    ("Portable WebAssembly modules export input offsets and float32 inference loops directly.", "dynamical_trajectory_lowering", 0),
    ("Oblique concept erasure removes linear nuisance covariance while preserving orthogonal subspaces.", "spectral_algebra", 0),
    ("Deterministic lockfiles bind every execution node path to an exact module digest.", "content_addressed_state", 1),
    ("Adaptive conformal intervals update empirical quantile thresholds online under distribution shift.", "sheaf_quorum_routing", 0),
    ("Frenet Serret discrete curvature tracks angular turning rates along latent state sequences.", "dynamical_trajectory_lowering", 1),
    ("Closed form oblique projections zero out cross covariance between representations and protected labels.", "spectral_algebra", 1),
    ("Pinned lockfile manifests reconstruct multi node trees across empty folders with zero searches.", "content_addressed_state", 0),
    ("Online conformal calibration adjusts miscoverage alpha step by step on streaming residuals.", "sheaf_quorum_routing", 1),
    ("Koopman spectral eigenvalues detect oscillatory limit cycles in recurrent trajectory rollouts.", "dynamical_trajectory_lowering", 0),
    ("Orthogonal Procrustes rotations align paired latent subspaces across distinct embedding widths.", "spectral_algebra", 0),
    ("Append only ledger files persist configuration cell outcomes using exclusive file locks.", "content_addressed_state", 1),
    ("Okapi BM25 lexical scoring weights term frequency saturation against document length.", "sheaf_quorum_routing", 0),
    ("Zero allocation no_std Rust kernels expose C ABI float32 matrix multiplication entrypoints.", "dynamical_trajectory_lowering", 1),
    ("Procrustes subspace alignment maps compact source vectors into wider target vector spaces.", "spectral_algebra", 1),
    ("Durable JSON line ledgers flush and fsync every recorded configuration cell trial.", "content_addressed_state", 0),
    ("Reciprocal rank fusion combines sparse lexical rankings with empirical outcome histories.", "sheaf_quorum_routing", 1),
    ("Standalone C99 shared libraries execute single precision affine layers with zero dependencies.", "dynamical_trajectory_lowering", 0),
    ("Kronecker factorized weight updates modulate dense layers inside anchor nullspace complements.", "spectral_algebra", 0),
    ("Bitemporal lease epochs reuse cached ridge weights until upstream epoch identifiers advance.", "content_addressed_state", 1),
    ("Disjoint maker and checker actor identifiers prevent self approval on quorum gates.", "sheaf_quorum_routing", 0),
    ("Fragment shaders in GLSL ES 300 compute dense layer outputs via texelFetch lookups.", "dynamical_trajectory_lowering", 1),
    ("Nullspace constrained Kronecker adapters leave invariant anchor projections unchanged.", "spectral_algebra", 1),
    ("Epoch scoped weight caches serve repeated solver queries in constant lookup time.", "content_addressed_state", 0),
    ("Independent quorum check nodes reject candidate states when maker equals checker.", "sheaf_quorum_routing", 1),
    ("WebGPU WGSL compute workgroups evaluate parallel float32 matrix vector products.", "dynamical_trajectory_lowering", 0),
    ("TopK sparse autoencoders reconstruct whitened activations using compact monosemantic dictionaries.", "spectral_algebra", 0),
    ("Portable git bundle archives package component source trees for offline snapshot replay.", "content_addressed_state", 1),
    ("Waterfall fallback combinators rerank transient and capability errors onto backup nodes.", "sheaf_quorum_routing", 0),
    ("Sinkhorn Knopp iterative scaling normalizes routing matrices to doubly stochastic marginals.", "dynamical_trajectory_lowering", 1),
    ("ZCA whitened sparse dictionary atoms separate correlated latent features cleanly.", "spectral_algebra", 1),
    ("Read only JSON object packs distribute self describing operator bricks across hosts.", "content_addressed_state", 0),
    ("Typed failure envelopes halt immediately on contract errors while recovering capability misses.", "sheaf_quorum_routing", 1),
    ("Balanced Sinkhorn transport equalizes row and column assignment probabilities iteratively.", "dynamical_trajectory_lowering", 0),
    ("Low rank Woodbury identities invert high dimensional gram matrices via dual systems.", "spectral_algebra", 0),
    ("Static abstract syntax tree literal evaluation reads component cards without code execution.", "content_addressed_state", 1),
    ("Autonomous gap builder loops draft new operator bricks after repeated capability misses.", "sheaf_quorum_routing", 0),
    ("Causal Green function kernels integrate exponentially decaying state histories across time.", "dynamical_trajectory_lowering", 1),
    ("Dual Woodbury ridge solvers accelerate wide feature regression when dimension exceeds sample count.", "spectral_algebra", 1),
    ("Syntax tree inspection extracts static metadata dictionaries safely prior to module import.", "content_addressed_state", 0),
    ("Self synthesizing capability loops test candidate modules on shown and hidden holdout suites.", "sheaf_quorum_routing", 1),
    ("Liquid state space convolutions smooth sequential representations using continuous decay rates.", "dynamical_trajectory_lowering", 0),
    ("Singular spectrum participation ratios quantify effective intrinsic rank of activation batches.", "spectral_algebra", 0),
    ("Immutable byte digest checks block tampered upstream files before module compilation.", "content_addressed_state", 1),
    ("Cellular sheaf consensus weights fuse parallel node predictions by inverse disagreement variance.", "sheaf_quorum_routing", 0),
    ("Binary WebAssembly cartridges run identically inside Node runtimes with zero disk files.", "dynamical_trajectory_lowering", 1),
    ("Oblique nuisance erasers drive protected attribute linear predictability down to chance.", "spectral_algebra", 1),
    ("Zero search lockfile hydration restores complete recursive execution graphs deterministically.", "content_addressed_state", 0),
    ("Adaptive conformal prediction sets maintain target empirical coverage across shifting regimes.", "sheaf_quorum_routing", 1),
    ("Trajectory curvature and Koopman modulus monitors halt repetitive loops early.", "dynamical_trajectory_lowering", 0),
    ("Paired Procrustes and ridge bridges translate embeddings across distinct model dimensions.", "spectral_algebra", 0),
    ("Fsynced grid/1 outcome rows tag every trial with its canonical configuration cell digest.", "content_addressed_state", 1),
    ("Hybrid BM25 and RRF discovery selects top matching operator cards from static catalogs.", "sheaf_quorum_routing", 0),
    ("No_std Rust and C99 native kernels match IEEE float32 reference accumulators bit for bit.", "dynamical_trajectory_lowering", 1),
    ("Anchor nullspace steering vectors shift target regimes with zero off target intrusion.", "spectral_algebra", 1),
    ("Epoch keyed solver caches eliminate redundant matrix factorizations within active leases.", "content_addressed_state", 0),
    ("Disjoint quorum checkers enforce strict separation between candidate makers and reviewers.", "sheaf_quorum_routing", 1),
    ("Safetensors binary buffers serialize float32 operator weights with exact shape metadata.", "dynamical_trajectory_lowering", 0),
)


def _deterministic_manifold_embeddings(
    texts: list[str],
    task_labels: np.ndarray,
    nuisance_labels: np.ndarray,
    dim: int,
    num: dict[str, Any],
    seed_offset: int = 0,
) -> np.ndarray:
    """Deterministic multi-factor latent representation used when local model runner is unreachable."""
    seed = int(num["default_rng_seed"]) + seed_offset
    rng = np.random.default_rng(seed)
    n_classes = int(num["four"])
    n_nuis = int(num["two"])
    half = float(num["half"])
    quarter = float(num["quarter"])
    eps = float(num["eps"])

    q_raw = rng.standard_normal((dim, n_classes + n_nuis + int(num["eight"])))
    q_orth, _ = np.linalg.qr(q_raw)
    class_dirs = q_orth[:, :n_classes].T
    nuis_dirs = q_orth[:, n_classes : n_classes + n_nuis].T
    shared_basis = q_orth[:, n_classes + n_nuis :]

    rows: list[np.ndarray] = []
    for idx, text in enumerate(texts):
        digest_b = bytes.fromhex(content_digest_hex(text))
        latent_c = np.array(
            [(float(digest_b[k]) / float(int(num["sixty_four"])) - 1.0 - 1.0) for k in range(shared_basis.shape[1])],
            dtype=np.float64,
        )
        vec = (
            class_dirs[int(task_labels[idx])]
            + half * nuis_dirs[int(nuisance_labels[idx])]
            + quarter * (shared_basis @ latent_c)
        )
        norm = float(np.linalg.norm(vec)) + eps
        rows.append(vec / norm)
    return np.vstack(rows)


def fetch_or_build_embedding_pair(registry: OmniNodeGridRegistry) -> dict[str, Any]:
    """Fetch real paired embeddings (384-d and 768-d) from local model runner or deterministic fallback."""
    lcfg = registry.local_runner
    num = registry.num
    stratum_idx = {name: idx for idx, name in enumerate(UNIVERSAL_SYSTEMS_STRATA)}
    texts = [item[0] for item in UNIVERSAL_SYSTEMS_CORPUS]
    task_labels = np.array([stratum_idx[item[1]] for item in UNIVERSAL_SYSTEMS_CORPUS], dtype=np.int64)
    nuisance_labels = np.array([item[int(num["two"])] for item in UNIVERSAL_SYSTEMS_CORPUS], dtype=np.int64)

    url = str(lcfg["embed_url"])
    small_tag = str(lcfg["small_model_tag"])
    large_tag = str(lcfg["large_model_tag"])
    timeout_s = float(lcfg["timeout_sec"])
    small_dim = int(lcfg["small_dim"])
    large_dim = int(lcfg["large_dim"])

    def _call_runner(model_tag: str) -> np.ndarray | None:
        req_body = json.dumps({"model": model_tag, "input": texts}).encode("utf-8")
        req = urllib.request.Request(url, data=req_body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embs = data.get("embeddings")
                if isinstance(embs, list) and len(embs) == len(texts):
                    return np.asarray(embs, dtype=np.float64)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None
        return None

    x_small = _call_runner(small_tag)
    x_large = _call_runner(large_tag)
    live_runner_used = x_small is not None and x_large is not None
    if x_small is None or x_large is None:
        x_small = _deterministic_manifold_embeddings(texts, task_labels, nuisance_labels, small_dim, num, seed_offset=0)
        x_large = _deterministic_manifold_embeddings(texts, task_labels, nuisance_labels, large_dim, num, seed_offset=0)

    half = float(num["half"])
    rng_nuis = np.random.default_rng(int(num["alt_rng_seed"]))
    nuis_vec_s = rng_nuis.standard_normal(x_small.shape[1])
    nuis_vec_s /= float(np.linalg.norm(nuis_vec_s)) + float(num["eps"])
    x_small_nuisance = x_small + half * ((nuisance_labels[:, None] * float(num["two"])) - 1.0) * nuis_vec_s[None, :]

    return {
        "texts": texts,
        "x_small": x_small,
        "x_small_nuisance": x_small_nuisance,
        "x_large": x_large,
        "task_labels": task_labels,
        "nuisance_labels": nuisance_labels,
        "live_runner_used": bool(live_runner_used),
        "small_model_tag": small_tag,
        "large_model_tag": large_tag,
    }


def orient_spectral_profile(x_mat: np.ndarray, num: dict[str, Any]) -> dict[str, Any]:
    """Compute Participation Ratio, top eigenvalue concentration, and Marchenko-Pastur RMT signal count."""
    eps = float(num["eps"])
    two = int(num["two"])
    arr = np.asarray(x_mat, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    n_samples, n_features = centered.shape
    denom = float(max(1, n_samples - 1))
    _, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
    eig_vals = (s_vals * s_vals) / denom
    sum_eig = float(np.sum(eig_vals)) + eps
    sum_sq = float(np.sum(eig_vals * eig_vals)) + eps
    participation_ratio = (sum_eig * sum_eig) / sum_sq
    anisotropy_ratio = float(eig_vals[0]) / sum_eig if eig_vals.size > 0 else 0.0

    median_eig = float(np.median(eig_vals)) if eig_vals.size > 0 else 0.0
    aspect = float(min(n_samples, n_features)) / float(max(n_samples, n_features))
    mp_edge = median_eig * ((1.0 + math.sqrt(aspect)) ** two)
    rmt_signal_dims = int(np.sum(eig_vals > mp_edge))
    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "participation_ratio": float(participation_ratio),
        "anisotropy_ratio": float(anisotropy_ratio),
        "rmt_signal_dims": rmt_signal_dims,
        "mp_upper_edge": float(mp_edge),
    }


def _apply_whitening(x_centered: np.ndarray, metric: str, svd_eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if metric == "None":
        eye = np.eye(x_centered.shape[1], dtype=np.float64)
        return x_centered, eye, eye
    n_s = float(max(1, x_centered.shape[0] - 1))
    _, s_vals, vt = np.linalg.svd(x_centered, full_matrices=False)
    eig = (s_vals * s_vals) / n_s
    shrink = svd_eps + float(np.mean(eig)) * svd_eps
    inv_sqrt = 1.0 / np.sqrt(eig + shrink)
    sqrt_eig = np.sqrt(eig + shrink)
    if metric == "ZCA":
        w_mat = (vt.T * inv_sqrt[None, :]) @ vt
        inv_w = (vt.T * sqrt_eig[None, :]) @ vt
        return x_centered @ w_mat, w_mat, inv_w
    w_maha = vt.T * inv_sqrt[None, :]
    inv_maha = (vt.T * sqrt_eig[None, :]).T
    return x_centered @ w_maha, w_maha, inv_maha


def _activate_codes(pre_act: np.ndarray, family: str, k_act: int, num: dict[str, Any]) -> np.ndarray:
    relu = np.maximum(pre_act, 0.0)
    n_s, n_atoms = relu.shape
    k_eff = max(1, min(int(k_act), n_atoms))
    if family == "TopK":
        out = np.zeros_like(relu)
        part_idx = np.argpartition(relu, -k_eff, axis=1)[:, -k_eff:]
        rows = np.arange(n_s)[:, None]
        out[rows, part_idx] = relu[rows, part_idx]
        return out
    if family == "BatchTopK":
        flat = relu.ravel()
        tot_k = max(1, min(flat.size, n_s * k_eff))
        out_f = np.zeros_like(flat)
        idx = np.argpartition(flat, -tot_k)[-tot_k:]
        out_f[idx] = flat[idx]
        return out_f.reshape(relu.shape)
    if family == "JumpReLU":
        q_val = float(num["jump_relu_quantile"])
        thresh = np.quantile(relu, q_val, axis=1, keepdims=True)
        return np.where(relu >= thresh, relu, 0.0)
    if family == "GatedSAE":
        gate = 1.0 / (1.0 + np.exp(-np.clip(pre_act, -float(num["eight"]), float(num["eight"]))))
        gated = relu * gate
        out = np.zeros_like(gated)
        part_idx = np.argpartition(gated, -k_eff, axis=1)[:, -k_eff:]
        rows = np.arange(n_s)[:, None]
        out[rows, part_idx] = gated[rows, part_idx]
        return out
    fracs = list(num["matryoshka_fractions"])
    accum = np.zeros_like(relu)
    weight = 1.0 / float(len(fracs))
    for frac in fracs:
        sub_atoms = max(1, int(round(float(n_atoms) * float(frac))))
        sub_k = max(1, min(sub_atoms, int(round(float(k_eff) * float(frac)))))
        slice_r = relu[:, :sub_atoms]
        sub_out = np.zeros_like(slice_r)
        p_idx = np.argpartition(slice_r, -sub_k, axis=1)[:, -sub_k:]
        rows = np.arange(n_s)[:, None]
        sub_out[rows, p_idx] = slice_r[rows, p_idx]
        accum[:, :sub_atoms] += weight * sub_out
    return accum


def fit_transform_sae(
    x_mat: np.ndarray,
    concept_labels: np.ndarray | None,
    cell: dict[str, Any],
    num: dict[str, Any],
) -> dict[str, Any]:
    """Fit parameterized Sparse Autoencoder across 5 activation families and 3 whitening metrics."""
    eps = float(num["eps"])
    svd_eps = float(num["svd_eps"])
    two = float(num["two"])
    arr = np.asarray(x_mat, dtype=np.float64)
    mean_v = np.mean(arr, axis=0, keepdims=True)
    centered = arr - mean_v

    metric = str(cell["sae_whitening_metric"])
    family = str(cell["sae_activation_family"])
    exp_factor = int(cell["sae_expansion_factor"])
    k_act = int(cell["sae_sparsity_k"])
    steps = int(cell["sae_refine_steps"])
    lr = float(cell["sae_step_scale"])

    z_white, _, inv_w = _apply_whitening(centered, metric, svd_eps)
    n_s, d_w = z_white.shape
    n_atoms = max(k_act, d_w * exp_factor)

    rng = np.random.default_rng(int(num["default_rng_seed"]))
    dict_w = rng.standard_normal((n_atoms, d_w))
    dict_w /= np.linalg.norm(dict_w, axis=1, keepdims=True) + eps
    for i_r in range(min(n_s, n_atoms)):
        row_norm = float(np.linalg.norm(z_white[i_r]))
        if row_norm > eps:
            dict_w[i_r] = z_white[i_r] / row_norm

    codes = np.zeros((n_s, n_atoms), dtype=np.float64)
    for _ in range(steps):
        pre = z_white @ dict_w.T
        codes = _activate_codes(pre, family, k_act, num)
        recon_w = codes @ dict_w
        err = z_white - recon_w
        grad = codes.T @ err
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True) + eps
        dict_w = dict_w + lr * (grad / grad_norms)
        dict_w /= np.linalg.norm(dict_w, axis=1, keepdims=True) + eps

    gram = codes.T @ codes
    trace_reg = float(num["hundredth"]) * (float(np.trace(gram)) / float(max(1, n_atoms)) + svd_eps)
    reg_mat = gram + trace_reg * np.eye(n_atoms, dtype=np.float64)
    dec_w = np.linalg.solve(reg_mat, codes.T @ z_white)
    recon_white = codes @ dec_w
    recon_orig = (recon_white @ inv_w) + mean_v

    total_var = float(np.sum((arr - mean_v) ** int(num["two"]))) + eps
    res_var = float(np.sum((arr - recon_orig) ** int(num["two"])))
    r2_score = max(0.0, 1.0 - res_var / total_var)
    mean_l0 = float(np.mean(np.sum(codes > eps, axis=1)))

    selectivity_f1 = 0.0
    if concept_labels is not None and concept_labels.size == n_s:
        active_bin = (codes > eps).astype(np.float64)
        uniq = np.unique(concept_labels)
        per_cls: list[float] = []
        for cls_v in uniq:
            mask = (concept_labels == cls_v).astype(np.float64)[:, None]
            tp = np.sum(active_bin * mask, axis=0)
            fp = np.sum(active_bin * (1.0 - mask), axis=0)
            fn = float(np.sum(mask)) - tp
            prec = tp / (tp + fp + eps)
            rec = tp / (tp + fn + eps)
            f1_vec = (two * prec * rec) / (prec + rec + eps)
            per_cls.append(float(np.max(f1_vec)))
        selectivity_f1 = float(np.mean(per_cls)) if per_cls else 0.0

    return {
        "reconstructed": recon_orig,
        "codes": codes,
        "r2_score": float(r2_score),
        "mean_l0": float(mean_l0),
        "concept_selectivity_f1": float(selectivity_f1),
        "n_atoms": int(n_atoms),
        "activation_family": family,
        "whitening_metric": metric,
    }


def fit_transform_crosscoder_transcoder(
    layer_views: list[np.ndarray],
    post_mlp_target: np.ndarray,
    cell: dict[str, Any],
    num: dict[str, Any],
) -> dict[str, Any]:
    """Multi-layer SparseCrosscoder + causal MlpTranscoder circuit approximation."""
    eps = float(num["eps"])
    n_atoms = int(cell["crosscoder_shared_atoms"])
    lam = float(cell["crosscoder_ridge_lambda"])
    k_act = int(cell["sae_sparsity_k"])

    concat_x = np.hstack([np.asarray(v, dtype=np.float64) for v in layer_views])
    target_y = np.asarray(post_mlp_target, dtype=np.float64)
    c_mean = np.mean(concat_x, axis=0, keepdims=True)
    xc = concat_x - c_mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    eff_atoms = min(n_atoms, vt.shape[0])
    enc = vt[:eff_atoms]
    shared_codes = _activate_codes(xc @ enc.T, "TopK", min(k_act, eff_atoms), num)

    gram = shared_codes.T @ shared_codes + lam * np.eye(eff_atoms, dtype=np.float64)
    w_cross = np.linalg.solve(gram, shared_codes.T @ xc)
    recon_concat = (shared_codes @ w_cross) + c_mean
    cross_r2 = max(
        0.0,
        1.0 - float(np.sum((concat_x - recon_concat) ** int(num["two"]))) / (float(np.sum(xc ** int(num["two"]))) + eps),
    )

    x_pre = np.asarray(layer_views[0], dtype=np.float64)
    pre_codes = _activate_codes(x_pre @ enc[:, : x_pre.shape[1]].T, "TopK", min(k_act, eff_atoms), num)
    gram_t = pre_codes.T @ pre_codes + lam * np.eye(eff_atoms, dtype=np.float64)
    w_trans = np.linalg.solve(gram_t, pre_codes.T @ target_y)
    pred_mlp = pre_codes @ w_trans
    y_cent = target_y - np.mean(target_y, axis=0, keepdims=True)
    trans_r2 = max(
        0.0,
        1.0 - float(np.sum((target_y - pred_mlp) ** int(num["two"]))) / (float(np.sum(y_cent ** int(num["two"]))) + eps),
    )
    return {
        "shared_codes": shared_codes,
        "transcoder_predictions": pred_mlp,
        "crosscoder_r2": float(cross_r2),
        "transcoder_r2": float(trans_r2),
        "shared_atom_count": int(eff_atoms),
    }


def evaluate_elk_sheaf_conformal(
    x_mat: np.ndarray,
    true_labels: np.ndarray,
    sycophantic_labels: np.ndarray,
    sibling_views: list[np.ndarray],
    cell: dict[str, Any],
    num: dict[str, Any],
) -> dict[str, Any]:
    """ELK contrast direction + cellular sheaf coboundary energy + ACI conformal calibration."""
    eps = float(num["eps"])
    two = float(num["two"])
    arr = np.asarray(x_mat, dtype=np.float64)
    y_true = np.asarray(true_labels, dtype=np.float64)
    y_syco = np.asarray(sycophantic_labels, dtype=np.float64)

    pos_m = y_true > float(num["half"])
    neg_m = ~pos_m
    mu_pos = np.mean(arr[pos_m], axis=0) if np.any(pos_m) else np.mean(arr, axis=0)
    mu_neg = np.mean(arr[neg_m], axis=0) if np.any(neg_m) else -mu_pos
    elk_dir = mu_pos - mu_neg
    elk_dir /= float(np.linalg.norm(elk_dir)) + eps

    s_pos = y_syco > float(num["half"])
    s_neg = ~s_pos
    syco_dir = (np.mean(arr[s_pos], axis=0) if np.any(s_pos) else mu_pos) - (
        np.mean(arr[s_neg], axis=0) if np.any(s_neg) else mu_neg
    )
    syco_dir /= float(np.linalg.norm(syco_dir)) + eps

    elk_clean = elk_dir - float(np.dot(elk_dir, syco_dir)) * syco_dir
    elk_clean /= float(np.linalg.norm(elk_clean)) + eps
    scores = arr @ elk_clean
    preds = (scores >= float(np.median(scores))).astype(np.float64)
    elk_acc = float(np.mean(preds == y_true))

    sheaf_energy = 0.0
    if len(sibling_views) >= int(num["two"]):
        min_d = min(v.shape[1] for v in sibling_views)
        normed = [
            v[:, :min_d] / (np.linalg.norm(v[:, :min_d], axis=1, keepdims=True) + eps)
            for v in sibling_views
        ]
        pairs = 0
        for i in range(len(normed)):
            for j in range(i + 1, len(normed)):
                diff = normed[i] - normed[j]
                sheaf_energy += float(np.mean(np.sum(diff * diff, axis=1))) / two
                pairs += 1
        if pairs > 0:
            sheaf_energy /= float(pairs)

    alpha_t = float(cell["conformal_alpha"])
    gamma = float(cell["conformal_aci_gamma"])
    probs = 1.0 / (1.0 + np.exp(-scores))
    nonconf = np.where(y_true > float(num["half"]), 1.0 - probs, probs)
    n_cal = max(1, arr.shape[0] // int(num["two"]))
    cal_scores = nonconf[:n_cal]
    covered: list[float] = []
    for t_idx in range(n_cal, arr.shape[0]):
        q_level = min(1.0, max(0.0, 1.0 - alpha_t))
        q_hat = float(np.quantile(cal_scores, q_level))
        hit = 1.0 if nonconf[t_idx] <= q_hat else 0.0
        covered.append(hit)
        err_t = 1.0 - hit
        alpha_t = min(float(num["half"]), max(float(num["hundredth"]), alpha_t + gamma * ( float(cell["conformal_alpha"]) - err_t)))

    empirical_cov = float(np.mean(covered)) if covered else 1.0
    sheaf_ceiling = float(cell["elk_sheaf_energy_ceiling"])
    check_passed = sheaf_energy <= sheaf_ceiling
    return {
        "elk_truth_accuracy": float(elk_acc),
        "sheaf_coboundary_energy": float(sheaf_energy),
        "sheaf_energy_ceiling": float(sheaf_ceiling),
        "conformal_empirical_coverage": float(empirical_cov),
        "check_passed": bool(check_passed),
    }


def apply_leace_concept_eraser(
    x_mat: np.ndarray,
    z_nuisance: np.ndarray,
    cell: dict[str, Any],
    num: dict[str, Any],
) -> dict[str, Any]:
    """Closed-form Economy-SVD Belrose LEACE oblique concept erasure driving Cov(PX, Z) -> 0."""
    eps = float(num["eps"])
    svd_eps = float(num["svd_eps"])
    shrink = float(cell["leace_shrinkage_factor"])
    use_ipw = bool(cell["leace_ipw_balance"])

    x_arr = np.asarray(x_mat, dtype=np.float64)
    z_raw = np.asarray(z_nuisance, dtype=np.float64)
    if z_raw.ndim == 1:
        z_arr = z_raw[:, None]
    else:
        z_arr = z_raw

    n_s = x_arr.shape[0]
    if use_ipw and z_arr.shape[1] == 1:
        vals, counts = np.unique(z_arr[:, 0], return_counts=True)
        w_sample = np.ones(n_s, dtype=np.float64)
        for v_item, c_item in zip(vals, counts):
            w_sample[z_arr[:, 0] == v_item] = float(n_s) / (float(len(vals)) * float(max(1, c_item)))
        w_sample /= float(np.sum(w_sample))
    else:
        w_sample = np.full(n_s, 1.0 / float(max(1, n_s)), dtype=np.float64)

    mu_x = np.sum(x_arr * w_sample[:, None], axis=0, keepdims=True)
    mu_z = np.sum(z_arr * w_sample[:, None], axis=0, keepdims=True)
    xc = x_arr - mu_x
    zc = z_arr - mu_z
    sqrt_w = np.sqrt(w_sample)[:, None]
    xw = xc * sqrt_w
    zw = zc * sqrt_w

    _, s_vals, vt = np.linalg.svd(xw, full_matrices=False)
    eig = s_vals * s_vals
    reg = shrink * float(np.mean(eig)) + svd_eps
    inv_sqrt = 1.0 / np.sqrt(eig + reg)
    sqrt_reg = np.sqrt(eig + reg)

    sigma_xz = xw.T @ zw
    whitened_cross = (vt @ sigma_xz) * inv_sqrt[:, None]
    u_c, _, _ = np.linalg.svd(whitened_cross, full_matrices=False)
    rank_z = min(z_arr.shape[1], u_c.shape[1])
    u_k = u_c[:, :rank_z]

    left_oblique = vt.T @ (sqrt_reg[:, None] * u_k)
    right_oblique = (u_k.T * inv_sqrt[None, :]) @ vt
    proj_remove_x = left_oblique @ right_oblique
    x_clean_prelim = xc - (xc @ proj_remove_x.T)

    zz_cov = zw.T @ zw + svd_eps * np.eye(z_arr.shape[1], dtype=np.float64)
    xz_resid = (x_clean_prelim * w_sample[:, None]).T @ zc
    beta_resid = np.linalg.solve(zz_cov, xz_resid.T)
    x_erased = (x_clean_prelim - (zc @ beta_resid)) + mu_x

    cov_before = float(np.max(np.abs((xc * w_sample[:, None]).T @ zc)))
    xc_after = x_erased - np.sum(x_erased * w_sample[:, None], axis=0, keepdims=True)
    cov_after = float(np.max(np.abs((xc_after * w_sample[:, None]).T @ zc)))

    return {
        "erased_activations": x_erased,
        "max_abs_cov_before": float(cov_before),
        "max_abs_cov_after": float(cov_after),
        "nuisance_rank_removed": int(rank_z),
        "eps_used": eps,
    }


def apply_conditional_nullspace_steerer(
    x_mat: np.ndarray,
    is_target_regime: np.ndarray,
    anchor_subspace: np.ndarray,
    cell: dict[str, Any],
    num: dict[str, Any],
) -> dict[str, Any]:
    """Nullspace-constrained steering P_{Null(A_anchor)} with shrinkage Fisher-centroid regime gate."""
    eps = float(num["eps"])
    svd_eps = float(num["svd_eps"])
    alpha = float(cell["steerer_strength_alpha"])
    pos_q = float(cell["steerer_gate_pos_quantile_weight"])

    arr = np.asarray(x_mat, dtype=np.float64)
    mask = np.asarray(is_target_regime, dtype=bool)
    anchors = np.asarray(anchor_subspace, dtype=np.float64)
    if anchors.ndim == 1:
        anchors = anchors[None, :]

    q_anc, _ = np.linalg.qr(anchors.T)
    pos_mean = np.mean(arr[mask], axis=0) if np.any(mask) else np.mean(arr, axis=0)
    neg_mean = np.mean(arr[~mask], axis=0) if np.any(~mask) else -pos_mean
    raw_steer = pos_mean - neg_mean
    null_steer = raw_steer - q_anc @ (q_anc.T @ raw_steer)
    null_steer /= float(np.linalg.norm(null_steer)) + eps

    var_diag = np.var(arr, axis=0) + float(np.mean(np.var(arr, axis=0))) + svd_eps
    gate_dir = (pos_mean - neg_mean) / var_diag
    gate_dir /= float(np.linalg.norm(gate_dir)) + eps
    gate_scores = arr @ gate_dir

    low_pos = float(np.min(gate_scores[mask])) if np.any(mask) else 0.0
    high_neg = float(np.max(gate_scores[~mask])) if np.any(~mask) else -1.0
    threshold = pos_q * low_pos + (1.0 - pos_q) * high_neg
    fired = gate_scores >= threshold

    steered = arr + (fired[:, None].astype(np.float64) * alpha) * null_steer[None, :]
    anchor_drift = float(np.max(np.abs((steered - arr) @ q_anc)))
    off_target_rate = float(np.mean(fired[~mask])) if np.any(~mask) else 0.0
    on_target_rate = float(np.mean(fired[mask])) if np.any(mask) else 1.0

    return {
        "steered_activations": steered,
        "anchor_max_abs_drift": float(anchor_drift),
        "off_target_intrusion_rate": float(off_target_rate),
        "on_target_activation_rate": float(on_target_rate),
        "gate_threshold": float(threshold),
    }


def apply_platonic_latent_bridge(
    x_src: np.ndarray,
    x_tgt: np.ndarray,
    train_split: int,
    cell: dict[str, Any],
    num: dict[str, Any],
) -> dict[str, Any]:
    """Zero-retraining Orthogonal Procrustes + Ridge cross-space bridge across heterogeneous dimensions."""
    eps = float(num["eps"])
    rank = int(cell["bridge_subspace_rank"])
    lam = float(cell["bridge_ridge_lambda"])

    src = np.asarray(x_src, dtype=np.float64)
    tgt = np.asarray(x_tgt, dtype=np.float64)
    src_n = src / (np.linalg.norm(src, axis=1, keepdims=True) + eps)
    tgt_n = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + eps)

    n_tr = max(int(num["two"]), min(int(train_split), src_n.shape[0] - int(num["two"])))
    s_tr, s_te = src_n[:n_tr], src_n[n_tr:]
    t_tr, t_te = tgt_n[:n_tr], tgt_n[n_tr:]

    mu_s = np.mean(s_tr, axis=0, keepdims=True)
    mu_t = np.mean(t_tr, axis=0, keepdims=True)
    sc = s_tr - mu_s
    tc = t_tr - mu_t

    _, _, vt_s = np.linalg.svd(sc, full_matrices=False)
    _, _, vt_t = np.linalg.svd(tc, full_matrices=False)
    r_eff = min(rank, max(1, n_tr - 1), vt_s.shape[0], vt_t.shape[0])
    basis_s = vt_s[:r_eff].T
    basis_t = vt_t[:r_eff].T

    z_s = sc @ basis_s
    z_t = tc @ basis_t
    u_p, _, vt_p = np.linalg.svd(z_s.T @ z_t, full_matrices=False)
    r_proc = u_p @ vt_p

    proc_tr = (z_s @ r_proc) @ basis_t.T
    residual_tr = tc - proc_tr
    gram_dual = sc @ sc.T + lam * np.eye(n_tr, dtype=np.float64)
    dual_w = np.linalg.solve(gram_dual, residual_tr)
    w_res = sc.T @ dual_w

    s_all_c = src_n - mu_s
    proc_all = ((s_all_c @ basis_s) @ r_proc) @ basis_t.T
    mapped_all = proc_all + (s_all_c @ w_res) + mu_t
    mapped_all /= np.linalg.norm(mapped_all, axis=1, keepdims=True) + eps

    mapped_te = mapped_all[n_tr:]
    sims = mapped_te @ t_te.T
    top1_hits = np.argmax(sims, axis=1) == np.arange(t_te.shape[0])
    top1_acc = float(np.mean(top1_hits)) if t_te.shape[0] > 0 else 1.0
    cos_align = float(np.mean(np.sum(mapped_te * t_te, axis=1))) if t_te.shape[0] > 0 else 1.0

    return {
        "mapped_activations": mapped_all,
        "held_out_top1_retrieval": float(top1_acc),
        "held_out_cosine_alignment": float(cos_align),
        "bridge_rank": int(r_eff),
        "source_dim": int(src.shape[1]),
        "target_dim": int(tgt.shape[1]),
    }


def analyze_trajectory_loop(
    trajectory: np.ndarray,
    cell: dict[str, Any],
    num: dict[str, Any],
) -> dict[str, Any]:
    """Frenet-Serret curvature, sliding Participation Ratio, and Koopman modulus loop breaker."""
    eps = float(num["eps"])
    svd_eps = float(num["svd_eps"])
    win = int(cell["trajectory_window_steps"])
    koop_limit = float(cell["trajectory_koopman_modulus_limit"])
    pr_floor = float(cell["orient_participation_low_threshold"]) * float(num["half"])

    traj = np.asarray(trajectory, dtype=np.float64)
    n_steps = traj.shape[0]
    diffs = np.diff(traj, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True) + eps
    tangents = diffs / norms

    curvatures: list[float] = []
    for idx in range(tangents.shape[0] - 1):
        dt_vec = tangents[idx + 1] - tangents[idx]
        ds = float(norms[idx, 0])
        curvatures.append(float(np.linalg.norm(dt_vec)) / (ds + eps))

    halt_step: int | None = None
    max_koopman = 0.0
    min_pr = float(n_steps)
    for t_end in range(win + 1, n_steps + 1):
        sub = traj[t_end - win - 1 : t_end]
        sub_c = sub - np.mean(sub, axis=0, keepdims=True)
        _, sv, _ = np.linalg.svd(sub_c, full_matrices=False)
        ev = sv * sv
        pr_w = ((float(np.sum(ev)) + eps) ** int(num["two"])) / (float(np.sum(ev * ev)) + eps)
        min_pr = min(min_pr, pr_w)

        x0 = sub_c[:-1]
        x1 = sub_c[1:]
        _, sv0, vt0 = np.linalg.svd(x0, full_matrices=False)
        r_k = min(int(num["two"]), sv0.size)
        if r_k > 0 and float(sv0[0]) > svd_eps:
            z0 = x0 @ vt0[:r_k].T
            z1 = x1 @ vt0[:r_k].T
            k_mat = np.linalg.solve(z0.T @ z0 + svd_eps * np.eye(r_k), z0.T @ z1)
            eig_k = np.linalg.eigvals(k_mat)
            mod_k = float(np.max(np.abs(eig_k)))
            max_koopman = max(max_koopman, mod_k)
            if mod_k >= koop_limit and pr_w <= pr_floor and halt_step is None:
                halt_step = t_end - 1

    saved_ratio = float(n_steps - halt_step) / float(max(1, n_steps)) if halt_step is not None else 0.0
    return {
        "loop_detected": halt_step is not None,
        "halt_step": halt_step,
        "total_steps": int(n_steps),
        "token_savings_ratio": float(saved_ratio),
        "max_koopman_modulus": float(max_koopman),
        "min_window_participation_ratio": float(min_pr),
        "mean_frenet_curvature": float(np.mean(curvatures)) if curvatures else 0.0,
    }


def apply_hyperlora_liquid_operator(
    x_seq: np.ndarray,
    context_vec: np.ndarray,
    anchor_dirs: np.ndarray | None,
    cell: dict[str, Any],
    num: dict[str, Any],
) -> dict[str, Any]:
    """Input-conditioned Kronecker nullspace HyperLoRA + Green's-function LiquidStateSpace."""
    eps = float(num["eps"])
    rank = int(cell["hyperlora_kronecker_rank"])
    decay = float(cell["liquid_decay_rate"])

    seq = np.asarray(x_seq, dtype=np.float64)
    ctx = np.asarray(context_vec, dtype=np.float64).ravel()
    n_t, d_model = seq.shape

    p_f = max(1, int(math.isqrt(d_model)))
    while d_model % p_f != 0 and p_f > 1:
        p_f -= 1
    q_f = d_model // p_f

    rng = np.random.default_rng(int(num["default_rng_seed"]))
    ctx_norm = ctx / (float(np.linalg.norm(ctx)) + eps)
    proj_c = rng.standard_normal((rank, ctx_norm.size))
    gates = np.tanh(proj_c @ ctx_norm)

    delta_w = np.zeros((d_model, d_model), dtype=np.float64)
    scale = float(num["tenth"]) / float(max(1, rank))
    for r_i in range(rank):
        a_fac = rng.standard_normal((p_f, p_f))
        b_fac = rng.standard_normal((q_f, q_f))
        a_fac /= float(np.linalg.norm(a_fac)) + eps
        b_fac /= float(np.linalg.norm(b_fac)) + eps
        delta_w += float(gates[r_i]) * scale * np.kron(a_fac, b_fac)

    anchor_drift = 0.0
    if anchor_dirs is not None:
        anc = np.asarray(anchor_dirs, dtype=np.float64)
        if anc.ndim == 1:
            anc = anc[None, :]
        q_anc, _ = np.linalg.qr(anc.T)
        delta_w = delta_w - (delta_w @ q_anc) @ q_anc.T
        anchor_drift = float(np.max(np.abs(delta_w @ q_anc)))

    modulated = seq + (seq @ delta_w.T)
    idx = np.arange(n_t, dtype=np.float64)
    diff_t = idx[:, None] - idx[None, :]
    green_k = np.where(diff_t >= 0.0, np.exp(-decay * np.maximum(diff_t, 0.0)), 0.0)
    green_k /= np.sum(green_k, axis=1, keepdims=True) + eps
    liquid_out = green_k @ modulated

    return {
        "activations": liquid_out,
        "kronecker_factors": [int(p_f), int(q_f)],
        "kronecker_rank": int(rank),
        "anchor_nullspace_drift": float(anchor_drift),
    }


def solve_woodbury_sinkhorn_attributed(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    lease_epoch: str,
    cache_store: dict[str, Any],
    cell: dict[str, Any],
    num: dict[str, Any],
) -> dict[str, Any]:
    """Woodbury primal/dual ridge + (eps, delta)-DP RLS + Sinkhorn-Knopp balanced routing + lease cache."""
    eps = float(num["eps"])
    lam = float(cell["woodbury_ridge_lambda"])
    dp_eps = float(cell["dp_rls_epsilon"])
    sink_iters = int(cell["sinkhorn_iterations"])

    arr_x = np.asarray(x_mat, dtype=np.float64)
    arr_y = np.asarray(y_mat, dtype=np.float64)
    if arr_y.ndim == 1:
        arr_y = arr_y[:, None]

    cache_key = content_digest_hex(
        f"{lease_epoch}:{arr_x.shape}:{arr_y.shape}:{lam}:{dp_eps}:{sink_iters}"
    )
    if cache_key in cache_store:
        cached = dict(cache_store[cache_key])
        cached["cache_hit"] = True
        return cached

    n_s, n_d = arr_x.shape
    if n_d > n_s:
        gram_dual = arr_x @ arr_x.T + lam * np.eye(n_s, dtype=np.float64)
        dual_alpha = np.linalg.solve(gram_dual, arr_y)
        w_woodbury = arr_x.T @ dual_alpha
    else:
        gram_primal = arr_x.T @ arr_x + lam * np.eye(n_d, dtype=np.float64)
        w_woodbury = np.linalg.solve(gram_primal, arr_x.T @ arr_y)

    clip_c = float(num["dp_grad_clip"])
    delta = float(num["dp_delta"])
    x_norms = np.linalg.norm(arr_x, axis=1, keepdims=True) + eps
    x_clip = arr_x * np.minimum(1.0, clip_c / x_norms)
    rng = np.random.default_rng(int(num["default_rng_seed"]))
    sigma_dp = (clip_c * math.sqrt(float(num["two"]) * math.log(1.0 / delta))) / (dp_eps * float(max(1, n_s)))
    sym_noise = rng.normal(0.0, sigma_dp, size=(n_d, n_d))
    sym_noise = float(num["half"]) * (sym_noise + sym_noise.T)
    dp_gram = (x_clip.T @ x_clip) / float(max(1, n_s)) + sym_noise + (lam + sigma_dp * float(num["two"])) * np.eye(n_d)
    dp_rhs = (x_clip.T @ arr_y) / float(max(1, n_s))
    w_dp = np.linalg.solve(dp_gram, dp_rhs)

    logits = arr_x @ w_woodbury
    k_cols = logits.shape[1]
    square_n = min(n_s, max(int(num["two"]), k_cols))
    cost_m = logits[:square_n, :k_cols]
    mat_s = np.exp(cost_m - np.max(cost_m, axis=1, keepdims=True)) + eps
    for _ in range(sink_iters):
        mat_s = mat_s / (np.sum(mat_s, axis=1, keepdims=True) + eps)
        mat_s = mat_s / (np.sum(mat_s, axis=0, keepdims=True) + eps) * (float(square_n) / float(max(1, k_cols)))

    preds = arr_x @ w_woodbury
    y_cent = arr_y - np.mean(arr_y, axis=0, keepdims=True)
    r2_val = max(
        0.0,
        1.0 - float(np.sum((arr_y - preds) ** int(num["two"]))) / (float(np.sum(y_cent ** int(num["two"]))) + eps),
    )
    row_sum_err = float(np.max(np.abs(np.sum(mat_s, axis=1) - 1.0)))
    result = {
        "weights": w_woodbury.T.astype(np.float32),
        "dp_weights": w_dp.T.astype(np.float32),
        "predictions": preds,
        "woodbury_r2": float(r2_val),
        "sinkhorn_row_marginal_error": float(row_sum_err),
        "lease_epoch": lease_epoch,
        "cache_hit": False,
    }
    cache_store[cache_key] = dict(result)
    return result
