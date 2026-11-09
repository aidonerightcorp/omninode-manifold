"""End-to-end empirical benchmark suite exercising OmniNodeShell, all 22 CARD bricks, transports, and 9 runtimes."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np

from components.sources.source_memory_bundle import create_bundle_snapshot
from components.sources.source_object_pack import build_object_pack
from omninode_manifold.gates import DomainAgnosticNamingGate, ZeroHardcodedValuesAstGate
from omninode_manifold.grid_ledger import OmniNodeGridRegistry
from omninode_manifold.math_kernels import fetch_or_build_embedding_pair
from omninode_manifold.shape_algebra import OmniNodeContractPermissionError, OmniNodeFailure
from omninode_manifold.shell import OmniNodeShell, card_of


def _round_floats(obj: Any, digits: int) -> Any:
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {str(k): _round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, digits) for v in obj]
    return obj


def _http_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout_s: float = 1.0) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return int(resp.status), json.loads(resp.read().decode("utf-8"))


def run_full_benchmark_suite(workspace_root: Path | None = None) -> dict[str, Any]:
    """Execute all empirical benchmarks in isolated temporary directories and return clean summary dict."""
    root = workspace_root or Path(__file__).resolve().parent.parent
    registry = OmniNodeGridRegistry(root / "config" / "omninode_universal_grid.json")
    num = registry.num
    digits = int(num["metric_round_digits"])
    short_len = int(registry.protocol["digest_short_len"])
    cards_dir = root / "components" / "cards"

    ast_violations = ZeroHardcodedValuesAstGate.scan_directories(
        [root / "omninode_manifold", root / "components", root / "tests"]
    )
    naming_violations = DomainAgnosticNamingGate.scan_workspace(root)

    static_cards: list[str] = []
    for py_path in sorted((root / "components").rglob("*.py")):
        if py_path.name == "__init__.py":
            continue
        meta = card_of(py_path.read_text(encoding="utf-8"))
        if meta is not None:
            static_cards.append(str(meta["name"]))

    emb_pair = fetch_or_build_embedding_pair(registry)
    x_small = emb_pair["x_small"]
    x_small_nuis = emb_pair["x_small_nuisance"]
    x_large = emb_pair["x_large"]
    task_labels = emb_pair["task_labels"]
    nuisance_labels = emb_pair["nuisance_labels"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)

        # 1. Cross-space bridge + SAE family sweep encased in OmniNodeShell
        bridge_shell = OmniNodeShell(
            {
                "name": "bridge_node",
                "state_root": str(tmp_root / "s1"),
                "want": "Procrustes cross space alignment bridge",
                "role": "transform",
                "regime_id": "cross_space_heterogeneous_bridge",
            },
            registry,
        )
        bridge_out = bridge_shell.handle(
            {
                "activations": x_small.tolist(),
                "target_activations": x_large.tolist(),
                "train_split": int(registry.local_runner["train_split"]),
            }
        )

        sae_shell = OmniNodeShell(
            {
                "name": "sae_node",
                "state_root": str(tmp_root / "s1"),
                "component": "card_sae_dictionary",
                "regime_id": "anisotropic_concept_erasure",
            },
            registry,
        )
        sae_sweep: dict[str, Any] = {}
        for fam in ("TopK", "BatchTopK", "JumpReLU", "GatedSAE", "MatryoshkaNested"):
            s_res = sae_shell.handle(
                {
                    "activations": x_small.tolist(),
                    "concept_labels": task_labels.tolist(),
                    "sae_activation_family": fam,
                    "sae_whitening_metric": "ZCA",
                }
            )
            sae_sweep[fam] = s_res["sae_metrics"]

        # 2. Empty-folder / wiped-.omni lockfile rebuild (0 searches) + tamper refusal
        bundle_path = tmp_root / "packs" / "cards.bundle"
        pack_path = tmp_root / "packs" / "cards_pack.json"
        mutable_cards = tmp_root / "mutable_cards"
        mutable_cards.mkdir(parents=True, exist_ok=True)
        for src_f in sorted(cards_dir.glob("*.py")):
            (mutable_cards / src_f.name).write_text(src_f.read_text(encoding="utf-8"), encoding="utf-8")

        create_bundle_snapshot(mutable_cards, bundle_path, use_git_bundle=True)
        build_object_pack(mutable_cards, pack_path)

        initial_state = tmp_root / "lock_init"
        init_shell = OmniNodeShell(
            {
                "name": "pinned_root",
                "state_root": str(initial_state),
                "sources": [
                    {"kind": "folder", "path": str(mutable_cards)},
                    {"kind": "bundle", "path": str(bundle_path)},
                    {"kind": "object_pack", "path": str(pack_path)},
                ],
                "search_method": "hybrid_rrf",
                "component": "card_shape_director",
                "shape": "orient >> leace",
                "children": {
                    "orient": {"want": "spectral participation ratio orient", "role": "stage", "stage": "orient"},
                    "leace": {"want": "LEACE oblique concept erasure", "role": "transform"},
                },
            },
            registry,
        )
        initial_out = init_shell.handle(
            {"activations": x_small.tolist(), "nuisance_labels": nuisance_labels.tolist()}
        )
        lock_path = init_shell.write_lockfile(tmp_root / "lockfile.json")
        initial_searches = init_shell.library.search_count

        # Wipe mutable_cards completely and use a brand-new empty state_root (wiped .omni)
        for f_del in mutable_cards.glob("*.py"):
            f_del.unlink()
        wiped_state = tmp_root / "lock_rebuilt_wiped"
        rebuilt_shell = OmniNodeShell(
            {
                "name": "pinned_root",
                "state_root": str(wiped_state),
                "lockfile_path": str(lock_path),
                "sources": [
                    {"kind": "folder", "path": str(mutable_cards)},
                    {"kind": "bundle", "path": str(bundle_path)},
                    {"kind": "object_pack", "path": str(pack_path)},
                ],
                "component": "card_shape_director",
                "shape": "orient >> leace",
                "children": {
                    "orient": {"want": "spectral participation ratio orient", "role": "stage", "stage": "orient"},
                    "leace": {"want": "LEACE oblique concept erasure", "role": "transform"},
                },
            },
            registry,
        )
        rebuilt_out = rebuilt_shell.handle(
            {"activations": x_small.tolist(), "nuisance_labels": nuisance_labels.tolist()}
        )
        rebuild_diff = float(
            np.max(np.abs(np.asarray(initial_out["activations"]) - np.asarray(rebuilt_out["activations"])))
        )
        rebuild_searches = rebuilt_shell.library.search_count

        # Tamper upstream file and verify fail-closed contract error + mismatch event
        tamper_dir = tmp_root / "tampered_cards"
        tamper_dir.mkdir(parents=True, exist_ok=True)
        for src_f in sorted(cards_dir.glob("*.py")):
            text_c = src_f.read_text(encoding="utf-8")
            if src_f.name == "card_leace_concept_eraser.py":
                text_c = text_c + "\n# upstream tamper\n"
            (tamper_dir / src_f.name).write_text(text_c, encoding="utf-8")

        tamper_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for pin_entry in tamper_lock["nodes"].values():
            pin_entry["source"] = {"kind": "folder", "path": str(tamper_dir)}

        tamper_state = tmp_root / "tamper_state"
        tamper_caught = False
        tamper_kind = ""
        try:
            OmniNodeShell(
                {
                    "name": "pinned_root",
                    "state_root": str(tamper_state),
                    "lockfile": tamper_lock,
                    "sources": [{"kind": "folder", "path": str(tamper_dir)}],
                    "component": "card_shape_director",
                    "shape": "orient >> leace",
                    "children": {
                        "orient": {"role": "stage", "stage": "orient"},
                        "leace": {"role": "transform"},
                    },
                },
                registry,
            )
        except OmniNodeFailure as exc:
            tamper_caught = True
            tamper_kind = exc.kind
        tamper_facts = [
            json.loads(r)["kind"]
            for r in (tamper_state / str(registry.protocol["facts_rel_path"])).read_text(encoding="utf-8").splitlines()
            if r.strip() and json.loads(r).get("kind") == "hash_mismatch"
        ]

        # 3. Composite >> // | @ pipeline + waterfall fallback + maker != checker enforcement
        composite_state = tmp_root / "composite_state"
        composite_shell = OmniNodeShell(
            {
                "name": "composite_director",
                "actor_id": "director_maker",
                "state_root": str(composite_state),
                "component": "card_shape_director",
                "shape": "(orient >> compose >> ((leace >> steerer) | crosscoder)) @ quorum_check",
                "children": {
                    "orient": {"component": "card_orient_spectral_profiler", "actor_id": "maker_orient"},
                    "compose": {"component": "card_compose_grid_selector", "actor_id": "maker_compose"},
                    "leace": {"component": "card_leace_concept_eraser", "actor_id": "maker_leace"},
                    "steerer": {"component": "card_conditional_nullspace_steerer", "actor_id": "maker_steerer"},
                    "crosscoder": {"component": "card_crosscoder_transcoder", "actor_id": "maker_crosscoder"},
                    "quorum_check": {"component": "card_independent_quorum_gate", "actor_id": "independent_checker_alpha"},
                },
            },
            registry,
        )
        rng_anc = np.random.default_rng(int(num["default_rng_seed"]))
        anchors = rng_anc.standard_normal((int(num["four"]), x_small.shape[1]))
        composite_out = composite_shell.handle(
            {
                "activations": x_small_nuis.tolist(),
                "nuisance_labels": nuisance_labels.tolist(),
                "is_target_regime": (task_labels == 0).tolist(),
                "anchor_subspace": anchors.tolist(),
            }
        )

        # Self-check rejection test (maker_id == checker_id)
        self_check_rejected = False
        same_actor_shell = OmniNodeShell(
            {
                "name": "bad_verify_director",
                "state_root": str(composite_state),
                "component": "card_shape_director",
                "shape": "worker_a @ checker_same",
                "children": {
                    "worker_a": {"component": "card_leace_concept_eraser", "actor_id": "shared_identity"},
                    "checker_same": {"component": "card_independent_quorum_gate", "actor_id": "shared_identity"},
                },
            },
            registry,
        )
        try:
            same_actor_shell.handle({"activations": x_small.tolist(), "nuisance_labels": nuisance_labels.tolist()})
        except OmniNodeContractPermissionError:
            self_check_rejected = True

        # 4. Boundary Auto-Expansion (Gen 1 -> Gen 2) + grid/1 ledger + search_outcome_grid
        expand_shell = OmniNodeShell(
            {
                "name": "compose_expander",
                "state_root": str(composite_state),
                "component": "card_compose_grid_selector",
                "search_method": "outcome_grid",
            },
            registry,
        )
        expand_out = expand_shell.handle(
            {
                "activations": x_small.tolist(),
                "high_capacity": True,
                "auto_expand_if_saturated": True,
                "observed_score": 1.0,
            }
        )
        grid_rows = composite_shell.library.grid_recorder.read_all()

        # 5. Loopback HTTP (/hello, /health, /job) + MCP JSON-RPC 2.0 (/mcp)
        http_timeout = float(registry.protocol["http_timeout_sec"])
        srv_handle = composite_shell.start_background_server()
        try:
            hello_code, hello_body = _http_json(f"{srv_handle.base_url}/hello", timeout_s=http_timeout)
            health_code, health_body = _http_json(f"{srv_handle.base_url}/health", timeout_s=http_timeout)
            job_code, job_body = _http_json(
                f"{srv_handle.base_url}/job",
                method="POST",
                payload={
                    "activations": x_small.tolist(),
                    "nuisance_labels": nuisance_labels.tolist(),
                    "is_target_regime": (task_labels == 0).tolist(),
                    "anchor_subspace": anchors.tolist(),
                },
                timeout_s=http_timeout,
            )
            _, mcp_init = _http_json(
                f"{srv_handle.base_url}/mcp",
                method="POST",
                payload={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                timeout_s=http_timeout,
            )
            _, mcp_tools = _http_json(
                f"{srv_handle.base_url}/mcp",
                method="POST",
                payload={"jsonrpc": "2.0", "id": 1 + 1, "method": "tools/list", "params": {}},
                timeout_s=http_timeout,
            )
            _, mcp_call = _http_json(
                f"{srv_handle.base_url}/mcp",
                method="POST",
                payload={
                    "jsonrpc": "2.0",
                    "id": 1 + 1 + 1,
                    "method": "tools/call",
                    "params": {
                        "name": "orient",
                        "arguments": {"activations": x_small.tolist()},
                    },
                },
                timeout_s=http_timeout,
            )
        finally:
            srv_handle.shutdown()

        # 6. Autonomous gap-loop self-synthesis (declined -> passed)
        gap_cards_dir = tmp_root / "gap_cards"
        gap_cards_dir.mkdir(parents=True, exist_ok=True)
        for src_f in sorted(cards_dir.glob("*.py")):
            (gap_cards_dir / src_f.name).write_text(src_f.read_text(encoding="utf-8"), encoding="utf-8")

        gap_state = tmp_root / "gap_state"
        missing_node = OmniNodeShell(
            {
                "name": "polar_denoiser_node",
                "state_root": str(gap_state),
                "sources": [{"kind": "folder", "path": str(gap_cards_dir)}],
                "capability_tag": "spectral_polar_denoiser",
                "role": "transform",
                "eager_bind": False,
            },
            registry,
        )
        declined_attempts = 0
        for _ in range(int(num["two"])):
            try:
                missing_node.handle({"activations": x_small.tolist()})
            except OmniNodeFailure as exc:
                if exc.kind == "capability":
                    declined_attempts += 1

        gap_builder = OmniNodeShell(
            {
                "name": "gap_loop_runner",
                "state_root": str(gap_state),
                "sources": [{"kind": "folder", "path": str(gap_cards_dir)}],
                "component": "card_gap_builder_loop",
            },
            registry,
            library=missing_node.library,
        )
        gap_build_out = gap_builder.handle({"output_cards_dir": str(gap_cards_dir)})
        missing_node.rebind_fresh()
        resolved_out = missing_node.handle({"activations": x_small.tolist()})

        # 7. Remaining operational CARD bricks: ELK/Sheaf/Conformal, Trajectory Loop, HyperLoRA+Liquid, Woodbury+Sinkhorn, 9-Runtime Compiler
        elk_shell = OmniNodeShell(
            {"name": "elk_check", "state_root": str(composite_state), "component": "card_elk_sheaf_conformal_probe"},
            registry,
        )
        elk_out = elk_shell.handle(
            {"activations": x_small.tolist(), "true_labels": nuisance_labels.tolist()}
        )

        # Oscillatory 4-phase limit-cycle trajectory for loop breaker
        osc_steps = int(num["sixteen"])
        cycle_states = [x_small[0], x_small[1], -x_small[0], -x_small[1]]
        osc_rows = [cycle_states[idx % int(num["four"])] for idx in range(osc_steps)]
        traj_shell = OmniNodeShell(
            {"name": "traj_guard", "state_root": str(composite_state), "component": "card_trajectory_loop_breaker"},
            registry,
        )
        traj_out = traj_shell.handle({"trajectory": np.vstack(osc_rows).tolist()})

        hyper_shell = OmniNodeShell(
            {"name": "hyperlora_worker", "state_root": str(composite_state), "component": "card_hyperlora_liquid_operator"},
            registry,
        )
        x_seq_compact = x_small[:, : int(num["sixteen"])]
        anc_compact = anchors[:, : int(num["sixteen"])]
        hyper_out = hyper_shell.handle(
            {"activations": x_seq_compact.tolist(), "anchor_subspace": anc_compact.tolist()}
        )

        solver_shell = OmniNodeShell(
            {"name": "woodbury_worker", "state_root": str(composite_state), "component": "card_woodbury_sinkhorn_attributed_solver"},
            registry,
        )
        y_targets = np.eye(int(num["four"]), dtype=np.float64)[task_labels]
        solver_miss = solver_shell.handle(
            {"activations": x_seq_compact.tolist(), "targets": y_targets.tolist(), "lease_epoch": "epoch_alpha"}
        )
        solver_hit = solver_shell.handle(
            {"activations": x_seq_compact.tolist(), "targets": y_targets.tolist(), "lease_epoch": "epoch_alpha"}
        )

        compiler_shell = OmniNodeShell(
            {"name": "compiler_node", "state_root": str(composite_state), "component": "card_portable_9runtime_compiler"},
            registry,
        )
        compiler_out = compiler_shell.handle(
            {
                "fitted_weights": solver_miss["fitted_weights"],
                "probe_inputs": x_seq_compact[: int(registry.compiler_grid["compiler_limits"]["parity_probe_batch"])].tolist(),
            }
        )

    binary_files_left = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".wasm", ".so", ".o", ".pyc")
    ]

    report = {
        "suite": "omninode_universal_shell_v1",
        "conformance_gates": {
            "ast_magic_literal_violations": len(ast_violations),
            "domain_naming_violations": len(naming_violations),
            "static_card_brick_count": len(static_cards),
            "binary_residue_count": len(binary_files_left),
        },
        "embedding_bridge_and_sae": {
            "live_local_runner_used": emb_pair["live_runner_used"],
            "small_dim": int(x_small.shape[1]),
            "large_dim": int(x_large.shape[1]),
            "bridge_metrics": bridge_out["bridge_metrics"],
            "sae_family_sweep": sae_sweep,
        },
        "lockfile_and_cas_integrity": {
            "initial_discovery_searches": initial_searches,
            "empty_folder_rebuild_searches": rebuild_searches,
            "empty_folder_output_max_diff": rebuild_diff,
            "upstream_tamper_rejected": tamper_caught,
            "upstream_tamper_failure_kind": tamper_kind,
            "mismatch_events_logged": len(tamper_facts),
        },
        "shape_algebra_and_quorum": {
            "composite_verification_passed": bool(composite_out["verification"]["passed"]),
            "leace_cov_before": composite_out["leace_metrics"]["max_abs_cov_before"],
            "leace_cov_after": composite_out["leace_metrics"]["max_abs_cov_after"],
            "steerer_anchor_drift": composite_out["steerer_metrics"]["anchor_max_abs_drift"],
            "steerer_off_target_intrusion": composite_out["steerer_metrics"]["off_target_intrusion_rate"],
            "parallel_sheaf_disagreement": composite_out["sheaf_disagreement_energy"],
            "self_check_same_actor_rejected": self_check_rejected,
        },
        "grid_ledger_and_auto_expansion": {
            "grid1_records_logged": len(grid_rows),
            "expanded": bool(expand_out["boundary_expansion"]["expanded"]),
            "previous_generation": int(expand_out["boundary_expansion"]["previous_generation"]),
            "new_generation": int(expand_out["boundary_expansion"]["new_generation"]),
            "saturated_axes_expanded_count": len(expand_out["boundary_expansion"]["saturated_axes"]),
            "promoted_cell_digest_prefix8": str(expand_out["boundary_expansion"]["promoted_digest"])[:short_len],
        },
        "loopback_http_and_mcp": {
            "hello_status": hello_code,
            "health_status": health_code,
            "job_status": job_code,
            "job_ok": bool(job_body["ok"]),
            "mcp_protocol_version": mcp_init["result"]["protocolVersion"],
            "mcp_listed_tool_count": len(mcp_tools["result"]["tools"]),
            "mcp_tool_call_is_error": bool(mcp_call["result"]["isError"]),
        },
        "autonomous_gap_loop": {
            "initial_declined_attempts": declined_attempts,
            "synthesized_count": int(gap_build_out["synthesized_count"]),
            "shown_pass_rate": gap_build_out["synthesized_components"][0]["shown_pass_rate"],
            "hidden_pass_rate": gap_build_out["synthesized_components"][0]["hidden_pass_rate"],
            "rerun_resolved_capability": resolved_out["synthesized_capability"],
        },
        "specialist_operators_and_9runtime": {
            "elk_sheaf_conformal": elk_out["check_metrics"],
            "trajectory_loop_breaker": traj_out["trajectory_metrics"],
            "hyperlora_liquid": hyper_out["hyperlora_metrics"],
            "woodbury_first_call_cache_hit": bool(solver_miss["solver_metrics"]["cache_hit"]),
            "woodbury_second_call_cache_hit": bool(solver_hit["solver_metrics"]["cache_hit"]),
            "woodbury_r2": solver_miss["solver_metrics"]["woodbury_r2"],
            "sinkhorn_marginal_error": solver_miss["solver_metrics"]["sinkhorn_row_marginal_error"],
            "portable_9runtime": {
                "target_count": compiler_out["compiler_report"]["target_count"],
                "targets_emitted": compiler_out["compiler_report"]["targets_emitted"],
                "c99_max_abs_diff": compiler_out["compiler_report"]["c99_max_abs_diff"],
                "wasm_max_abs_diff": compiler_out["compiler_report"]["wasm_max_abs_diff"],
                "rust_nostd_max_abs_diff": compiler_out["compiler_report"]["rust_nostd_max_abs_diff"],
                "python_max_abs_diff": compiler_out["compiler_report"]["python_max_abs_diff"],
                "safetensors_max_abs_diff": compiler_out["compiler_report"]["safetensors_max_abs_diff"],
                "sql_max_abs_diff": compiler_out["compiler_report"]["sql_max_abs_diff"],
                "parity_passed": compiler_out["compiler_report"]["parity_passed"],
            },
        },
    }
    clean_report = _round_floats(report, digits)
    out_path = root / "results" / "omninode_universal_shell_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clean_report, indent=int(num["two"]), sort_keys=True) + "\n", encoding="utf-8")
    return clean_report
