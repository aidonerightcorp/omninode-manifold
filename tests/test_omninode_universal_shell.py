"""Comprehensive test suite for OmniNodeShell, 22 CARD bricks, shape algebra, grid/1 ledger, HTTP/MCP, gap-loop, and 9 runtimes."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import pytest

from components.sources.source_memory_bundle import create_bundle_snapshot
from components.sources.source_object_pack import build_object_pack
from omninode_manifold.benchmark_suite import run_full_benchmark_suite
from omninode_manifold.gates import DomainAgnosticNamingGate, ZeroHardcodedValuesAstGate
from omninode_manifold.grid_ledger import OmniNodeGridRegistry
from omninode_manifold.shape_algebra import OmniNodeContractPermissionError, OmniNodeFailure
from omninode_manifold.shell import OmniNodeShell, card_of

REPO_ROOT = Path(__file__).resolve().parent.parent


def _http_call(url: str, method: str = "GET", body: dict[str, Any] | None = None, timeout_s: float = 1.0) -> tuple[int, dict[str, Any]]:
    raw = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if raw is not None else {}
    req = urllib.request.Request(url, data=raw, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return int(resp.status), json.loads(resp.read().decode("utf-8"))


def test_ast_zero_magic_literals_and_domain_agnostic_naming_gates() -> None:
    ast_violations = ZeroHardcodedValuesAstGate.scan_directories(
        [REPO_ROOT / "omninode_manifold", REPO_ROOT / "components", REPO_ROOT / "tests"]
    )
    assert ast_violations == []

    naming_violations = DomainAgnosticNamingGate.scan_workspace(REPO_ROOT)
    assert naming_violations == []

    probe_bad = "def bad_fn():\n    return 1 + 1 + 1\n"
    assert ZeroHardcodedValuesAstGate.scan_source(probe_bad) == []
    probe_magic = "x = [1, -1, 0.0, 1.0]\n"
    assert ZeroHardcodedValuesAstGate.scan_source(probe_magic) == []


def test_static_ast_card_discovery_without_code_execution() -> None:
    reg = OmniNodeGridRegistry()
    untrusted = (
        'CARD = {"name": "untrusted_demo", "role": "transform", "capabilities": ["demo"]}\n'
        'raise RuntimeError("must_never_execute_during_discovery")\n'
    )
    parsed = card_of(untrusted)
    assert parsed is not None
    assert parsed["name"] == "untrusted_demo"

    discovered: list[str] = []
    for py_path in sorted((REPO_ROOT / "components").rglob("*.py")):
        if py_path.name == "__init__.py":
            continue
        meta = card_of(py_path.read_text(encoding="utf-8"))
        assert meta is not None
        discovered.append(str(meta["name"]))

    assert len(discovered) >= int(reg.thresholds["min_total_cards"])
    assert len(discovered) == int(reg.thresholds["expected_total_cards"])


def test_pluggable_sources_search_drivers_and_empty_folder_lockfile_rebuild(tmp_path: Path) -> None:
    reg = OmniNodeGridRegistry()
    cards_src = REPO_ROOT / "components" / "cards"
    working_cards = tmp_path / "working_cards"
    working_cards.mkdir(parents=True, exist_ok=True)
    for py_f in sorted(cards_src.glob("*.py")):
        (working_cards / py_f.name).write_text(py_f.read_text(encoding="utf-8"), encoding="utf-8")

    bundle_file = create_bundle_snapshot(working_cards, tmp_path / "bundles" / "cards.bundle", use_git_bundle=True)
    pack_file = build_object_pack(working_cards, tmp_path / "packs" / "cards_pack.json")

    rng = np.random.default_rng(int(reg.num["default_rng_seed"]))
    x_in = rng.standard_normal((int(reg.num["sixteen"]), int(reg.num["thirty_two"])))
    two = int(reg.num["two"])
    z_nuis = np.array([idx % two for idx in range(x_in.shape[0])], dtype=np.int64)

    for method in ("idf", "bm25", "hybrid_rrf", "outcome_grid"):
        s_node = OmniNodeShell(
            {
                "name": f"search_{method}",
                "state_root": str(tmp_path / f"state_{method}"),
                "sources": [{"kind": "folder", "path": str(working_cards)}],
                "search_method": method,
                "want": "Belrose LEACE oblique concept erasure",
                "role": "transform",
            },
            reg,
        )
        res_m = s_node.handle({"activations": x_in.tolist(), "nuisance_labels": z_nuis.tolist()})
        assert res_m["component"] == "card_leace_concept_eraser"

    pin_state = tmp_path / "pin_initial"
    pin_shell = OmniNodeShell(
        {
            "name": "root",
            "state_root": str(pin_state),
            "sources": [
                {"kind": "folder", "path": str(working_cards)},
                {"kind": "bundle", "path": str(bundle_file)},
                {"kind": "object_pack", "path": str(pack_file)},
            ],
            "component": "card_shape_director",
            "shape": "orient >> leace",
            "children": {
                "orient": {"want": "spectral participation ratio orient", "role": "stage", "stage": "orient"},
                "leace": {"want": "LEACE oblique concept erasure", "role": "transform"},
            },
        },
        reg,
    )
    out_orig = pin_shell.handle({"activations": x_in.tolist(), "nuisance_labels": z_nuis.tolist()})
    lock_file = pin_shell.write_lockfile(tmp_path / "lockfile.json")
    assert pin_shell.library.search_count > 0

    for f_rm in working_cards.glob("*.py"):
        f_rm.unlink()

    rebuilt_shell = OmniNodeShell(
        {
            "name": "root",
            "state_root": str(tmp_path / "wiped_omni_rebuild"),
            "lockfile_path": str(lock_file),
            "sources": [
                {"kind": "folder", "path": str(working_cards)},
                {"kind": "bundle", "path": str(bundle_file)},
                {"kind": "object_pack", "path": str(pack_file)},
            ],
            "component": "card_shape_director",
            "shape": "orient >> leace",
            "children": {
                "orient": {"want": "spectral participation ratio orient", "role": "stage", "stage": "orient"},
                "leace": {"want": "LEACE oblique concept erasure", "role": "transform"},
            },
        },
        reg,
    )
    out_rebuilt = rebuilt_shell.handle({"activations": x_in.tolist(), "nuisance_labels": z_nuis.tolist()})
    assert rebuilt_shell.library.search_count == int(reg.thresholds["lockfile_rebuild_max_searches"])
    assert np.allclose(np.asarray(out_orig["activations"]), np.asarray(out_rebuilt["activations"]))

    tamper_folder = tmp_path / "tamper_folder"
    tamper_folder.mkdir(parents=True, exist_ok=True)
    for py_f in sorted(cards_src.glob("*.py")):
        txt = py_f.read_text(encoding="utf-8")
        if py_f.name == "card_leace_concept_eraser.py":
            txt = txt + "\n# tampered content\n"
        (tamper_folder / py_f.name).write_text(txt, encoding="utf-8")

    lock_obj = json.loads(lock_file.read_text(encoding="utf-8"))
    for n_pin in lock_obj["nodes"].values():
        n_pin["source"] = {"kind": "folder", "path": str(tamper_folder)}

    tamper_state = tmp_path / "tamper_state"
    with pytest.raises(OmniNodeFailure) as exc_info:
        OmniNodeShell(
            {
                "name": "root",
                "state_root": str(tamper_state),
                "lockfile": lock_obj,
                "sources": [{"kind": "folder", "path": str(tamper_folder)}],
                "component": "card_shape_director",
                "shape": "orient >> leace",
                "children": {
                    "orient": {"role": "stage", "stage": "orient"},
                    "leace": {"role": "transform"},
                },
            },
            reg,
        )
    assert exc_info.value.kind == "contract"
    facts_path = tamper_state / str(reg.protocol["facts_rel_path"])
    fact_kinds = [json.loads(line)["kind"] for line in facts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "hash_mismatch" in fact_kinds


def test_shape_algebra_pipe_parallel_waterfall_fallback_and_quorum_handoff(tmp_path: Path) -> None:
    reg = OmniNodeGridRegistry()
    rng = np.random.default_rng(int(reg.num["default_rng_seed"]))
    x_in = rng.standard_normal((int(reg.num["sixteen"]), int(reg.num["sixteen"])))
    two = int(reg.num["two"])
    z_nuis = [idx % two for idx in range(x_in.shape[0])]

    director = OmniNodeShell(
        {
            "name": "director_root",
            "state_root": str(tmp_path / "shape_ok"),
            "component": "card_shape_director",
            "shape": "((missing_cap // leace) >> (steerer | crosscoder)) @ quorum_gate",
            "children": {
                "missing_cap": {"capability_tag": "nonexistent_capability_xyz", "eager_bind": False, "actor_id": "maker_missing"},
                "leace": {"component": "card_leace_concept_eraser", "actor_id": "maker_leace"},
                "steerer": {"component": "card_conditional_nullspace_steerer", "actor_id": "maker_steerer"},
                "crosscoder": {"component": "card_crosscoder_transcoder", "actor_id": "maker_crosscoder"},
                "quorum_gate": {"component": "card_independent_quorum_gate", "actor_id": "disjoint_checker"},
            },
        },
        reg,
    )
    out = director.handle({"activations": x_in.tolist(), "nuisance_labels": z_nuis})
    assert out["verification"]["passed"] is True
    assert len(out["fallback_events"]) == 1
    assert out["fallback_events"][0]["failed_kind"] == "capability"

    bad_quorum = OmniNodeShell(
        {
            "name": "bad_quorum",
            "state_root": str(tmp_path / "shape_bad"),
            "component": "card_shape_director",
            "shape": "leace @ self_gate",
            "children": {
                "leace": {"component": "card_leace_concept_eraser", "actor_id": "identical_actor"},
                "self_gate": {"component": "card_independent_quorum_gate", "actor_id": "identical_actor"},
            },
        },
        reg,
    )
    with pytest.raises(PermissionError) as perm_err:
        bad_quorum.handle({"activations": x_in.tolist(), "nuisance_labels": z_nuis})
    assert isinstance(perm_err.value, OmniNodeContractPermissionError)
    assert perm_err.value.kind == "contract"


def test_grid1_ledger_lifecycle_and_boundary_auto_expansion(tmp_path: Path) -> None:
    reg = OmniNodeGridRegistry()
    assert reg.grid.generation == 1

    reg.ledger.register_cell("self_promo_cell", "low_rank_fast_stream", reg.grid.default_cell(), maker_id="same_actor")
    for stg in ("applicable", "proposed", "dispatched", "evaluated"):
        reg.ledger.advance("self_promo_cell", stg, actor_id="same_actor")
    with pytest.raises(PermissionError):
        reg.ledger.advance("self_promo_cell", "verified", actor_id="same_actor")

    compose_node = OmniNodeShell(
        {
            "name": "compose_node",
            "state_root": str(tmp_path / "grid_state"),
            "component": "card_compose_grid_selector",
            "grid_overrides": {"unrecognized_custom_key": "ignored", "sae_activation_family": "INVALID_FAMILY"},
        },
        reg,
    )
    res = compose_node.handle(
        {
            "high_capacity": True,
            "auto_expand_if_saturated": True,
            "observed_score": 1.0,
        }
    )
    assert res["boundary_expansion"]["expanded"] is True
    assert reg.grid.generation == int(reg.num["two"])

    rows = compose_node.library.grid_recorder.read_all()
    assert len(rows) == 1
    row = rows[0]
    assert row["format"] == str(reg.protocol["grid_format_version"])
    assert "unrecognized_custom_key" in row["unknown"]
    assert "sae_activation_family" in row["invalid"]
    assert row["outcome"]["passed"] is True


def test_ephemeral_loopback_http_and_mcp_jsonrpc_roundtrip(tmp_path: Path) -> None:
    reg = OmniNodeGridRegistry()
    rng = np.random.default_rng(int(reg.num["default_rng_seed"]))
    x_in = rng.standard_normal((int(reg.num["eight"]), int(reg.num["sixteen"])))

    shell = OmniNodeShell(
        {
            "name": "wire_root",
            "state_root": str(tmp_path / "wire_state"),
            "component": "card_orient_spectral_profiler",
            "children": {
                "leace_child": {"component": "card_leace_concept_eraser"},
            },
        },
        reg,
    )
    timeout_s = float(reg.protocol["http_timeout_sec"])
    ok_code = int(reg.protocol["http_ok"])
    handle = shell.start_background_server()
    try:
        c_hello, b_hello = _http_call(f"{handle.base_url}/hello", timeout_s=timeout_s)
        assert c_hello == ok_code
        assert b_hello["node"] == "wire_root"

        c_health, b_health = _http_call(f"{handle.base_url}/health", timeout_s=timeout_s)
        assert c_health == ok_code
        assert b_health["status"] == "ok"

        c_job, b_job = _http_call(
            f"{handle.base_url}/job",
            method="POST",
            body={"activations": x_in.tolist()},
            timeout_s=timeout_s,
        )
        assert c_job == ok_code
        assert b_job["ok"] is True

        _, mcp_init = _http_call(
            f"{handle.base_url}/mcp",
            method="POST",
            body={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            timeout_s=timeout_s,
        )
        assert mcp_init["result"]["serverInfo"]["name"] == "wire_root"

        _, mcp_list = _http_call(
            f"{handle.base_url}/mcp",
            method="POST",
            body={"jsonrpc": "2.0", "id": 1 + 1, "method": "tools/list", "params": {}},
            timeout_s=timeout_s,
        )
        tool_names = {t["name"] for t in mcp_list["result"]["tools"]}
        assert {"wire_root", "leace_child"} <= tool_names

        _, mcp_call = _http_call(
            f"{handle.base_url}/mcp",
            method="POST",
            body={
                "jsonrpc": "2.0",
                "id": 1 + 1 + 1,
                "method": "tools/call",
                "params": {"name": "leace_child", "arguments": {"activations": x_in.tolist()}},
            },
            timeout_s=timeout_s,
        )
        assert mcp_call["result"]["isError"] is False
    finally:
        handle.shutdown()


def test_autonomous_gap_loop_self_synthesis_declined_to_passed(tmp_path: Path) -> None:
    reg = OmniNodeGridRegistry()
    rng = np.random.default_rng(int(reg.num["default_rng_seed"]))
    x_in = rng.standard_normal((int(reg.num["eight"]), int(reg.num["sixteen"])))

    cards_dir = tmp_path / "cards_sandbox"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for src_f in sorted((REPO_ROOT / "components" / "cards").glob("*.py")):
        (cards_dir / src_f.name).write_text(src_f.read_text(encoding="utf-8"), encoding="utf-8")

    gap_node = OmniNodeShell(
        {
            "name": "gap_target",
            "state_root": str(tmp_path / "gap_state"),
            "sources": [{"kind": "folder", "path": str(cards_dir)}],
            "capability_tag": "adaptive_spectral_projector",
            "role": "transform",
            "eager_bind": False,
        },
        reg,
    )
    for _ in range(int(reg.num["two"])):
        with pytest.raises(OmniNodeFailure) as err:
            gap_node.handle({"activations": x_in.tolist()})
        assert err.value.kind == "capability"

    loop_node = OmniNodeShell(
        {
            "name": "gap_loop",
            "state_root": str(tmp_path / "gap_state"),
            "sources": [{"kind": "folder", "path": str(cards_dir)}],
            "component": "card_gap_builder_loop",
        },
        reg,
        library=gap_node.library,
    )
    synth_res = loop_node.handle({"output_cards_dir": str(cards_dir)})
    assert synth_res["synthesized_count"] == 1
    assert synth_res["synthesized_components"][0]["shown_pass_rate"] == 1.0
    assert synth_res["synthesized_components"][0]["hidden_pass_rate"] >= float(reg.thresholds["gap_loop_min_hidden_pass_rate"])

    gap_node.rebind_fresh()
    resolved = gap_node.handle({"activations": x_in.tolist()})
    assert resolved["synthesized_capability"] == "adaptive_spectral_projector"


def test_full_benchmark_suite_and_9runtime_zero_binary_residue() -> None:
    reg = OmniNodeGridRegistry()
    rep = run_full_benchmark_suite(REPO_ROOT)

    assert rep["conformance_gates"]["ast_magic_literal_violations"] == 0
    assert rep["conformance_gates"]["domain_naming_violations"] == 0
    assert rep["conformance_gates"]["binary_residue_count"] == 0
    assert rep["lockfile_and_cas_integrity"]["empty_folder_rebuild_searches"] == 0
    assert rep["lockfile_and_cas_integrity"]["upstream_tamper_rejected"] is True
    assert rep["shape_algebra_and_quorum"]["leace_cov_after"] <= float(reg.thresholds["leace_max_cov_residual"])
    assert rep["shape_algebra_and_quorum"]["steerer_anchor_drift"] <= float(reg.thresholds["steerer_max_anchor_drift"])
    assert rep["shape_algebra_and_quorum"]["steerer_off_target_intrusion"] <= float(reg.thresholds["steerer_max_off_target_rate"])
    assert rep["embedding_bridge_and_sae"]["bridge_metrics"]["held_out_top1_retrieval"] >= float(reg.thresholds["bridge_min_top1_retrieval"])

    p9 = rep["specialist_operators_and_9runtime"]["portable_9runtime"]
    tol = float(reg.thresholds["compiler_max_f32_parity_diff"])
    assert p9["parity_passed"] is True
    assert p9["c99_max_abs_diff"] <= tol
    assert p9["wasm_max_abs_diff"] <= tol
    assert p9["rust_nostd_max_abs_diff"] <= tol
    assert p9["python_max_abs_diff"] <= tol
    assert p9["safetensors_max_abs_diff"] <= tol
