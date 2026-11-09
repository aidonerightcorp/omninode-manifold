"""Universal parameterized recursive OmniNodeShell with static AST CARD discovery, .omni/cas, lockfile, HTTP, and MCP."""

from __future__ import annotations

import ast
import copy
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from omninode_manifold.grid_ledger import (
    FactsLedger,
    Grid1Recorder,
    OmniNodeGridRegistry,
    content_digest_hex,
)
from omninode_manifold.shape_algebra import (
    OmniNodeContractPermissionError,
    OmniNodeFailure,
    run_shape_expression,
)


def card_of(source_text: str) -> dict[str, Any] | None:
    """Extract module-level CARD dictionary via static AST parsing without executing code."""
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "CARD":
                    try:
                        val = ast.literal_eval(stmt.value)
                    except (ValueError, SyntaxError):
                        return None
                    return dict(val) if isinstance(val, dict) else None
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == "CARD" and stmt.value is not None:
                try:
                    val = ast.literal_eval(stmt.value)
                except (ValueError, SyntaxError):
                    return None
                return dict(val) if isinstance(val, dict) else None
    return None


def _load_verified_module(
    source_text: str,
    *,
    cas_dir: Path,
    short_len: int,
    expected_digest: str | None = None,
    facts: FactsLedger | None = None,
    node_path: str = "boot",
    source_desc: dict[str, Any] | None = None,
) -> tuple[types.ModuleType, dict[str, Any], str]:
    """Verify source digest against optional pin, persist into .omni/cas/<digest>.py, and load module."""
    actual_digest = content_digest_hex(source_text)
    if expected_digest is not None and actual_digest != expected_digest:
        if facts is not None:
            facts.emit(
                "hash_mismatch",
                node=node_path,
                expected=expected_digest,
                actual=actual_digest,
                source=source_desc or {},
            )
        raise OmniNodeFailure(
            f"Lockfile digest mismatch at {node_path}: expected {expected_digest[:short_len]}, got {actual_digest[:short_len]}",
            kind="contract",
            details={"node": node_path, "expected": expected_digest, "actual": actual_digest},
        )

    card = card_of(source_text)
    if card is None:
        raise OmniNodeFailure(f"Component at {node_path} missing static CARD literal", kind="contract")

    cas_dir.mkdir(parents=True, exist_ok=True)
    cas_file = cas_dir / f"{actual_digest}.py"
    if not cas_file.exists():
        cas_file.write_text(source_text, encoding="utf-8")

    mod_name = f"omninode_cas_{actual_digest[:short_len]}"
    module = types.ModuleType(mod_name)
    module.__file__ = str(cas_file)
    code_obj = compile(source_text, str(cas_file), "exec")
    exec(code_obj, module.__dict__)  # noqa: S102
    return module, card, actual_digest


class ComponentLibrary:
    """Shared content-addressed component library, boot source/search loader, and lockfile manager."""

    def __init__(
        self,
        *,
        state_root: Path,
        registry: OmniNodeGridRegistry,
        sources_cfg: list[dict[str, Any]],
        search_method: str = "hybrid_rrf",
        boot_dirs: list[Path] | None = None,
        lockfile_data: dict[str, Any] | None = None,
    ) -> None:
        self.state_root = state_root
        self.registry = registry
        self.proto = registry.protocol
        self.short_len = int(self.proto["digest_short_len"])
        self.cas_dir = state_root / str(self.proto["cas_rel_dir"])
        self.facts = FactsLedger(state_root / str(self.proto["facts_rel_path"]))
        self.grid_recorder = Grid1Recorder(
            state_root / str(self.proto["grid_records_rel_path"]),
            format_version=str(self.proto["grid_format_version"]),
        )
        self.sources_cfg = [ copy.deepcopy(s) for s in sources_cfg ]
        self.search_method = str(search_method)
        self.lockfile_data = copy.deepcopy(lockfile_data) if lockfile_data else None
        self.search_count = 0
        self.pull_count = 0
        self.node_pins: dict[str, dict[str, Any]] = {}
        self.boot_pins: dict[str, dict[str, Any]] = {}
        self.runtime_cache: dict[str, Any] = {}

        default_comp_root = Path(__file__).resolve().parent.parent / "components"
        self.boot_dirs = boot_dirs if boot_dirs is not None else [
            default_comp_root / "sources",
            default_comp_root / "search",
        ]
        self._source_drivers: dict[str, tuple[types.ModuleType, dict[str, Any], str]] = {}
        self._search_drivers: dict[str, tuple[types.ModuleType, dict[str, Any], str]] = {}
        self._loaded_modules: dict[str, tuple[types.ModuleType, dict[str, Any]]] = {}
        self._bootstrap_drivers()

    def _bootstrap_drivers(self) -> None:
        for b_dir in self.boot_dirs:
            if not b_dir.exists():
                continue
            for py_file in sorted(b_dir.glob("*.py")):
                text = py_file.read_text(encoding="utf-8")
                card = card_of(text)
                if card is None:
                    continue
                role = str(card.get("role", ""))
                if role == "source":
                    s_kind = str(card.get("source_kind", ""))
                    mod, c_meta, dig = _load_verified_module(
                        text,
                        cas_dir=self.cas_dir,
                        short_len=self.short_len,
                        facts=self.facts,
                        node_path=f"boot/source/{s_kind}",
                    )
                    self._source_drivers[s_kind] = (mod, c_meta, dig)
                    self.boot_pins[f"source:{s_kind}"] = {
                        "component": str(c_meta.get("name", py_file.stem)),
                        "sha256": dig,
                        "role": "source",
                        "source_kind": s_kind,
                    }
                elif role == "search":
                    m_name = str(card.get("search_method", ""))
                    mod, c_meta, dig = _load_verified_module(
                        text,
                        cas_dir=self.cas_dir,
                        short_len=self.short_len,
                        facts=self.facts,
                        node_path=f"boot/search/{m_name}",
                    )
                    self._search_drivers[m_name] = (mod, c_meta, dig)
                    self.boot_pins[f"search:{m_name}"] = {
                        "component": str(c_meta.get("name", py_file.stem)),
                        "sha256": dig,
                        "role": "search",
                        "search_method": m_name,
                    }

    def list_all_catalog_entries(self) -> list[dict[str, Any]]:
        """Enumerate static CARD metadata across all configured sources without executing candidate code."""
        catalog: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for src_spec in self.sources_cfg:
            s_kind = str(src_spec.get("kind", "folder"))
            driver = self._source_drivers.get(s_kind)
            if driver is None:
                self.facts.emit("source_down", source=src_spec, reason=f"unknown_driver:{s_kind}")
                continue
            mod = driver[0]
            try:
                raw_items = mod.list_entries(src_spec, self.registry)
            except Exception as exc:  # noqa: BLE001
                self.facts.emit("source_down", source=src_spec, reason=str(exc))
                continue
            for item in raw_items:
                text = str(item["source_text"])
                card = card_of(text)
                if card is None:
                    continue
                comp_name = str(card.get("name", item.get("name", "")))
                if comp_name in seen_names:
                    continue
                seen_names.add(comp_name)
                catalog.append(
                    {
                        "name": comp_name,
                        "card": card,
                        "source_text": text,
                        "sha256": content_digest_hex(text),
                        "source_spec": copy.deepcopy(src_spec),
                        "locator": str(item.get("locator", comp_name)),
                    }
                )
        return catalog

    def _fetch_text_from_source(self, src_spec: dict[str, Any], locator: str, comp_name: str) -> str | None:
        s_kind = str(src_spec.get("kind", "folder"))
        driver = self._source_drivers.get(s_kind)
        if driver is None:
            return None
        mod = driver[0]
        try:
            return mod.read_entry(src_spec, locator, comp_name, self.registry)
        except Exception as exc:  # noqa: BLE001
            self.facts.emit("source_down", source=src_spec, reason=str(exc))
            return None

    def resolve_for_node(
        self,
        *,
        node_path: str,
        node_cfg: dict[str, Any],
        grid_cell: dict[str, Any],
        grid_cell_digest: str,
        regime_id: str,
    ) -> tuple[types.ModuleType, dict[str, Any], str]:
        """Resolve, verify, and load the CARD module for node_path via lockfile pin or search."""
        if self.lockfile_data is not None and node_path in self.lockfile_data.get("nodes", {}):
            pin = self.lockfile_data["nodes"][node_path]
            expected_sha = str(pin["sha256"])
            comp_name = str(pin["component"])
            locator = str(pin.get("locator", comp_name))
            pin_source = dict(pin.get("source", {}))

            source_text: str | None = None
            tried_upstream = False
            for src_candidate in ([pin_source] if pin_source else []) + self.sources_cfg:
                if not src_candidate:
                    continue
                fetched = self._fetch_text_from_source(src_candidate, locator, comp_name)
                if fetched is not None:
                    source_text = fetched
                    tried_upstream = True
                    self.pull_count += 1
                    self.facts.emit("pull", node=node_path, component=comp_name, source=src_candidate)
                    break

            if source_text is None:
                cas_path = self.cas_dir / f"{expected_sha}.py"
                if cas_path.exists():
                    source_text = cas_path.read_text(encoding="utf-8")

            if source_text is None:
                raise OmniNodeFailure(
                    f"Pinned component {comp_name} unavailable from sources or CAS at {node_path}",
                    kind="transient" if not tried_upstream else "contract",
                )

            mod, card, dig = _load_verified_module(
                source_text,
                cas_dir=self.cas_dir,
                short_len=self.short_len,
                expected_digest=expected_sha,
                facts=self.facts,
                node_path=node_path,
                source_desc=pin_source,
            )
            self._loaded_modules[dig] = (mod, card)
            self.node_pins[node_path] = {
                "component": comp_name,
                "sha256": dig,
                "locator": locator,
                "source": pin_source,
                "grid_cell_sha256": grid_cell_digest,
                "regime_id": regime_id,
            }
            self.facts.emit("pinned", node=node_path, component=comp_name, sha256=dig[: self.short_len])
            return mod, card, dig

        catalog = self.list_all_catalog_entries()
        explicit_comp = node_cfg.get("component")
        required_cap = node_cfg.get("capability_tag")
        role_filter = node_cfg.get("role")
        stage_filter = node_cfg.get("stage")
        want_query = str(node_cfg.get("want", explicit_comp or required_cap or role_filter or "universal operator"))

        filtered = catalog
        if explicit_comp:
            filtered = [entry for entry in filtered if entry["name"] == str(explicit_comp)]
        else:
            if role_filter:
                filtered = [entry for entry in filtered if str(entry["card"].get("role", "")) == str(role_filter)]
            if stage_filter:
                filtered = [entry for entry in filtered if str(entry["card"].get("stage", "")) == str(stage_filter)]
            if required_cap:
                filtered = [
                    entry
                    for entry in filtered
                    if str(required_cap) in list(entry["card"].get("capabilities", []))
                ]

        if not filtered:
            self.facts.emit(
                "gap_observed",
                node=node_path,
                want=want_query,
                capability_tag=required_cap,
                role=role_filter,
            )
            raise OmniNodeFailure(
                f"No CARD component matched capability={required_cap!r} want={want_query!r} at {node_path}",
                kind="capability",
                details={"node": node_path, "capability_tag": required_cap, "want": want_query},
            )

        if explicit_comp and len(filtered) == 1:
            chosen = filtered[0]
        else:
            s_method = str(node_cfg.get("search_method", self.search_method))
            s_driver = self._search_drivers.get(s_method) or next(iter(self._search_drivers.values()))
            s_mod = s_driver[0]
            top_k = int(grid_cell.get("shell_search_top_k", 1))
            ranked = s_mod.rank_cards(
                query=want_query,
                candidates=filtered,
                grid_cell=grid_cell,
                registry=self.registry,
                facts_rows=self.facts.read_all(),
                top_k=top_k,
            )
            self.search_count += 1
            self.facts.emit(
                "search",
                node=node_path,
                method=s_method,
                query=want_query,
                candidates=len(filtered),
                top_match=ranked[0]["name"] if ranked else None,
            )
            if not ranked:
                self.facts.emit("gap_observed", node=node_path, want=want_query, capability_tag=required_cap)
                raise OmniNodeFailure(f"Search returned zero candidates at {node_path}", kind="capability")
            chosen = ranked[0]

        mod, card, dig = _load_verified_module(
            str(chosen["source_text"]),
            cas_dir=self.cas_dir,
            short_len=self.short_len,
            facts=self.facts,
            node_path=node_path,
            source_desc=chosen["source_spec"],
        )
        self._loaded_modules[dig] = (mod, card)
        self.node_pins[node_path] = {
            "component": str(chosen["name"]),
            "sha256": dig,
            "locator": str(chosen["locator"]),
            "source": copy.deepcopy(chosen["source_spec"]),
            "grid_cell_sha256": grid_cell_digest,
            "regime_id": regime_id,
        }
        self.facts.emit("chosen", node=node_path, component=chosen["name"], sha256=dig[: self.short_len])
        return mod, card, dig

    def export_lockfile(self) -> dict[str, Any]:
        return {
            "schema": "omninode.lockfile.v1",
            "grid_id": self.registry.grid.grid_id,
            "grid_generation": self.registry.grid.generation,
            "search_method": self.search_method,
            "boot": copy.deepcopy(self.boot_pins),
            "nodes": copy.deepcopy(self.node_pins),
        }


@dataclass
class LoopbackServerHandle:
    """Ephemeral loopback HTTP + MCP server handle with clean deterministic shutdown."""

    host: str
    port: int
    base_url: str
    server: socketserver.TCPServer
    thread: threading.Thread

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class OmniNodeShell:
    """Universal recursive OmniNode shell: every node is an OmniNodeShell parameterized by CARD + NodeGrid."""

    def __init__(
        self,
        cfg: dict[str, Any],
        grid_registry: OmniNodeGridRegistry,
        parent: OmniNodeShell | None = None,
        depth: int = 0,
        library: ComponentLibrary | None = None,
    ) -> None:
        self.cfg = copy.deepcopy(cfg)
        self.registry = grid_registry
        self.parent = parent
        self.depth = int(depth)
        self.name = str(cfg.get("name", "root" if parent is None else f"child_{depth}"))
        self.actor_id = str(cfg.get("actor_id", f"actor_{self.name}"))

        overrides = dict(cfg.get("grid_overrides", {}))
        requested_regime = cfg.get("regime_id")
        if requested_regime and str(requested_regime) in grid_registry.regime_cells:
            self.regime_id = str(requested_regime)
            merged_c = dict(grid_registry.regime_cells[self.regime_id])
            merged_c.update(overrides)
            self.grid_cell, self.unknown_keys, self.invalid_keys = grid_registry.grid.normalize_and_partition(merged_c)
            self.grid_cell_digest = grid_registry.grid.cell_digest(self.grid_cell)
        else:
            self.regime_id, self.grid_cell, self.grid_cell_digest = grid_registry.resolve_regime_cell(
                overrides=overrides
            )
            _, self.unknown_keys, self.invalid_keys = grid_registry.grid.normalize_and_partition(overrides)

        max_depth = int(self.grid_cell["shell_max_recursion_depth"])
        if self.depth > max_depth:
            raise OmniNodeFailure(
                f"Recursion depth {self.depth} exceeds grid shell_max_recursion_depth={max_depth}",
                kind="budget",
            )

        if library is not None:
            self.library = library
        elif parent is not None:
            self.library = parent.library
        else:
            state_root = Path(cfg.get("state_root", Path.cwd()))
            default_cards_dir = Path(__file__).resolve().parent.parent / "components" / "cards"
            sources_cfg = list(
                cfg.get(
                    "sources",
                    [{"kind": "folder", "path": str(default_cards_dir)}],
                )
            )
            lockfile_data = cfg.get("lockfile")
            if lockfile_data is None and cfg.get("lockfile_path"):
                lf_p = Path(str(cfg["lockfile_path"]))
                if lf_p.exists():
                    lockfile_data = json.loads(lf_p.read_text(encoding="utf-8"))
            self.library = ComponentLibrary(
                state_root=state_root,
                registry=grid_registry,
                sources_cfg=sources_cfg,
                search_method=str(cfg.get("search_method", "hybrid_rrf")),
                lockfile_data=lockfile_data,
            )

        self.children: dict[str, OmniNodeShell] = {}
        for child_name, child_cfg in dict(cfg.get("children", {})).items():
            c_spec = copy.deepcopy(child_cfg)
            c_spec.setdefault("name", child_name)
            self.children[child_name] = OmniNodeShell(
                c_spec,
                grid_registry,
                parent=self,
                depth=self.depth + 1,
                library=self.library,
            )

        self._module: types.ModuleType | None = None
        self._card: dict[str, Any] | None = None
        self._component_digest: str | None = None
        if bool(cfg.get("eager_bind", True)):
            self.bind_component()

    def path(self) -> str:
        if self.parent is None:
            return self.name
        return f"{self.parent.path()}/{self.name}"

    def bind_component(self) -> tuple[types.ModuleType, dict[str, Any], str]:
        if self._module is not None and self._card is not None and self._component_digest is not None:
            return self._module, self._card, self._component_digest
        mod, card, dig = self.library.resolve_for_node(
            node_path=self.path(),
            node_cfg=self.cfg,
            grid_cell=self.grid_cell,
            grid_cell_digest=self.grid_cell_digest,
            regime_id=self.regime_id,
        )
        self._module = mod
        self._card = card
        self._component_digest = dig
        return mod, card, dig

    def rebind_fresh(self) -> tuple[types.ModuleType, dict[str, Any], str]:
        """Clear cached component binding so newly synthesized CAS/source bricks can be discovered."""
        self._module = None
        self._card = None
        self._component_digest = None
        return self.bind_component()

    def card_meta(self) -> dict[str, Any]:
        _, card, dig = self.bind_component()
        short_len = int(self.registry.protocol["digest_short_len"])
        return {
            **copy.deepcopy(card),
            "sha256_prefix": dig[:short_len],
            "node_path": self.path(),
            "regime_id": self.regime_id,
        }

    def resolve_child(self, child_name: str) -> OmniNodeShell:
        if child_name not in self.children:
            raise OmniNodeFailure(f"Unknown child OmniNode {child_name!r} under {self.path()}", kind="bad_input")
        return self.children[child_name]

    def execute_shape(self, shape_expr: str, job: dict[str, Any]) -> dict[str, Any]:
        eps = float(self.registry.num["eps"])
        return run_shape_expression(
            shape_expr,
            job,
            resolve_child=self.resolve_child,
            grid_cell=self.grid_cell,
            rerank_kinds=self.registry.rerank_failure_kinds,
            halt_kinds=self.registry.halt_failure_kinds,
            eps=eps,
        )

    def handle(self, job: dict[str, Any]) -> dict[str, Any]:
        """Execute job through the bound CARD brick and record grid/1 + facts.jsonl observations."""
        t_start = time.perf_counter()
        ms_scale = float(self.registry.num["thousand_ms"])
        run_id = str(job.get("run_id") or f"run_{uuid.uuid4().hex[:int(self.registry.protocol['digest_short_len'])]}")
        raw_cfg_for_grid = {**self.grid_cell, **dict(self.cfg.get("grid_overrides", {})), **dict(job.get("grid_overrides", {}))}

        try:
            mod, card, dig = self.bind_component()
            comp_name = str(card.get("name", self.name))
            out = mod.execute(job, self)
            if not isinstance(out, dict):
                out = {"value": out}
            latency_ms = (time.perf_counter() - t_start) * ms_scale
            score = float(out.get("quality_score", 1.0))

            out.setdefault("actor_id", self.actor_id)
            out["node_path"] = self.path()
            out["component"] = comp_name
            out["component_digest_prefix"] = dig[: int(self.registry.protocol["digest_short_len"])]
            out["grid_cell_digest"] = self.grid_cell_digest
            out["regime_id"] = self.regime_id
            out["latency_ms"] = latency_ms

            self.library.grid_recorder.record_run(
                run_id=run_id,
                kind=comp_name,
                source=self.path(),
                grid=self.registry.grid,
                raw_config=raw_cfg_for_grid,
                passed=True,
                score=score,
                latency_ms=latency_ms,
                failure=None,
            )
            self.library.facts.emit(
                "observed",
                node=self.path(),
                component=comp_name,
                passed=True,
                score=score,
                latency_ms=latency_ms,
                cell=self.grid_cell_digest[: int(self.registry.protocol["digest_short_len"])],
            )
            self.registry.cost_ledger.record(
                regime_id=self.regime_id,
                component_name=comp_name,
                cell_digest=self.grid_cell_digest,
                passed=True,
                quality_score=score,
                latency_ms=latency_ms,
            )
            return out
        except OmniNodeContractPermissionError:
            latency_ms = (time.perf_counter() - t_start) * ms_scale
            self.library.grid_recorder.record_run(
                run_id=run_id,
                kind=self.name,
                source=self.path(),
                grid=self.registry.grid,
                raw_config=raw_cfg_for_grid,
                passed=False,
                score=0.0,
                latency_ms=latency_ms,
                failure="contract",
            )
            raise
        except OmniNodeFailure as err:
            latency_ms = (time.perf_counter() - t_start) * ms_scale
            comp_label = str(self._card.get("name", self.name)) if self._card else self.name
            if err.kind == "capability":
                self.library.facts.emit(
                    "gap_observed",
                    node=self.path(),
                    want=str(job.get("want", self.cfg.get("want", comp_label))),
                    capability_tag=job.get("capability_tag", self.cfg.get("capability_tag")),
                    reason=str(err),
                )
            self.library.grid_recorder.record_run(
                run_id=run_id,
                kind=comp_label,
                source=self.path(),
                grid=self.registry.grid,
                raw_config=raw_cfg_for_grid,
                passed=False,
                score=0.0,
                latency_ms=latency_ms,
                failure=err.kind,
            )
            self.library.facts.emit(
                "observed",
                node=self.path(),
                component=comp_label,
                passed=False,
                failure=err.kind,
                latency_ms=latency_ms,
            )
            self.registry.cost_ledger.record(
                regime_id=self.regime_id,
                component_name=comp_label,
                cell_digest=self.grid_cell_digest,
                passed=False,
                quality_score=0.0,
                latency_ms=latency_ms,
            )
            raise

    def write_lockfile(self, path: Path | None = None) -> Path:
        target = path or (self.library.state_root / str(self.registry.protocol["lockfile_rel_path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.library.export_lockfile()
        target.write_text(json.dumps(data, indent=int(self.registry.num["two"]), sort_keys=True) + "\n", encoding="utf-8")
        return target

    def handle_mcp_rpc(self, rpc_req: dict[str, Any]) -> dict[str, Any]:
        """Handle JSON-RPC 2.0 MCP methods: initialize, tools/list, tools/call."""
        proto = self.registry.protocol
        ver = str(proto["jsonrpc_version"])
        req_id = rpc_req.get("id")
        if rpc_req.get("jsonrpc") != ver or "method" not in rpc_req:
            return {
                "jsonrpc": ver,
                "id": req_id,
                "error": {"code": int(proto["jsonrpc_invalid_request"]), "message": "Invalid JSON-RPC envelope"},
            }

        method = str(rpc_req["method"])
        params = dict(rpc_req.get("params") or {})

        if method == "initialize":
            return {
                "jsonrpc": ver,
                "id": req_id,
                "result": {
                    "protocolVersion": str(proto["mcp_protocol_version"]),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.name, "version": "1.0.0"},
                },
            }

        if method == "tools/list":
            tools = [
                {
                    "name": self.name,
                    "description": str(self.card_meta().get("summary", "Root OmniNode operator")),
                    "inputSchema": {"type": "object", "additionalProperties": True},
                }
            ]
            for c_name, c_shell in sorted(self.children.items()):
                tools.append(
                    {
                        "name": c_name,
                        "description": str(c_shell.card_meta().get("summary", f"Child OmniNode {c_name}")),
                        "inputSchema": {"type": "object", "additionalProperties": True},
                    }
                )
            return {"jsonrpc": ver, "id": req_id, "result": {"tools": tools}}

        if method == "tools/call":
            tool_name = str(params.get("name", ""))
            tool_args = dict(params.get("arguments") or {})
            target_shell: OmniNodeShell | None = None
            if tool_name == self.name:
                target_shell = self
            elif tool_name in self.children:
                target_shell = self.children[tool_name]
            if target_shell is None:
                return {
                    "jsonrpc": ver,
                    "id": req_id,
                    "error": {"code": int(proto["jsonrpc_invalid_params"]), "message": f"Unknown tool: {tool_name}"},
                }
            try:
                call_out = target_shell.handle(tool_args)
                return {
                    "jsonrpc": ver,
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(call_out)}],
                        "isError": False,
                    },
                }
            except OmniNodeFailure as err:
                return {
                    "jsonrpc": ver,
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": str(err), "kind": err.kind})}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": ver,
            "id": req_id,
            "error": {"code": int(proto["jsonrpc_method_not_found"]), "message": f"Method not found: {method}"},
        }

    def start_background_server(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> LoopbackServerHandle:
        """Start background loopback HTTP (/hello, /health, /job) + MCP (/mcp) server on 127.0.0.1:0."""
        proto = self.registry.protocol
        bind_host = str(host if host is not None else proto["loopback_host"])
        bind_port = int(port if port is not None else proto["ephemeral_port"])
        shell_ref = self

        class _OmniHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_json(self, status_code: int, body_obj: dict[str, Any]) -> None:
                raw = json.dumps(body_obj).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/hello":
                    self._send_json(
                        int(proto["http_ok"]),
                        {
                            "protocol": "omninode/1",
                            "node": shell_ref.path(),
                            "card": shell_ref.card_meta(),
                            "grid_cell_digest": shell_ref.grid_cell_digest,
                        },
                    )
                    return
                if self.path == "/health":
                    self._send_json(
                        int(proto["http_ok"]),
                        {
                            "status": "ok",
                            "node": shell_ref.path(),
                            "grid_generation": shell_ref.registry.grid.generation,
                            "children": sorted(shell_ref.children.keys()),
                            "search_count": shell_ref.library.search_count,
                        },
                    )
                    return
                self._send_json(int(proto["http_not_found"]), {"error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError:
                    if self.path == "/mcp":
                        self._send_json(
                            int(proto["http_ok"]),
                            {
                                "jsonrpc": str(proto["jsonrpc_version"]),
                                "id": None,
                                "error": {"code": int(proto["jsonrpc_parse_error"]), "message": "Parse error"},
                            },
                        )
                    else:
                        self._send_json(int(proto["http_bad_request"]), {"ok": False, "error": "invalid_json"})
                    return

                if self.path == "/job":
                    try:
                        res = shell_ref.handle(payload)
                        self._send_json(int(proto["http_ok"]), {"ok": True, "result": res})
                    except OmniNodeFailure as err:
                        self._send_json(
                            int(proto["http_bad_request"]),
                            {"ok": False, "failure_kind": err.kind, "error": str(err)},
                        )
                    return

                if self.path == "/mcp":
                    rpc_out = shell_ref.handle_mcp_rpc(payload)
                    self._send_json(int(proto["http_ok"]), rpc_out)
                    return

                self._send_json(int(proto["http_not_found"]), {"error": "not_found"})

        httpd = _ThreadingHTTPServer((bind_host, bind_port), _OmniHandler)
        actual_host, actual_port = httpd.server_address[0], int(httpd.server_address[1])
        srv_thread = threading.Thread(target=httpd.serve_forever, name=f"omninode-http-{actual_port}", daemon=True)
        srv_thread.start()
        return LoopbackServerHandle(
            host=str(actual_host),
            port=actual_port,
            base_url=f"http://{actual_host}:{actual_port}",
            server=httpd,
            thread=srv_thread,
        )
