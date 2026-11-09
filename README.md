# OmniNode Manifold

**OmniNode Manifold** (`v1.0.0`) is a universal, parameterized, recursive component shell (`OmniNodeShell`) that encapsulates every stage, search driver, source connector, latent-space operator, consistency gate, self-synthesizing loop, and portable multi-runtime compiler inside one uniform execution envelope.

---

## Core Architectural Invariants

1. **One Recursive Shell (`OmniNodeShell`)**:
   Every node in an execution graph is an instance of `OmniNodeShell(cfg, grid_registry, parent=None, depth=0)`. Node behavior comes strictly from the self-describing `CARD = {...}` module bound to that node and the contextual `NodeGrid` configuration cell injected into it.
2. **Static Syntax-Tree `CARD` Discovery & Content-Addressed Storage**:
   Candidate bricks are inspected statically via `ast.parse` and `ast.literal_eval` (`card_of`) without executing untrusted code during discovery. Loaded modules are stored in `.omni/cas/` by content digest and pinned in `lockfile.json`. Rebuilding an `OmniNodeShell` from `lockfile.json` across an emptied source directory performs `0` searches and reproduces identical outputs; upstream content tampering fails closed with a `contract` error and logs a mismatch observation.
3. **Declarative `NodeGrid` & Zero Magic Literals**:
   All capacities, recursion limits, Okapi BM25 / Reciprocal Rank Fusion constants, sparse dictionary dimensions, ridge penalties, conformal alpha levels, nullspace steering scales, Koopman thresholds, and WebAssembly binary opcodes are governed by `config/omninode_universal_grid.json` (`36` parameter axes, `ParameterBounds`, cross-axis constraints, `7`-stage `GridLedger` enforcing disjoint `maker_id != checker_id`, and `Generation 1 -> Generation 2` boundary auto-expansion). `ZeroHardcodedValuesAstGate` and `DomainAgnosticNamingGate` scan all Python modules mechanically.
4. **Algebraic Shape Composition (`>> // | @`)**:
   - `a >> b` — Sequential pipe forwarding activations and state metadata.
   - `a | b` — Parallel fan-out fused via cellular sheaf inverse-variance or ridge consensus.
   - `a // b` — Waterfall fallback reranking recoverable failures (`transient`, `capability`, `data`, `budget`) while halting immediately on `contract` or `bad_input` errors.
   - `body @ checker` — Independent quorum check handoff enforcing `maker_id != checker_id`.
5. **Multi-Transport Wire Envelope & `9`-Runtime Lowering**:
   Every `OmniNodeShell` runs in-process, serves loopback HTTP (`GET /hello`, `GET /health`, `POST /job`) and JSON-RPC 2.0 Model Context Protocol (`POST /mcp`: `initialize`, `tools/list`, `tools/call`) on ephemeral port `127.0.0.1:0`, and lowers fitted float32 weights into `WebAssembly (.wasm)`, `Rust (#![no_std])`, `ISO C99`, `WGSL`, `GLSL (#version 300 es)`, `SQL`, `ES2022 JS`, `Python`, and `.safetensors` in memory with zero binary residue on disk.
6. **Autonomous `gap-loop` Self-Synthesis**:
   When repeated `capability` misses accumulate in `.omni/facts.jsonl`, `card_gap_builder_loop` drafts a clean `built_<slug>.py` `CARD` brick, checks AST and naming gates, evaluates shown and frozen hidden holdout suites under disjoint `maker_id != checker_id`, admits passing modules into `.omni/cas/`, and resolves the previously declined job on re-run.

---

## Encapsulated `22` Self-Describing `CARD` Bricks

### Boot Source Drivers (`components/sources/`)
- `source_folder` — Local filesystem directory catalog.
- `source_memory_bundle` — Portable `git bundle` and archive snapshot reader.
- `source_object_pack` — Immutable JSON object-pack reader.

### Boot Search Drivers (`components/search/`)
- `search_idf` — Inverse document frequency token overlap ranker.
- `search_bm25` — Grid-parameterized Okapi BM25 ranker (`k1`, `b`).
- `search_hybrid_rrf` — Hybrid Reciprocal Rank Fusion over BM25 and capability overlap.
- `search_outcome_grid` — Empirical outcome ranker combining lexical match with `.omni/facts.jsonl` pass rates and `OperationCostLedger` latency.

### Universal Stage, Director, Transform, Worker, Check, & Loop Bricks (`components/cards/`)
1. `card_orient_spectral_profiler` (`stage: orient`) — Singular spectrum Participation Ratio, anisotropy, and Marchenko-Pastur RMT signal count.
2. `card_compose_grid_selector` (`stage: compose`) — Contextual `NodeGrid` cell router and boundary auto-expander.
3. `card_shape_director` (`director`) — Recursive parser and evaluator for `>> // | @` expressions.
4. `card_sae_dictionary` (`transform`) — Parameterized `TopK | BatchTopK | JumpReLU | GatedSAE | MatryoshkaNested` × `None | ZCA | Mahalanobis` sparse autoencoder.
5. `card_crosscoder_transcoder` (`transform`) — Multi-layer `SparseCrosscoder` and causal `MlpTranscoder`.
6. `card_elk_sheaf_conformal_probe` (`check`) — Internal consistency contrast direction, sheaf coboundary energy, and Adaptive Conformal Inference (`ACI`).
7. `card_leace_concept_eraser` (`transform`) — Economy-SVD propensity-balanced oblique concept erasure driving nuisance covariance to zero.
8. `card_conditional_nullspace_steerer` (`transform`) — Anchor-nullspace steering with shrinkage Fisher-centroid regime gating.
9. `card_platonic_latent_bridge` (`transform`) — Zero-retraining Orthogonal Procrustes and Ridge cross-space latent bridge (`384-d <-> 768-d`).
10. `card_trajectory_loop_breaker` (`check`) — Frenet-Serret discrete curvature, sliding Participation Ratio, and Koopman spectral modulus loop detector.
11. `card_hyperlora_liquid_operator` (`worker`) — Input-conditioned Kronecker nullspace `HyperLoRA` and causal Green's-function `LiquidStateSpace` sequence operator.
12. `card_woodbury_sinkhorn_attributed_solver` (`worker`) — Primal/dual Woodbury ridge, differentially private RLS, Sinkhorn-Knopp balanced routing, and bitemporal `lease_epoch` cache reuse.
13. `card_portable_9runtime_compiler` (`transform`) — In-memory `9`-target float32 lowering and cross-runtime parity checker.
14. `card_independent_quorum_gate` (`check`) — Disjoint `maker_id != checker_id` invariant gate.
15. `card_gap_builder_loop` (`loop`) — Autonomous capability-miss `CARD` synthesizer.

---

## Quickstart

```bash
# 1. Run mechanical AST and naming conformance gates
PYTHONDONTWRITEBYTECODE=1 python3 -m omninode_manifold.cli check-gates

# 2. Run the full test suite
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q

# 3. Execute the empirical benchmark suite
PYTHONDONTWRITEBYTECODE=1 python3 -m omninode_manifold.cli benchmark
```

## License

MIT License — Copyright (c) 2026 AI Done Right Corp.
