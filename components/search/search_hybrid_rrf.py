"""Hybrid Reciprocal Rank Fusion (RRF) CARD search driver."""

from __future__ import annotations

import math
import re
from typing import Any

CARD = {
    "name": "search_hybrid_rrf",
    "role": "search",
    "search_method": "hybrid_rrf",
    "summary": "Fuses BM25 lexical rank, IDF specificity rank, and capability-tag overlap via RRF.",
    "capabilities": ["search_hybrid_rrf", "reciprocal_rank_fusion"],
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _entry_tokens(entry: dict[str, Any]) -> tuple[list[str], set[str]]:
    card = entry.get("card", {})
    cap_tokens = set(TOKEN_RE.findall(" ".join(str(c) for c in card.get("capabilities", [])).lower()))
    all_tokens = TOKEN_RE.findall(
        " ".join(
            [
                str(entry.get("name", "")),
                str(card.get("summary", "")),
                " ".join(str(c) for c in card.get("capabilities", [])),
                str(card.get("role", "")),
                str(card.get("stage", "")),
            ]
        ).lower()
    )
    return all_tokens, cap_tokens


def rank_cards(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    grid_cell: dict[str, Any],
    registry: Any,
    facts_rows: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    _ = facts_rows
    rrf_k = float(grid_cell["search_rrf_rank_constant"])
    k1 = float(grid_cell["search_bm25_k1"])
    b_param = float(grid_cell["search_bm25_b"])
    half = float(registry.num["half"])
    eps = float(registry.num["eps"])

    q_toks = TOKEN_RE.findall(query.lower())
    q_set = set(q_toks)
    parsed = [_entry_tokens(c) for c in candidates]
    n_docs = float(max(1, len(candidates)))
    avg_dl = sum(float(len(p[0])) for p in parsed) / n_docs + eps

    df_map = {t: sum(1 for p in parsed if t in p[0]) for t in q_set}

    bm25_list: list[tuple[float, str]] = []
    cap_list: list[tuple[float, str]] = []
    for item, (all_t, cap_t) in zip(candidates, parsed):
        name = str(item["name"])
        dl = float(len(all_t))
        b_score = 0.0
        for tok in q_toks:
            tf = float(all_t.count(tok))
            if tf > 0.0:
                df = float(df_map[tok])
                idf = math.log(1.0 + (n_docs - df + half) / (df + half))
                b_score += idf * ((tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b_param + b_param * (dl / avg_dl)) + eps))
        c_overlap = float(len(q_set & cap_t)) / float(max(1, len(q_set)))
        bm25_list.append((b_score, name))
        cap_list.append((c_overlap, name))

    bm25_list.sort(key=lambda t: (-t[0], t[1]))
    cap_list.sort(key=lambda t: (-t[0], t[1]))
    rank_bm = {name: idx + 1 for idx, (_, name) in enumerate(bm25_list)}
    rank_cp = {name: idx + 1 for idx, (_, name) in enumerate(cap_list)}

    fused: list[tuple[float, str, dict[str, Any]]] = []
    for item in candidates:
        nm = str(item["name"])
        rrf = (1.0 / (rrf_k + float(rank_bm[nm]))) + (1.0 / (rrf_k + float(rank_cp[nm])))
        fused.append((rrf, nm, item))

    fused.sort(key=lambda t: (-t[0], t[1]))
    return [t[-1] for t in fused[: max(1, int(top_k))]]
