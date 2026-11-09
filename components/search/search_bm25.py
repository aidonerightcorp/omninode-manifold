"""Parameterized Okapi BM25 CARD search driver governed by NodeGrid k1 and b axes."""

from __future__ import annotations

import math
import re
from typing import Any

CARD = {
    "name": "search_bm25",
    "role": "search",
    "search_method": "bm25",
    "summary": "Ranks CARD bricks via Okapi BM25 using grid-injected k1 and length-normalization b.",
    "capabilities": ["search_bm25", "okapi_bm25"],
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _doc_tokens(entry: dict[str, Any]) -> list[str]:
    card = entry.get("card", {})
    blob = " ".join(
        [
            str(entry.get("name", "")),
            str(card.get("summary", "")),
            " ".join(str(c) for c in card.get("capabilities", [])),
            str(card.get("role", "")),
            str(card.get("stage", "")),
        ]
    ).lower()
    return TOKEN_RE.findall(blob)


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
    k1 = float(grid_cell["search_bm25_k1"])
    b_param = float(grid_cell["search_bm25_b"])
    half = float(registry.num["half"])
    eps = float(registry.num["eps"])

    q_tokens = TOKEN_RE.findall(query.lower())
    docs = [_doc_tokens(item) for item in candidates]
    n_docs = float(max(1, len(docs)))
    avg_dl = sum(float(len(d)) for d in docs) / n_docs + eps

    df_map: dict[str, int] = {}
    for tok in set(q_tokens):
        df_map[tok] = sum(1 for d in docs if tok in d)

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for item, d_toks in zip(candidates, docs):
        dl = float(len(d_toks))
        score = 0.0
        for tok in q_tokens:
            tf = float(d_toks.count(tok))
            if tf <= 0.0:
                continue
            df = float(df_map[tok])
            idf = math.log(1.0 + (n_docs - df + half) / (df + half))
            denom = tf + k1 * (1.0 - b_param + b_param * (dl / avg_dl))
            score += idf * ((tf * (k1 + 1.0)) / (denom + eps))
        scored.append((score, str(item["name"]), item))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [t[-1] for t in scored[: max(1, int(top_k))]]
