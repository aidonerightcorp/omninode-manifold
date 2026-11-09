"""Inverse Document Frequency lexical CARD search driver."""

from __future__ import annotations

import math
import re
from typing import Any

CARD = {
    "name": "search_idf",
    "role": "search",
    "search_method": "idf",
    "summary": "Ranks self-describing CARD bricks by inverse document frequency token overlap.",
    "capabilities": ["search_idf", "lexical_idf"],
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens_of_entry(entry: dict[str, Any]) -> list[str]:
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
    _ = (grid_cell, facts_rows)
    eps = float(registry.num["eps"])
    q_tokens = set(TOKEN_RE.findall(query.lower()))
    n_docs = float(max(1, len(candidates)))
    doc_sets = [set(_tokens_of_entry(item)) for item in candidates]

    df_counts: dict[str, int] = {}
    for tok in q_tokens:
        df_counts[tok] = sum(1 for ds in doc_sets if tok in ds)

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for item, d_set in zip(candidates, doc_sets):
        score = 0.0
        for tok in q_tokens:
            if tok in d_set:
                score += math.log((n_docs + 1.0) / (float(df_counts[tok]) + 1.0)) + 1.0
        scored.append((score + eps, str(item["name"]), item))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [t[-1] for t in scored[: max(1, int(top_k))]]
