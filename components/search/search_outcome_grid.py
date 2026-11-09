"""Outcome-conditioned CARD search driver fusing lexical relevance with facts.jsonl and OperationCostLedger."""

from __future__ import annotations

import re
from typing import Any

CARD = {
    "name": "search_outcome_grid",
    "role": "search",
    "search_method": "outcome_grid",
    "summary": "Ranks CARD bricks by combining lexical match with empirical pass rate and latency from facts.jsonl and OperationCostLedger.",
    "capabilities": ["search_outcome_grid", "empirical_ledger_reranking"],
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def rank_cards(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    grid_cell: dict[str, Any],
    registry: Any,
    facts_rows: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    pass_w = float(grid_cell["search_outcome_pass_weight"])
    lex_w = 1.0 - pass_w
    ms_scale = float(registry.num["thousand_ms"])
    half = float(registry.num["half"])

    q_set = set(TOKEN_RE.findall(query.lower()))

    fact_passes: dict[str, list[float]] = {}
    fact_latencies: dict[str, list[float]] = {}
    for row in facts_rows:
        if row.get("kind") == "observed" and row.get("component"):
            comp = str(row["component"])
            fact_passes.setdefault(comp, []).append(1.0 if bool(row.get("passed", False)) else 0.0)
            fact_latencies.setdefault(comp, []).append(float(row.get("latency_ms", 0.0)))

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for item in candidates:
        name = str(item["name"])
        card = item.get("card", {})
        doc_set = set(
            TOKEN_RE.findall(
                " ".join(
                    [
                        name,
                        str(card.get("summary", "")),
                        " ".join(str(c) for c in card.get("capabilities", [])),
                    ]
                ).lower()
            )
        )
        lex_score = float(len(q_set & doc_set)) / float(max(1, len(q_set)))

        ledger_stats = registry.cost_ledger.component_summary(name)
        if float(ledger_stats["count"]) > 0.0:
            emp_pass = float(ledger_stats["pass_rate"])
            emp_lat = float(ledger_stats["mean_latency_ms"])
        elif name in fact_passes:
            p_list = fact_passes[name]
            l_list = fact_latencies[name]
            emp_pass = sum(p_list) / float(len(p_list))
            emp_lat = sum(l_list) / float(len(l_list))
        else:
            emp_pass = half
            emp_lat = 1.0

        speed_bonus = 1.0 / (1.0 + emp_lat / ms_scale)
        empirical_utility = emp_pass * speed_bonus
        total_score = lex_w * lex_score + pass_w * empirical_utility
        scored.append((total_score, name, item))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [t[-1] for t in scored[: max(1, int(top_k))]]
