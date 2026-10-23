"""Retrieval metrics and generation-score aggregation.

Unlike a pipeline-coupled harness, CrossJudge computes retrieval metrics from
IDs you already have: each record carries ``retrieved_ids`` (what your system
returned, in rank order) and ``relevant_ids`` (ground truth). Bring your own
retrieval — CrossJudge just scores it.
"""

from __future__ import annotations

import math
from collections import defaultdict


def retrieval_metrics_for_query(
    retrieved_ids: list[str], relevant_ids: list[str], top_k: int | None = None
) -> dict:
    """MRR / hit / recall / NDCG for a single query (binary relevance).

    ``retrieved_ids`` must be in rank order (best first). ``top_k`` truncates
    the retrieved list; ``None`` uses all of it.
    """
    retrieved = retrieved_ids[:top_k] if top_k is not None else list(retrieved_ids)
    gt = set(relevant_ids)
    if not gt:
        return {"reciprocal_rank": 0.0, "hit": 0, "recall": 0.0, "ndcg": 0.0}

    # Reciprocal rank of the first relevant hit.
    rr = 0.0
    for rank, rid in enumerate(retrieved, 1):
        if rid in gt:
            rr = 1.0 / rank
            break

    found = gt & set(retrieved)
    hit = 1 if found else 0
    recall = len(found) / len(gt)

    # A relevant id earns gain only at its first rank; repeats count as non-relevant,
    # so duplicates in the retrieved list can't push NDCG above 1.0.
    dcg = 0.0
    seen: set[str] = set()
    for rank, rid in enumerate(retrieved, 1):
        if rid in gt and rid not in seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen.add(rid)
    ideal_k = min(len(gt), len(retrieved))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"reciprocal_rank": rr, "hit": hit, "recall": recall, "ndcg": ndcg}


def aggregate_retrieval_metrics(records: list[dict], top_k: int | None = 5) -> dict:
    """Aggregate retrieval metrics over all records that carry retrieval IDs.

    Records without both ``retrieved_ids`` and ``relevant_ids`` are skipped.
    """
    scored = [
        r for r in records if r.get("retrieved_ids") is not None and r.get("relevant_ids")
    ]
    if not scored:
        return {}

    mrr = hit = recall = ndcg = 0.0
    for r in scored:
        m = retrieval_metrics_for_query(r["retrieved_ids"], r["relevant_ids"], top_k)
        mrr += m["reciprocal_rank"]
        hit += m["hit"]
        recall += m["recall"]
        ndcg += m["ndcg"]

    n = len(scored)
    return {
        "mrr": round(mrr / n, 4),
        "hit_rate": round(hit / n, 4),
        "recall_at_k": round(recall / n, 4),
        "ndcg_at_k": round(ndcg / n, 4),
        "num_queries": n,
        "top_k": top_k,
    }


def summarize_generation(scored_records: list[dict]) -> dict:
    """Aggregate faithfulness / relevancy / correctness, grouped by ``config``.

    ``scored_records`` are runner outputs: each has ``config`` and a ``scores``
    dict. Scores of ``-1`` (judge failure / not applicable) are excluded from means.
    """
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in scored_records:
        by_config[r.get("config", "default")].append(r)

    summary: dict[str, dict] = {}
    for config in sorted(by_config):
        rows = by_config[config]
        valid = [r for r in rows if r["scores"].get("faithfulness", -1) >= 0]
        if not valid:
            summary[config] = {"count": len(rows), "note": "no valid scores"}
            continue

        stats = {
            "count": len(valid),
            "faithfulness_mean": _mean(valid, "faithfulness"),
            "relevancy_mean": _mean(valid, "relevancy"),
        }

        correctness_rows = [r for r in valid if r["scores"].get("correctness", -1) >= 0]
        if correctness_rows:
            stats["correctness_mean"] = _mean(correctness_rows, "correctness")
            stats["correctness_count"] = len(correctness_rows)
            total_hits = sum(r["scores"].get("correctness_hits", 0) for r in correctness_rows)
            total_facts = sum(r["scores"].get("correctness_total", 0) for r in correctness_rows)
            if total_facts:
                stats["claim_recall"] = round(total_hits / total_facts, 3)
                stats["facts_hit"] = total_hits
                stats["facts_total"] = total_facts

        # Retrieval metrics for this config, if the records carry IDs.
        retrieval = aggregate_retrieval_metrics(rows)
        if retrieval:
            stats["retrieval"] = retrieval

        summary[config] = stats
    return summary


def _mean(rows: list[dict], key: str) -> float:
    return round(sum(r["scores"][key] for r in rows) / len(rows), 3)
