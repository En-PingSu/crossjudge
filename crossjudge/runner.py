"""Run a cross-family evaluation over a list of records.

A *record* is one answer your RAG system produced, plus whatever ground truth
you have. Only ``question`` and ``response`` are required::

    {
      "id": "q1",                       # optional, defaults to index
      "question": "...",                # required
      "response": "...",                # required — the answer to judge
      "context": "...retrieved text...", # optional — enables faithfulness
      "key_facts": ["...", "..."],      # optional — enables correctness
      "config": "rag_rerank",           # optional — grouping label in the summary
      "generator_model": "openai/gpt-4o", # optional — enables the cross-family check
      "retrieved_ids": ["c3", "c1"],    # optional (rank order) — enables retrieval metrics
      "relevant_ids": ["c1"]            # optional — ground-truth ids
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from .client import DEFAULT_JUDGE_MODEL, check_cross_family, get_openrouter_client
from .judges import judge_correctness, judge_faithfulness, judge_relevancy
from .metrics import summarize_generation


def evaluate(
    records: list[dict],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    client: OpenAI | None = None,
    progress: bool = True,
) -> dict:
    """Score every record and return ``{judge_model, summary, records, ...}``.

    The judge runs at temperature 0. Faithfulness is scored only when a record
    has ``context``; correctness only when it has ``key_facts``. Retrieval
    metrics are folded into the summary for any config whose records carry
    ``retrieved_ids`` + ``relevant_ids``.
    """
    client = client or get_openrouter_client()

    generator_models = [r["generator_model"] for r in records if r.get("generator_model")]
    collisions = check_cross_family(judge_model, generator_models)

    scored: list[dict] = []
    for i, r in enumerate(records):
        rid = r.get("id", i)
        if progress:
            print(f"  [{i + 1}/{len(records)}] {r.get('config', 'default')}: "
                  f"{str(r['question'])[:55]}...")

        scores: dict = {}

        faith = judge_faithfulness(
            r["question"], r["response"], r.get("context", ""), client, judge_model
        )
        scores["faithfulness"] = faith["score"]
        scores["faithfulness_reasoning"] = faith["reasoning"]

        relev = judge_relevancy(r["question"], r["response"], client, judge_model)
        scores["relevancy"] = relev["score"]
        scores["relevancy_reasoning"] = relev["reasoning"]

        if r.get("key_facts"):
            correct = judge_correctness(
                r["question"], r["response"], r["key_facts"], client, judge_model
            )
            scores["correctness"] = correct["score"]
            scores["correctness_reasoning"] = correct["reasoning"]
            scores["correctness_per_fact"] = correct.get("per_fact", [])
            scores["correctness_hits"] = correct.get("hits", 0)
            scores["correctness_total"] = correct.get("total", 0)

        scored.append({
            "id": rid,
            "question": r["question"],
            "config": r.get("config", "default"),
            "generator_model": r.get("generator_model"),
            "retrieved_ids": r.get("retrieved_ids"),
            "relevant_ids": r.get("relevant_ids"),
            "scores": scores,
        })

    return {
        "judge_model": judge_model,
        "cross_family_collisions": collisions,
        "summary": summarize_generation(scored),
        "records": scored,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def save_report(result: dict, path: str | Path) -> Path:
    """Write the full result (summary + per-record scores) to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path


def print_summary(result: dict) -> None:
    """Pretty-print the per-config summary to stdout."""
    print("\n" + "=" * 60)
    print(f"CrossJudge summary  (judge: {result['judge_model']})")
    if result.get("cross_family_collisions"):
        print(f"  WARNING self-eval risk: {result['cross_family_collisions']}")
    print("=" * 60)
    for config, stats in result["summary"].items():
        print(f"\n{config}:")
        for k, v in stats.items():
            if k == "retrieval":
                print(f"  retrieval: " + "  ".join(f"{rk}={rv}" for rk, rv in v.items()))
            elif isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
