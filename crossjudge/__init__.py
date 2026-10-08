"""CrossJudge — honest, cross-family LLM-as-judge evaluation for RAG systems.

Score your RAG outputs on faithfulness, relevancy, and per-fact correctness
with a judge from a *different* provider family than your generator — the
cheapest way to keep self-evaluation bias out of your numbers.

Quick start::

    from crossjudge import evaluate, print_summary

    records = [{"question": "...", "response": "...", "context": "...",
                "generator_model": "openai/gpt-4o"}]
    result = evaluate(records, judge_model="anthropic/claude-sonnet-4")
    print_summary(result)
"""

from .client import (
    DEFAULT_JUDGE_MODEL,
    check_cross_family,
    get_openrouter_client,
    model_family,
)
from .judges import (
    judge_correctness,
    judge_faithfulness,
    judge_relevancy,
    judge_retrieval_coverage,
)
from .metrics import (
    aggregate_retrieval_metrics,
    retrieval_metrics_for_query,
    summarize_generation,
)
from .runner import evaluate, print_summary, save_report

__version__ = "0.1.0"

__all__ = [
    "evaluate",
    "print_summary",
    "save_report",
    "judge_faithfulness",
    "judge_relevancy",
    "judge_correctness",
    "judge_retrieval_coverage",
    "retrieval_metrics_for_query",
    "aggregate_retrieval_metrics",
    "summarize_generation",
    "get_openrouter_client",
    "check_cross_family",
    "model_family",
    "DEFAULT_JUDGE_MODEL",
]
