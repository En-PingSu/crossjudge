"""Offline tests — no network, no API key required.

Judge functions are exercised with a fake OpenAI-shaped client so the parsing
and scoring logic is covered without calling a real model.
"""

import warnings

import pytest

from crossjudge import (
    aggregate_retrieval_metrics,
    check_cross_family,
    judge_correctness,
    judge_faithfulness,
    judge_relevancy,
    model_family,
    retrieval_metrics_for_query,
    summarize_generation,
)


# --- a minimal stand-in for the OpenAI client -----------------------------

class _FakeClient:
    """Returns a fixed completion string, mimicking client.chat.completions.create."""

    def __init__(self, reply: str, raise_exc: bool = False):
        self._reply = reply
        self._raise = raise_exc
        self.chat = self
        self.completions = self

    def create(self, **_kwargs):
        if self._raise:
            raise RuntimeError("boom")
        message = type("Msg", (), {"content": self._reply})
        choice = type("Choice", (), {"message": message})
        return type("Resp", (), {"choices": [choice]})


# --- model_family / cross-family guardrail --------------------------------

def test_model_family():
    assert model_family("openai/gpt-4o") == "openai"
    assert model_family("anthropic/claude-sonnet-4") == "anthropic"
    assert model_family("") == ""


def test_cross_family_collision_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        collisions = check_cross_family(
            "openai/gpt-4o", ["openai/gpt-4o", "meta-llama/llama-3.3-70b-instruct"]
        )
    assert collisions == ["openai/gpt-4o"]
    assert len(caught) == 1


def test_cross_family_clean_is_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        collisions = check_cross_family(
            "anthropic/claude-sonnet-4", ["openai/gpt-4o", "mistralai/mistral-small"]
        )
    assert collisions == []
    assert len(caught) == 0


# --- retrieval metrics ----------------------------------------------------

def test_retrieval_hit_and_rank():
    m = retrieval_metrics_for_query(["c3", "c1", "c9"], ["c1"], top_k=5)
    assert m["hit"] == 1
    assert m["reciprocal_rank"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)


def test_retrieval_miss():
    m = retrieval_metrics_for_query(["x", "y"], ["c1"], top_k=5)
    assert m == {"reciprocal_rank": 0.0, "hit": 0, "recall": 0.0, "ndcg": 0.0}


def test_retrieval_top_k_truncates():
    # relevant doc is at rank 3 but top_k=2 cuts it off
    m = retrieval_metrics_for_query(["a", "b", "c1"], ["c1"], top_k=2)
    assert m["hit"] == 0 and m["reciprocal_rank"] == 0.0


def test_retrieval_empty_ground_truth():
    assert retrieval_metrics_for_query(["a"], [], top_k=5)["hit"] == 0


def test_aggregate_skips_records_without_ids():
    records = [
        {"retrieved_ids": ["c1"], "relevant_ids": ["c1"]},
        {"question": "no ids here"},
    ]
    agg = aggregate_retrieval_metrics(records, top_k=5)
    assert agg["num_queries"] == 1
    assert agg["hit_rate"] == 1.0


# --- summary aggregation --------------------------------------------------

def test_summary_groups_and_excludes_invalid():
    scored = [
        {"config": "baseline", "scores": {"faithfulness": 0.0, "relevancy": 1.0}},
        {"config": "baseline", "scores": {"faithfulness": -1.0, "relevancy": -1.0}},  # excluded
        {"config": "rag", "scores": {"faithfulness": 1.0, "relevancy": 1.0,
                                       "correctness": 1.0, "correctness_hits": 2,
                                       "correctness_total": 2},
         "retrieved_ids": ["c1"], "relevant_ids": ["c1"]},
    ]
    s = summarize_generation(scored)
    assert s["baseline"]["count"] == 1  # the -1 row dropped
    assert s["baseline"]["faithfulness_mean"] == 0.0
    assert s["rag"]["faithfulness_mean"] == 1.0
    assert s["rag"]["claim_recall"] == 1.0
    assert s["rag"]["retrieval"]["hit_rate"] == 1.0


# --- judges (fake client) -------------------------------------------------

def test_judge_faithfulness_yes_no():
    assert judge_faithfulness("q", "r", "ctx", _FakeClient("YES: grounded"), "m")["score"] == 1.0
    assert judge_faithfulness("q", "r", "ctx", _FakeClient("NO: made up"), "m")["score"] == 0.0


def test_judge_relevancy_yes():
    assert judge_relevancy("q", "r", _FakeClient("YES: on topic"), "m")["score"] == 1.0


def test_judge_correctness_per_fact_recall():
    reply = "FACT 1: YES — present\nFACT 2: NO — missing"
    out = judge_correctness("q", "r", ["fact a", "fact b"], _FakeClient(reply), "m")
    assert out["score"] == 0.5
    assert out["hits"] == 1 and out["total"] == 2


def test_judge_correctness_no_key_facts():
    out = judge_correctness("q", "r", [], _FakeClient("irrelevant"), "m")
    assert out["score"] == -1.0


def test_judge_error_returns_minus_one():
    out = judge_faithfulness("q", "r", "ctx", _FakeClient("", raise_exc=True), "m")
    assert out["score"] == -1.0
    assert "judge error" in out["reasoning"]
