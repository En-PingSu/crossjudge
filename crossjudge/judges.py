"""LLM-as-judge scoring functions.

Each judge takes the question/response (and context or key facts), calls the
judge model at temperature 0, and returns ``{"score": float, "reasoning": str, ...}``.
A score of ``-1.0`` means the judge call failed or was not applicable — callers
filter those out before aggregating.
"""

from __future__ import annotations

import re

from openai import OpenAI

from .prompts import CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT, RELEVANCY_PROMPT

BASELINE_CONTEXT = "(No source context provided — baseline mode)"


def _complete(client: OpenAI, judge_model: str, prompt: str, max_tokens: int) -> str:
    result = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return result.choices[0].message.content.strip()


_VERDICT = re.compile(r"\s*(YES|NO)\b")


def _parse_per_fact(text: str, key_facts: list[str]) -> tuple[int, list[dict]]:
    """Parse ``FACT N: YES/NO`` lines into hits + per-fact judgments.

    Only the first word after the colon is the verdict, so an explanation that
    mentions "yes" cannot flip a NO. If a fact has several lines (for example an
    echoed or injected one), it counts as present only when every line says YES.
    """
    lines = [line.strip() for line in text.splitlines()]
    hits = 0
    per_fact: list[dict] = []
    for i, fact in enumerate(key_facts):
        prefixes = (f"FACT {i + 1}:", f"FACT {i + 1} :")
        matches = [line for line in lines if line.upper().startswith(prefixes)]
        if not matches:
            per_fact.append(
                {"fact": fact, "present": False, "judgment": "(not found in judge output)"}
            )
            continue
        verdicts = []
        for line in matches:
            m = _VERDICT.match(line.upper().split(":", 1)[1])
            verdicts.append(m.group(1) if m else None)
        present = all(v == "YES" for v in verdicts)
        per_fact.append({"fact": fact, "present": present, "judgment": " | ".join(matches)})
        hits += int(present)
    return hits, per_fact


def judge_faithfulness(
    question: str,
    response: str,
    context: str,
    client: OpenAI,
    judge_model: str,
    prompt_template: str = FAITHFULNESS_PROMPT,
) -> dict:
    """Binary faithfulness: is every claim supported by the context? (1.0 / 0.0)."""
    prompt = prompt_template.format(
        question=question, response=response, context=context or BASELINE_CONTEXT
    )
    try:
        text = _complete(client, judge_model, prompt, max_tokens=200)
        return {"score": 1.0 if text.upper().startswith("YES") else 0.0, "reasoning": text}
    except Exception as e:  # noqa: BLE001 — surface the failure, keep the run going
        return {"score": -1.0, "reasoning": f"[judge error] {e}"}


def judge_relevancy(
    question: str,
    response: str,
    client: OpenAI,
    judge_model: str,
    prompt_template: str = RELEVANCY_PROMPT,
) -> dict:
    """Binary relevancy: does the response actually address the question? (1.0 / 0.0)."""
    prompt = prompt_template.format(question=question, response=response)
    try:
        text = _complete(client, judge_model, prompt, max_tokens=200)
        return {"score": 1.0 if text.upper().startswith("YES") else 0.0, "reasoning": text}
    except Exception as e:  # noqa: BLE001
        return {"score": -1.0, "reasoning": f"[judge error] {e}"}


def judge_correctness(
    question: str,
    response: str,
    key_facts: list[str],
    client: OpenAI,
    judge_model: str,
    prompt_template: str = CORRECTNESS_PROMPT,
) -> dict:
    """Per-fact claim recall: fraction of expected key facts present in the response."""
    if not key_facts:
        return {"score": -1.0, "reasoning": "No key facts to evaluate against", "per_fact": []}

    numbered = "\n".join(f"FACT {i + 1}: {f}" for i, f in enumerate(key_facts))
    prompt = prompt_template.format(question=question, response=response, numbered_facts=numbered)
    try:
        text = _complete(client, judge_model, prompt, max_tokens=500)
        hits, per_fact = _parse_per_fact(text, key_facts)
        return {
            "score": round(hits / len(key_facts), 3),
            "reasoning": text,
            "per_fact": per_fact,
            "hits": hits,
            "total": len(key_facts),
        }
    except Exception as e:  # noqa: BLE001
        return {"score": -1.0, "reasoning": f"[judge error] {e}", "per_fact": []}


def judge_retrieval_coverage(
    question: str,
    retrieved_context: str,
    key_facts: list[str],
    client: OpenAI,
    judge_model: str,
    prompt_template: str = CORRECTNESS_PROMPT,
) -> dict:
    """Retrieval ceiling: which key facts are present in the *retrieved context*?

    Same per-fact check as ``judge_correctness`` but run against the retrieved
    chunks instead of the generated answer. It tells you the maximum
    correctness a generator could reach without hallucinating — if a fact is
    not even retrieved, no amount of generation quality can recover it.
    """
    if not key_facts:
        return {
            "score": -1.0,
            "reasoning": "No key facts to evaluate against",
            "per_fact": [],
            "hits": 0,
            "total": 0,
        }

    numbered = "\n".join(f"FACT {i + 1}: {f}" for i, f in enumerate(key_facts))
    prompt = prompt_template.format(
        question=question, response=retrieved_context, numbered_facts=numbered
    )
    try:
        text = _complete(client, judge_model, prompt, max_tokens=500)
        hits, per_fact = _parse_per_fact(text, key_facts)
        return {
            "score": round(hits / len(key_facts), 3),
            "reasoning": text,
            "per_fact": per_fact,
            "hits": hits,
            "total": len(key_facts),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "score": -1.0,
            "reasoning": f"[judge error] {e}",
            "per_fact": [],
            "hits": 0,
            "total": 0,
        }
