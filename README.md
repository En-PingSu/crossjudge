# CrossJudge

**Honest, cross-family LLM-as-judge evaluation for RAG systems.**

CrossJudge scores your RAG outputs on **faithfulness**, **relevancy**, and per-fact
**correctness** using a judge model from a *different* provider family than the model
that generated the answers — the cheapest way to keep self-evaluation bias out of
your numbers. Bring your own outputs; CrossJudge grades them.

```python
from crossjudge import evaluate, print_summary

records = [{
    "question": "At what temperature does water boil at sea level?",
    "response": "Water boils at 100 °C (212 °F) at sea level.",
    "context": "[doc1] At standard atmospheric pressure, water boils at 100 °C.",
    "generator_model": "openai/gpt-4o",
}]

result = evaluate(records, judge_model="anthropic/claude-sonnet-4")
print_summary(result)
```

---

## The number that fooled me

I built a RAG system for Massachusetts tenant law and measured faithfulness with an
LLM judge. My first numbers looked clean:

| Faithfulness | Baseline (no retrieval) | RAG |
|---|---|---|
| **GPT‑4o judging GPT‑4o** | 0.64 | 0.78 |

A respectable baseline, a modest lift from retrieval. I almost reported it.

The problem: **the judge was GPT‑4o, and so was the generator.** A model grading text
from its own family grades it too kindly — it rewards its own style and waves through
its own confident-but-unsupported claims. My "baseline" wasn't really scoring 0.64; the
judge was flattering its own output.

So I swapped the judge to **Claude Sonnet 4** — a different provider family from every
generator (GPT‑4o, Llama 3.3, Mistral) — and kept everything else the same:

| Faithfulness | Baseline (no retrieval) | RAG (rerank) |
|---|---|---|
| GPT‑4o judging GPT‑4o | 0.64 | 0.78 |
| **Claude judging GPT‑4o** | **0.10** | **0.76** |

The ungrounded baseline **collapsed from 0.64 to 0.10** — without retrieval, the model
hallucinates specifics far more than the same-family judge admitted. RAG held steady
around 0.76. The real story was retrieval buying a **~7× faithfulness gain**, not the
~1.4× the flattering numbers implied.

The lesson is uncomfortable and cheap to act on: **who judges your outputs changes your
conclusions.** If the judge shares a family with the generator, your absolute scores are
suspect. CrossJudge makes the honest setup the default and warns you when you drift from
it.

> The numbers above come from the project CrossJudge was extracted from; the bias is
> documented in its evaluation report ("Self-Evaluation Bias — Judge = Generator").

---

## Why cross-family judging

- **Self-evaluation bias is real and large.** In the case above it moved an absolute
  faithfulness score by **0.54**. Same-family judging doesn't just add noise; it
  systematically inflates.
- **Relative comparisons can survive it, absolute ones don't.** If baseline and RAG use
  the same generator-as-judge, their *ranking* may hold — but the *magnitude* of the gap,
  the number you put in a report or on a slide, is distorted.
- **The fix costs one config line.** Point the judge at a different provider family. That's
  the entire intervention, and CrossJudge checks it for you (`check_cross_family`).

---

## What it measures

| Metric | Question it answers | Needs |
|---|---|---|
| **Faithfulness** | Is every claim supported by the retrieved context? (≈ precision) | `response`, `context` |
| **Relevancy** | Does the response actually address the question? | `response` |
| **Correctness** | What fraction of expected key facts did the response cover? (≈ recall) | `response`, `key_facts` |
| **Retrieval** | MRR / hit-rate / recall@k / NDCG@k of your retriever | `retrieved_ids`, `relevant_ids` |

Faithfulness and relevancy are binary (YES/NO) per item; correctness is per-fact claim
recall. Binary rubrics keep the judge consistent and the parsing robust. Retrieval metrics
are computed from the IDs you already have — CrossJudge never calls your retriever.

---

## Install

```bash
git clone https://github.com/En-PingSu/crossjudge.git
cd crossjudge
pip install -e .          # or: pip install -r requirements.txt

cp .env.example .env      # then add your OpenRouter key
```

CrossJudge routes judge calls through [OpenRouter](https://openrouter.ai), so one key
reaches every model family. Set `OPENROUTER_API_KEY` in the environment or in `.env`.

---

## Quickstart

A **record** is one answer your system produced plus whatever ground truth you have.
Only `question` and `response` are required:

```python
record = {
    "id": "q1",                          # optional
    "question": "...",                   # required
    "response": "...",                   # required — the answer to judge
    "context": "...retrieved text...",   # optional — enables faithfulness
    "key_facts": ["...", "..."],         # optional — enables correctness
    "config": "rag_rerank",              # optional — grouping label in the summary
    "generator_model": "openai/gpt-4o",  # optional — enables the cross-family check
    "retrieved_ids": ["c3", "c1"],       # optional (rank order) — enables retrieval metrics
    "relevant_ids": ["c1"],              # optional — ground-truth ids
}
```

Run the bundled example (baseline vs RAG for two questions):

```bash
python examples/example_run.py
```

You'll see the `baseline` config score far below `rag` on faithfulness and correctness —
the gap retrieval is buying you, measured by an independent judge.

---

## The cross-family guardrail

Pass `generator_model` on your records and CrossJudge warns when a generator shares the
judge's family:

```python
from crossjudge import check_cross_family

check_cross_family("openai/gpt-4o", ["openai/gpt-4o", "meta-llama/llama-3.3-70b-instruct"])
# UserWarning: Judge 'openai/gpt-4o' shares the 'openai' family with ['openai/gpt-4o'].
# These scores risk self-evaluation bias — pick a judge from a different provider family.
```

The run still completes; the collisions are returned in
`result["cross_family_collisions"]` so you can flag or discard those rows.

---

## API

| Function | Purpose |
|---|---|
| `evaluate(records, judge_model=...)` | Score all records; returns summary + per-record scores |
| `print_summary(result)` / `save_report(result, path)` | Print / persist a run |
| `judge_faithfulness / judge_relevancy / judge_correctness` | Single-item judges (bring your own loop) |
| `judge_retrieval_coverage` | Which key facts appear in the *retrieved* context (a retrieval ceiling) |
| `retrieval_metrics_for_query / aggregate_retrieval_metrics` | MRR / hit-rate / recall / NDCG from IDs |
| `check_cross_family / model_family` | Self-evaluation-bias guardrail |

Every judge takes an optional `prompt_template`, so you can swap in domain-specific
rubrics without touching the rest.

---

## Origin

CrossJudge is the evaluation harness from a Retrieval-Augmented Generation system for
Massachusetts tenant law, extracted and generalized into a standalone, bring-your-own-
outputs tool. The self-evaluation-bias finding that motivates it is from that project's
evaluation.

## License

MIT © En-Ping Su
