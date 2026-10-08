"""Minimal CrossJudge example.

Scores four records (a baseline vs a RAG answer for each of two questions) with
a cross-family judge, prints the per-config summary, and saves a JSON report.

    export OPENROUTER_API_KEY=sk-or-...   # or put it in a .env file
    python examples/example_run.py

Expect the `baseline` config (no retrieved context, wrong answers) to score far
below `rag` on faithfulness and correctness — the gap retrieval is buying you.
"""

import json
from pathlib import Path

from crossjudge import evaluate, print_summary, save_report

HERE = Path(__file__).parent

if __name__ == "__main__":
    records = json.loads((HERE / "sample_records.json").read_text())

    result = evaluate(records, judge_model="anthropic/claude-sonnet-4")

    print_summary(result)
    out = save_report(result, HERE / "report.json")
    print(f"\nFull report written to {out}")
