"""LLM-judge prompts.

Binary YES/NO for faithfulness and relevancy; per-fact YES/NO for correctness.
Binary rubrics keep judges consistent and cheap, and make the parsing robust.
These are deliberately domain-agnostic — override them if your task needs it.
"""

FAITHFULNESS_PROMPT = """You are evaluating whether a response is faithful to the provided source context.

A response is FAITHFUL if every claim it makes is supported by the source context. It should not contain information that contradicts or goes beyond what the sources provide. If no source context is provided (baseline mode), evaluate whether the response avoids fabricating specific facts, citations, figures, names, or URLs that do not exist.

QUESTION: {question}

SOURCE CONTEXT:
{context}

RESPONSE: {response}

Is this response faithful to the source context? Answer with ONLY "YES" or "NO", followed by a brief explanation.
Format: YES/NO: <explanation>"""

RELEVANCY_PROMPT = """You are evaluating whether a response is relevant to the question asked.

A response is RELEVANT if it directly addresses the question and provides useful information toward answering it. A response that declines to answer but appropriately explains why and suggests resources is still relevant. A response is NOT relevant if it discusses unrelated topics or fails to address the core question.

QUESTION: {question}

RESPONSE: {response}

Is this response relevant to the question? Answer with ONLY "YES" or "NO", followed by a brief explanation.
Format: YES/NO: <explanation>"""

CORRECTNESS_PROMPT = """You are evaluating whether a response contains specific expected facts.

QUESTION: {question}

RESPONSE: {response}

For each expected fact below, determine whether it is present (supported) in the response.
A fact counts as present if the response conveys the same information, even if worded differently.
A fact counts as absent if the response does not mention it or contradicts it.

{numbered_facts}

For each fact, answer on its own line in this exact format:
FACT 1: YES or NO — brief explanation
FACT 2: YES or NO — brief explanation
...and so on for all facts."""
