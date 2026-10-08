"""OpenRouter client and model-family helpers.

CrossJudge routes every judge call through OpenRouter so you can use a judge
model from a *different* provider family than the model that generated the
answers. Mixing families is the whole point: a model tends to grade text from
its own family too generously (self-evaluation bias), so an independent judge
gives you honest numbers. See ``check_cross_family``.
"""

from __future__ import annotations

import os
import warnings

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Sensible default: a strong judge from a family you are unlikely to also be
# generating with. Override per-run if your generators are Anthropic models.
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4"


def get_openrouter_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at OpenRouter.

    Reads ``OPENROUTER_API_KEY`` from the environment (or a local ``.env``).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Add it to your environment or a .env file. "
            "Get a key at https://openrouter.ai/keys"
        )
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def model_family(model: str) -> str:
    """Provider family for an OpenRouter model id, e.g. ``openai/gpt-4o`` -> ``openai``."""
    return model.split("/", 1)[0].strip().lower() if model else ""


def check_cross_family(judge_model: str, generator_models: list[str]) -> list[str]:
    """Warn for any generator that shares the judge's provider family.

    Returns the list of colliding generator models (empty == all cross-family).
    A collision does not stop the run; it just means those scores carry
    self-evaluation bias and should be read with suspicion.
    """
    judge_fam = model_family(judge_model)
    collisions = sorted(
        {m for m in generator_models if m and model_family(m) == judge_fam}
    )
    if collisions:
        warnings.warn(
            f"Judge '{judge_model}' shares the '{judge_fam}' family with "
            f"{collisions}. These scores risk self-evaluation bias — pick a "
            f"judge from a different provider family for honest numbers.",
            stacklevel=2,
        )
    return collisions
