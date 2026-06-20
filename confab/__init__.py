"""Confab — Inline hallucination confidence via self-consistency."""

__version__ = "0.4.0"

from confab.engine import Claim, annotate_response, extract_claims, score_claims
from confab.llm import call_llm, call_llm_n
from confab.providers import call_provider, call_provider_n, detect_provider

__all__ = [
    "extract_claims", "score_claims", "annotate_response", "Claim",
    "call_llm", "call_llm_n",
    "call_provider", "call_provider_n", "detect_provider",
]
