"""Confab — Inline hallucination confidence via self-consistency."""

__version__ = "0.3.0"

from confab.db import clear_history, get_history, save_check
from confab.engine import Claim, annotate_response, extract_claims, score_claims
from confab.llm import call_llm, call_llm_n

__all__ = [
    "extract_claims", "score_claims", "annotate_response", "Claim",
    "call_llm", "call_llm_n",
    "save_check", "get_history", "clear_history",
]
