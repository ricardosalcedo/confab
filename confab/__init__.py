"""Confab — Inline hallucination confidence via self-consistency."""

__version__ = "0.3.0"

from confab.engine import extract_claims, score_claims, annotate_response, Claim
from confab.llm import call_llm, call_llm_n
from confab.db import save_check, get_history, clear_history
