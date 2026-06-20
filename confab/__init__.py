"""Confab — Inline hallucination confidence via self-consistency."""

__version__ = "0.5.0"

from confab.engine import Claim, annotate_response, extract_claims, score_claims
from confab.llm import call_llm, call_llm_n
from confab.providers import call_provider, call_provider_n, detect_provider
from confab.scoring import score, score_embeddings, score_nli, score_word_overlap

__all__ = [
    "extract_claims", "score_claims", "annotate_response", "Claim",
    "call_llm", "call_llm_n",
    "call_provider", "call_provider_n", "detect_provider",
    "score", "score_word_overlap", "score_embeddings", "score_nli",
]
