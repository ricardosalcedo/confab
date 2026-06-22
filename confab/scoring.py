"""Scoring backends: word-overlap (fast), embeddings (accurate), NLI (verify)."""

import asyncio
import os
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from confab.engine import Claim, _content_words, _normalize, extract_claims

# --- Word Overlap (default, zero-dep) ---


def score_word_overlap(primary_response: str, all_responses: list[str]) -> list[Claim]:
    """Original fast scoring: content-word overlap ratio."""
    claims = extract_claims(primary_response)
    other_texts = [_normalize(r) for r in all_responses[1:]]
    n = len(all_responses)
    scored: list[Claim] = []

    for claim_text in claims:
        words = _content_words(claim_text)
        support = sum(1 for other in other_texts if sum(1 for w in words if w in other) / max(len(words), 1) > 0.5)
        scored.append(
            Claim(
                text=claim_text,
                confidence=(support + 1) / n,
                support_count=support + 1,
                total_samples=n,
            )
        )
    return scored


# --- Embedding Similarity ---


async def _get_embeddings(texts: list[str], api_key: str, base_url: str, model: str) -> list[list[float]]:
    """Get embeddings from OpenAI-compatible API."""
    if not httpx:
        raise RuntimeError("httpx required for embedding scoring")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base_url}/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


async def score_embeddings(
    primary_response: str,
    all_responses: list[str],
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    model: str = "text-embedding-3-small",
) -> list[Claim]:
    """Score claims using embedding cosine similarity against other responses."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    claims_text = extract_claims(primary_response)
    if not claims_text:
        return []

    n = len(all_responses)
    # Get embeddings for claims and other responses
    all_texts = claims_text + all_responses[1:]
    embeddings = await _get_embeddings(all_texts, api_key, base_url, model)

    claim_embeddings = embeddings[: len(claims_text)]
    response_embeddings = embeddings[len(claims_text) :]

    scored: list[Claim] = []
    for i, claim_text in enumerate(claims_text):
        # Average similarity of this claim against all other responses
        sims = [_cosine_similarity(claim_embeddings[i], re) for re in response_embeddings]
        # Threshold: similarity > 0.5 counts as support
        support = sum(1 for s in sims if s > 0.5)
        confidence = (support + 1) / n
        scored.append(
            Claim(
                text=claim_text,
                confidence=confidence,
                support_count=support + 1,
                total_samples=n,
            )
        )
    return scored


# --- NLI Verification ---


async def score_nli(
    primary_response: str,
    all_responses: list[str],
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    judge_model: str = "gpt-4o-mini",
) -> list[Claim]:
    """Score claims using NLI: ask LLM if each response entails each claim."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    claims_text = extract_claims(primary_response)
    if not claims_text:
        return []

    n = len(all_responses)
    other_responses = all_responses[1:]

    async def _check_entailment(claim: str, response: str) -> bool:
        """Ask if response entails claim."""
        prompt = (
            f"Does the following text support this claim? Answer only YES or NO.\n\nText: {response}\n\nClaim: {claim}"
        )
        if not httpx:
            raise RuntimeError("httpx required for NLI scoring")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": judge_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 5,
                },
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
            return "YES" in answer

    scored: list[Claim] = []
    for claim_text in claims_text:
        tasks = [_check_entailment(claim_text, r) for r in other_responses]
        results = await asyncio.gather(*tasks)
        support = sum(results)
        scored.append(
            Claim(
                text=claim_text,
                confidence=(support + 1) / n,
                support_count=support + 1,
                total_samples=n,
            )
        )
    return scored


# --- Unified scoring function ---

BACKENDS = {"fast": "word_overlap", "accurate": "embeddings", "nli": "nli"}


async def score(
    primary_response: str,
    all_responses: list[str],
    backend: str = "fast",
    **kwargs: Any,
) -> list[Claim]:
    """Unified scoring with configurable backend.

    Backends:
        fast — word-overlap (zero API calls, default)
        accurate — embedding cosine similarity (1 API call)
        nli — LLM-as-judge entailment (N*claims API calls)
    """
    if backend == "fast":
        return score_word_overlap(primary_response, all_responses)
    elif backend == "accurate":
        return await score_embeddings(primary_response, all_responses, **kwargs)
    elif backend == "nli":
        return await score_nli(primary_response, all_responses, **kwargs)
    else:
        raise ValueError(f"Unknown scoring backend: {backend}. Use: fast, accurate, nli")
