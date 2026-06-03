"""Self-consistency engine: extract claims, score confidence, annotate."""

import re
from dataclasses import dataclass

STOPWORDS = frozenset({'the', 'a', 'an', 'is', 'was', 'were', 'are', 'it', 'its',
                       'and', 'or', 'of', 'in', 'to', 'for', 'with', 'by', 'that',
                       'this', 'from', 'on', 'as'})


@dataclass
class Claim:
    """A single extracted claim with confidence score."""
    text: str
    confidence: float  # 0.0–1.0
    support_count: int
    total_samples: int

    @property
    def level(self) -> str:
        if self.confidence >= 0.8: return "high"
        if self.confidence >= 0.5: return "medium"
        return "low"


def extract_claims(text: str) -> list[str]:
    """Split text into sentence-level claims. Handles prose, bullets, numbered lists."""
    claims = []
    for line in text.strip().split('\n'):
        line = re.sub(r'^(?:[\-\*•]\s*|\d+[\.\)]\s*)', '', line.strip())
        if not line:
            continue
        if ':' in line and len(line.split(':')) == 2:
            claims.append(line)
        else:
            claims.extend(s.strip() for s in re.split(r'(?<=[.!?])\s+', line) if len(s.strip()) > 10)
    return claims


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


def _content_words(text: str) -> list[str]:
    """Extract meaningful words, skipping stopwords."""
    words = [w for w in _normalize(text).split() if w not in STOPWORDS and len(w) > 2]
    return words or _normalize(text).split()


def score_claims(primary_response: str, all_responses: list[str]) -> list[Claim]:
    """Score each claim by how consistently it appears across N responses."""
    claims = extract_claims(primary_response)
    other_texts = [_normalize(r) for r in all_responses[1:]]
    n = len(all_responses)
    scored = []

    for claim_text in claims:
        words = _content_words(claim_text)
        support = sum(
            1 for other in other_texts
            if sum(1 for w in words if w in other) / max(len(words), 1) > 0.5
        )
        scored.append(Claim(
            text=claim_text,
            confidence=(support + 1) / n,
            support_count=support + 1,
            total_samples=n,
        ))

    return scored


def annotate_response(claims: list[Claim]) -> str:
    """Format claims with confidence icons."""
    icons = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    return '\n'.join(f"{icons[c.level]} [{c.level} {c.confidence:.0%}] {c.text}" for c in claims)
