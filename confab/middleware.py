"""SDK middleware: OpenAI client wrapper and LangChain callback."""

from __future__ import annotations

import asyncio
from typing import Any

from confab.scoring import score


class ConfabClient:
    """Drop-in wrapper around the OpenAI client that adds confidence scoring.

    Usage:
        from openai import OpenAI
        from confab.middleware import ConfabClient

        client = ConfabClient(OpenAI(), n=5, scoring="fast")
        result = client.chat("What is Python?")
        print(result.content)        # raw response
        print(result.claims)         # list of Claim objects
        print(result.confidence)     # average confidence 0-1
    """

    def __init__(
        self,
        client: Any,
        n: int = 5,
        temperature: float = 0.8,
        scoring: str = "fast",
    ):
        self._client = client
        self.n = n
        self.temperature = temperature
        self.scoring = scoring

    def chat(self, prompt: str, model: str = "gpt-4o-mini", **kwargs: Any) -> "ConfabResult":
        """Send prompt N times, score claims, return result with confidence."""
        messages = [{"role": "user", "content": prompt}]
        responses: list[str] = []
        for _ in range(self.n):
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                **kwargs,
            )
            responses.append(resp.choices[0].message.content)

        claims = asyncio.run(score(responses[0], responses, backend=self.scoring))
        return ConfabResult(content=responses[0], claims=claims)

    async def achat(self, prompt: str, model: str = "gpt-4o-mini", **kwargs: Any) -> "ConfabResult":
        """Async version using the async OpenAI client."""
        messages = [{"role": "user", "content": prompt}]
        responses: list[str] = []

        async def _call():
            resp = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                **kwargs,
            )
            return resp.choices[0].message.content

        import asyncio as aio

        responses = await aio.gather(*[_call() for _ in range(self.n)])
        claims = await score(list(responses)[0], list(responses), backend=self.scoring)
        return ConfabResult(content=list(responses)[0], claims=claims)


class ConfabResult:
    """Result from ConfabClient with response content and confidence data."""

    def __init__(self, content: str, claims: list):
        self.content = content
        self.claims = claims

    @property
    def confidence(self) -> float:
        """Average confidence across all claims."""
        if not self.claims:
            return 1.0
        return sum(c.confidence for c in self.claims) / len(self.claims)

    @property
    def high_confidence_claims(self) -> list:
        return [c for c in self.claims if c.level == "high"]

    @property
    def low_confidence_claims(self) -> list:
        return [c for c in self.claims if c.level == "low"]

    def __repr__(self) -> str:
        return f"ConfabResult(confidence={self.confidence:.0%}, claims={len(self.claims)})"


# --- LangChain Integration ---


class ConfabCallbackHandler:
    """LangChain callback that scores LLM outputs for hallucination confidence.

    Usage:
        from langchain_openai import ChatOpenAI
        from confab.middleware import ConfabCallbackHandler

        handler = ConfabCallbackHandler(n=5)
        llm = ChatOpenAI(callbacks=[handler])
        result = llm.invoke("What is Python?")
        print(handler.last_result)  # ConfabResult
    """

    def __init__(self, n: int = 5, scoring: str = "fast"):
        self.n = n
        self.scoring = scoring
        self.last_result: ConfabResult | None = None
        self._last_prompt: str = ""

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        """Capture the prompt for re-sampling."""
        self._last_prompt = prompts[0] if prompts else ""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Score the response using self-consistency (re-calls the LLM)."""
        # This is a simplified handler — full implementation would re-invoke the LLM
        # For now, it scores the single response against itself (baseline)
        text = ""
        if hasattr(response, "generations") and response.generations:
            gen = response.generations[0]
            if gen:
                text = gen[0].text if hasattr(gen[0], "text") else str(gen[0])

        if text:
            claims = asyncio.run(score(text, [text], backend=self.scoring))
            self.last_result = ConfabResult(content=text, claims=claims)
