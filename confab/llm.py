"""LLM client and demo fixtures."""

import asyncio
import sys

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

# Demo responses for --demo mode (no API key needed)
DEMO_RESPONSES: list[str] = [
    "The Python programming language was created by Guido van Rossum and first released in 1991. It was named after Monty Python's Flying Circus. Python 3.0 was released in 2008. The language is dynamically typed and supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    "Python was created by Guido van Rossum and released in 1991. It was inspired by the BBC show Monty Python's Flying Circus. Python 3 was released in December 2008. It is a dynamically typed language that supports object-oriented, procedural, and functional programming styles.",
    "The Python language was developed by Guido van Rossum, with its first release in 1991. The name comes from Monty Python's Flying Circus. Python 3.0 came out in 2008. Python is dynamically typed and multi-paradigm, supporting OOP, procedural, and functional approaches.",
    "Python was created by Guido van Rossum in 1991. It's named after Monty Python. Python 3 was released in 2008. The language uses dynamic typing and supports multiple paradigms including object-oriented and functional programming.",
    "Guido van Rossum created Python, first released in 1991. The name derives from Monty Python's Flying Circus. Python 3.0 was released in October 2008. It features dynamic typing and supports procedural, object-oriented, and functional programming paradigms.",
]

DEMO_VERIFY_RESPONSES: list[str] = [
    "This claim is SUPPORTED. Guido van Rossum created Python, first released in February 1991.",
    "This claim is SUPPORTED. Python 1.0 was released in January 1994, but the first version (0.9.0) was released in February 1991.",
]


async def call_llm(prompt: str, model: str, temperature: float, api_key: str, base_url: str) -> str:
    """Send a single chat completion request."""
    if not httpx:
        print("ERROR: httpx required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature},
        )
        resp.raise_for_status()
        data: dict = resp.json()
        return data["choices"][0]["message"]["content"]


async def call_llm_n(prompt: str, n: int, model: str, temperature: float, api_key: str, base_url: str) -> list[str]:
    """Send prompt N times in parallel with progress indicator."""
    done = 0

    async def _call() -> str:
        nonlocal done
        result = await call_llm(prompt, model, temperature, api_key, base_url)
        done += 1
        print(f"\r   \u23f3 {done}/{n} responses received", end="", flush=True)
        return result

    results = await asyncio.gather(*[_call() for _ in range(n)])
    print()
    return list(results)
