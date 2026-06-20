"""Multi-provider LLM client with retry logic."""

import asyncio
import os
import sys
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY"},
    "anthropic": {"base_url": "https://api.anthropic.com", "key_env": "ANTHROPIC_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": ""},
    "bedrock": {"base_url": "", "key_env": ""},
}

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


def detect_provider(model: str) -> str:
    """Auto-detect provider from model name."""
    if model.startswith(("claude", "anthropic")):
        return "anthropic"
    if model.startswith(("llama", "mistral", "gemma", "phi", "qwen")) and not model.startswith("meta."):
        return "ollama"
    if model.startswith(("us.", "meta.", "amazon.", "anthropic.")):
        return "bedrock"
    return "openai"


async def _retry(coro_fn, retries: int = MAX_RETRIES) -> Any:
    """Retry with exponential backoff on transient errors."""
    for attempt in range(retries):
        try:
            return await coro_fn()
        except Exception as e:
            err_str = str(e).lower()
            retriable = any(s in err_str for s in ["429", "500", "502", "503", "timeout", "rate"])
            if not retriable or attempt == retries - 1:
                raise
            delay = RETRY_BASE_DELAY * (2**attempt)
            await asyncio.sleep(delay)


async def call_provider(
    prompt: str,
    model: str,
    temperature: float,
    api_key: str = "",
    base_url: str = "",
    provider: str = "auto",
) -> str:
    """Route a call to the appropriate provider."""
    if not httpx:
        print("ERROR: httpx required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)

    if provider == "auto":
        provider = detect_provider(model)

    if provider == "anthropic":
        return await _call_anthropic(prompt, model, temperature, api_key, base_url)
    elif provider == "bedrock":
        return await _call_bedrock(prompt, model, temperature)
    else:
        # openai, ollama, any openai-compatible
        resolved_url = base_url or PROVIDER_CONFIGS.get(provider, {}).get("base_url", "https://api.openai.com/v1")
        resolved_key = api_key or os.environ.get(
            PROVIDER_CONFIGS.get(provider, {}).get("key_env", "OPENAI_API_KEY"), ""
        )
        return await _call_openai_compat(prompt, model, temperature, resolved_key, resolved_url)


async def call_provider_n(
    prompt: str,
    n: int,
    model: str,
    temperature: float,
    api_key: str = "",
    base_url: str = "",
    provider: str = "auto",
) -> list[str]:
    """Send prompt N times in parallel."""
    results = await asyncio.gather(*[
        call_provider(prompt, model, temperature, api_key, base_url, provider)
        for _ in range(n)
    ])
    return list(results)


async def _call_openai_compat(prompt: str, model: str, temperature: float, api_key: str, base_url: str) -> str:
    """OpenAI-compatible API call (OpenAI, Ollama, vLLM, etc.)."""
    async def _do():
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    return await _retry(_do)


async def _call_anthropic(prompt: str, model: str, temperature: float, api_key: str, base_url: str = "") -> str:
    """Anthropic Messages API call."""
    url = base_url or PROVIDER_CONFIGS["anthropic"]["base_url"]
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    async def _do():
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{url}/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
    return await _retry(_do)


async def _call_bedrock(prompt: str, model: str, temperature: float) -> str:
    """AWS Bedrock converse API call."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 required for Bedrock. Install with: pip install boto3")

    region = os.environ.get("AWS_REGION", "us-west-2")

    def _invoke():
        client = boto3.client("bedrock-runtime", region_name=region)
        resp = client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": temperature, "maxTokens": 4096},
        )
        return resp["output"]["message"]["content"][0]["text"]

    # Bedrock is sync, run in executor
    return await asyncio.get_event_loop().run_in_executor(None, _invoke)
