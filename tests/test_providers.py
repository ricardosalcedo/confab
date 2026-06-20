"""Tests for confab.providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from confab.providers import _retry, call_provider, call_provider_n, detect_provider


class TestDetectProvider:
    def test_openai(self):
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("gpt-4o-mini") == "openai"

    def test_anthropic(self):
        assert detect_provider("claude-3-5-sonnet") == "anthropic"
        assert detect_provider("claude-3-opus") == "anthropic"

    def test_ollama(self):
        assert detect_provider("llama3") == "ollama"
        assert detect_provider("mistral") == "ollama"

    def test_bedrock(self):
        assert detect_provider("us.anthropic.claude-3") == "bedrock"
        assert detect_provider("amazon.titan-text") == "bedrock"

    def test_fallback_openai(self):
        assert detect_provider("some-unknown-model") == "openai"


class TestRetry:
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        fn = AsyncMock(return_value="ok")
        result = await _retry(fn)
        assert result == "ok"
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        fn = AsyncMock(side_effect=[Exception("429 rate limit"), "ok"])
        with patch("confab.providers.RETRY_BASE_DELAY", 0.01):
            result = await _retry(fn)
        assert result == "ok"
        assert fn.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retriable(self):
        fn = AsyncMock(side_effect=Exception("401 unauthorized"))
        with pytest.raises(Exception, match="401"):
            await _retry(fn)
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        fn = AsyncMock(side_effect=Exception("503 service unavailable"))
        with patch("confab.providers.RETRY_BASE_DELAY", 0.01), pytest.raises(Exception, match="503"):
            await _retry(fn, retries=2)
        assert fn.call_count == 2


class TestCallProvider:
    @pytest.mark.asyncio
    async def test_openai_compat(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "response"}}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("confab.providers.httpx.AsyncClient", return_value=mock_client):
            result = await call_provider("test", "gpt-4o-mini", 0.8, "key", "https://api.openai.com/v1")
        assert result == "response"

    @pytest.mark.asyncio
    async def test_anthropic(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"text": "anthropic response"}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("confab.providers.httpx.AsyncClient", return_value=mock_client):
            result = await call_provider("test", "claude-3-5-sonnet", 0.8, "key", provider="anthropic")
        assert result == "anthropic response"

    @pytest.mark.asyncio
    async def test_call_provider_n(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "r"}}]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("confab.providers.httpx.AsyncClient", return_value=mock_client):
            results = await call_provider_n("test", 3, "gpt-4o-mini", 0.8, "key", "https://api.openai.com/v1")
        assert len(results) == 3
