"""Tests for confab.llm."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from confab.llm import DEMO_RESPONSES, call_llm, call_llm_n


class TestCallLLM:
    @pytest.mark.asyncio
    async def test_call_llm_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "response text"}}]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("confab.llm.httpx.AsyncClient", return_value=mock_client):
            result = await call_llm("test", "gpt-4o-mini", 0.8, "key", "https://api.openai.com/v1")

        assert result == "response text"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_n_returns_n_results(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "response"}}]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("confab.llm.httpx.AsyncClient", return_value=mock_client):
            results = await call_llm_n("test", 3, "gpt-4o-mini", 0.8, "key", "https://api.openai.com/v1")

        assert len(results) == 3
        assert all(r == "response" for r in results)


class TestDemoFixtures:
    def test_demo_responses_not_empty(self):
        assert len(DEMO_RESPONSES) >= 5
        assert all(isinstance(r, str) for r in DEMO_RESPONSES)
