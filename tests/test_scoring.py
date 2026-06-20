"""Tests for confab.scoring backends."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from confab.scoring import (
    _cosine_similarity,
    score,
    score_embeddings,
    score_nli,
    score_word_overlap,
)


class TestWordOverlap:
    def test_high_confidence(self):
        primary = "Python was created by Guido van Rossum in 1991."
        responses = [primary] * 5
        claims = score_word_overlap(primary, responses)
        assert claims[0].confidence == 1.0

    def test_low_confidence(self):
        primary = "Xylophone quantum blockchain synergy."
        responses = [primary, "Totally different.", "Nothing related.", "Unrelated text."]
        claims = score_word_overlap(primary, responses)
        assert claims[0].confidence < 0.5

    def test_empty(self):
        assert score_word_overlap("", ["", ""]) == []


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 0.0, 1.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal(self):
        assert abs(_cosine_similarity([1, 0], [0, 1])) < 1e-6

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 1]) == 0.0


class TestScoreEmbeddings:
    @pytest.mark.asyncio
    async def test_with_mocked_api(self):
        # Mock embeddings: claim similar to response 1, different from response 2
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0, 0.0]},  # claim
                {"index": 1, "embedding": [0.9, 0.1, 0.0]},  # response 2 (similar)
                {"index": 2, "embedding": [0.0, 0.0, 1.0]},  # response 3 (different)
            ]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        primary = "Python was created in 1991."
        responses = [primary, "Python appeared in 1991.", "Cats are cute animals."]

        with patch("confab.scoring.httpx.AsyncClient", return_value=mock_client):
            claims = await score_embeddings(primary, responses, api_key="key")

        assert len(claims) == 1
        # One response is similar (>0.5), one is not
        assert claims[0].support_count >= 1

    @pytest.mark.asyncio
    async def test_empty_input(self):
        claims = await score_embeddings("", ["", ""], api_key="key")
        assert claims == []


class TestScoreNLI:
    @pytest.mark.asyncio
    async def test_with_mocked_llm(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        # First call: YES, second call: NO
        mock_resp.json.side_effect = [
            {"choices": [{"message": {"content": "YES"}}]},
            {"choices": [{"message": {"content": "NO"}}]},
        ]

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        primary = "Python was created in 1991."
        responses = [primary, "Python appeared in 1991.", "Cats are animals."]

        with patch("confab.scoring.httpx.AsyncClient", return_value=mock_client):
            claims = await score_nli(primary, responses, api_key="key")

        assert len(claims) == 1
        # 1 YES + 1 (self) out of 3 total
        assert claims[0].support_count == 2

    @pytest.mark.asyncio
    async def test_empty_input(self):
        claims = await score_nli("", ["", ""], api_key="key")
        assert claims == []


class TestUnifiedScore:
    @pytest.mark.asyncio
    async def test_fast_backend(self):
        primary = "Python was created in 1991."
        responses = [primary] * 3
        claims = await score(primary, responses, backend="fast")
        assert len(claims) == 1
        assert claims[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown scoring backend"):
            await score("test", ["test"], backend="invalid")

    @pytest.mark.asyncio
    async def test_accurate_backend_calls_embeddings(self):
        with patch("confab.scoring.score_embeddings", new_callable=AsyncMock, return_value=[]) as mock:
            await score("test", ["test", "other"], backend="accurate", api_key="k")
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_nli_backend_calls_nli(self):
        with patch("confab.scoring.score_nli", new_callable=AsyncMock, return_value=[]) as mock:
            await score("test", ["test", "other"], backend="nli", api_key="k")
            mock.assert_called_once()
