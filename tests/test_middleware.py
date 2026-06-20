"""Tests for confab.middleware."""

from unittest.mock import MagicMock, patch

from confab.middleware import ConfabClient, ConfabResult


class TestConfabResult:
    def test_confidence_with_claims(self):
        from confab.engine import Claim

        claims = [
            Claim(text="a", confidence=0.9, support_count=4, total_samples=5),
            Claim(text="b", confidence=0.5, support_count=2, total_samples=5),
        ]
        r = ConfabResult(content="test", claims=claims)
        assert r.confidence == 0.7
        assert len(r.high_confidence_claims) == 1
        assert len(r.low_confidence_claims) == 0

    def test_confidence_no_claims(self):
        r = ConfabResult(content="test", claims=[])
        assert r.confidence == 1.0

    def test_repr(self):
        r = ConfabResult(content="test", claims=[])
        assert "ConfabResult" in repr(r)


class TestConfabClient:
    def test_chat(self):
        mock_openai = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Python was created in 1991."))]
        mock_openai.chat.completions.create.return_value = mock_resp

        client = ConfabClient(mock_openai, n=3, scoring="fast")
        result = client.chat("What is Python?")

        assert isinstance(result, ConfabResult)
        assert result.content == "Python was created in 1991."
        assert mock_openai.chat.completions.create.call_count == 3


class TestConfabCallbackHandler:
    def test_on_llm_start(self):
        from confab.middleware import ConfabCallbackHandler

        handler = ConfabCallbackHandler()
        handler.on_llm_start({}, ["test prompt"])
        assert handler._last_prompt == "test prompt"

    def test_on_llm_end(self):
        from confab.middleware import ConfabCallbackHandler

        handler = ConfabCallbackHandler()

        mock_response = MagicMock()
        mock_gen = MagicMock()
        mock_gen.text = "Python was created in 1991."
        mock_response.generations = [[mock_gen]]

        handler.on_llm_end(mock_response)
        assert handler.last_result is not None
        assert handler.last_result.content == "Python was created in 1991."
