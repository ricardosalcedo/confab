"""Tests for the ASGI proxy (starlette-based)."""

import json

import pytest

try:
    from starlette.testclient import TestClient

    from confab.proxy import _build_app

    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False


pytestmark = pytest.mark.skipif(not HAS_STARLETTE, reason="starlette not installed")

DEMO_CONF = {
    "model": "gpt-4o-mini",
    "n": 3,
    "temperature": 0.8,
    "demo": True,
    "api_key": "",
    "base_url": "",
    "provider": "auto",
}


@pytest.fixture
def client():
    app = _build_app(DEMO_CONF)
    return TestClient(app)


class TestASGIProxy:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_completions(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Tell me about Python"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "confab_metadata" in data
        # Raw response (not annotated)
        content = data["choices"][0]["message"]["content"]
        assert "Python" in content
        assert "[high" not in content

    def test_completions_metadata_structure(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Tell me about Python"}],
            },
        )
        data = resp.json()
        meta = data["confab_metadata"]
        assert meta["samples"] == 3
        for claim in meta["claims"]:
            assert "text" in claim
            assert "confidence" in claim
            assert "level" in claim

    def test_streaming(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Tell me about Python"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) >= 2  # at least one content chunk + [DONE]

        # Last data line before [DONE] should have confab_metadata
        last_chunk_line = data_lines[-2]
        last_chunk = json.loads(last_chunk_line.removeprefix("data: "))
        assert "confab_metadata" in last_chunk

        # Final line is [DONE]
        assert data_lines[-1] == "data: [DONE]"
