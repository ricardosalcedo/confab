"""Tests for confab.proxy."""

import json
import threading
import time
from http.client import HTTPConnection

import pytest


@pytest.fixture
def demo_proxy():
    """Start a demo proxy server in a background thread and yield the port."""
    port = 18923
    conf = {
        "model": "gpt-4o-mini", "n": 3, "temperature": 0.8,
        "demo": True, "port": port, "api_key": "", "base_url": "",
    }

    from confab.proxy import ConfabHandler, ConfabServer
    server = ConfabServer(("127.0.0.1", port), ConfabHandler)
    server.conf = conf
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield port
    server.shutdown()


class TestProxy:
    def test_completions_endpoint(self, demo_proxy: int):
        conn = HTTPConnection("127.0.0.1", demo_proxy)
        body = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Tell me about Python"}],
        })
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 200

        data = json.loads(resp.read())
        assert data["object"] == "chat.completion"
        assert "confab_metadata" in data
        assert "claims" in data["confab_metadata"]
        assert len(data["confab_metadata"]["claims"]) > 0

    def test_claims_have_required_fields(self, demo_proxy: int):
        conn = HTTPConnection("127.0.0.1", demo_proxy)
        body = json.dumps({
            "messages": [{"role": "user", "content": "Tell me about Python"}],
        })
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        data = json.loads(conn.getresponse().read())

        for claim in data["confab_metadata"]["claims"]:
            assert "text" in claim
            assert "confidence" in claim
            assert "level" in claim
            assert claim["level"] in ("high", "medium", "low")
            assert 0.0 <= claim["confidence"] <= 1.0

    def test_404_on_wrong_path(self, demo_proxy: int):
        conn = HTTPConnection("127.0.0.1", demo_proxy)
        conn.request("POST", "/v1/models", body="{}", headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 404
