"""Tests for confab.proxy."""

import json
import threading
import time
from http.client import HTTPConnection

import pytest


@pytest.fixture
def demo_proxy():
    """Start a demo legacy proxy server in a background thread."""
    port = 18923

    # Use legacy handler directly for testing without starlette dep issues
    import hashlib
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from confab.engine import score_claims
    from confab.llm import DEMO_RESPONSES

    # Manually start the legacy server in a thread

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            messages = body.get("messages", [])
            prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            responses = DEMO_RESPONSES[:3]
            claims = score_claims(responses[0], responses)
            result = {
                "id": f"confab-{hashlib.md5(prompt.encode()).hexdigest()[:8]}",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": responses[0]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "confab_metadata": {
                    "samples": 3,
                    "claims": [{"text": c.text, "confidence": c.confidence, "level": c.level} for c in claims],
                },
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, fmt, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
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

    def test_returns_raw_response(self, demo_proxy: int):
        """New behavior: response content is raw, not annotated."""
        conn = HTTPConnection("127.0.0.1", demo_proxy)
        body = json.dumps({
            "messages": [{"role": "user", "content": "Tell me about Python"}],
        })
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        data = json.loads(conn.getresponse().read())
        content = data["choices"][0]["message"]["content"]
        # Raw response should NOT have confidence icons
        assert "\U0001f7e2" not in content
        assert "[high" not in content

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
