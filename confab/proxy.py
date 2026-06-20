"""OpenAI-compatible proxy that annotates responses with confidence."""

import asyncio
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from confab.engine import annotate_response, score_claims
from confab.llm import DEMO_RESPONSES, call_llm_n


class ConfabHandler(BaseHTTPRequestHandler):
    """Handler for OpenAI-compatible proxy requests."""

    server: "ConfabServer"

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        body: dict[str, Any] = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        messages = body.get("messages", [])
        prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        model = body.get("model", self.server.conf["model"])
        n: int = self.server.conf["n"]
        temp = body.get("temperature", self.server.conf["temperature"])

        if self.server.conf["demo"]:
            responses = DEMO_RESPONSES[:n]
        else:
            responses = asyncio.run(call_llm_n(
                prompt, n, model, temp, self.server.conf["api_key"], self.server.conf["base_url"]
            ))

        claims = score_claims(responses[0], responses)
        result: dict[str, Any] = {
            "id": f"confab-{hashlib.md5(prompt.encode()).hexdigest()[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": annotate_response(claims)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "confab_metadata": {
                "samples": n,
                "claims": [{"text": c.text, "confidence": c.confidence, "level": c.level} for c in claims],
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[proxy] {args[0]}")


class ConfabServer(HTTPServer):
    """HTTPServer subclass carrying config."""

    conf: dict[str, Any]


def serve(conf: dict[str, Any]) -> None:
    """Start the proxy server."""
    server = ConfabServer(("0.0.0.0", conf["port"]), ConfabHandler)
    server.conf = conf

    print(f"\U0001f680 Confab proxy listening on http://localhost:{conf['port']}")
    print(f"   Model: {conf['model']} | Samples: {conf['n']} | Temp: {conf['temperature']}")
    print(f"   Demo: {'ON' if conf['demo'] else 'OFF'}")
    print(f"\n   curl http://localhost:{conf['port']}/v1/chat/completions -d '{{...}}'")
    print("   Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\U0001f44b Proxy stopped.")
        server.shutdown()
