"""OpenAI-compatible proxy that annotates responses with confidence."""

import asyncio
import hashlib
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

from confab.engine import score_claims, annotate_response
from confab.llm import call_llm_n, DEMO_RESPONSES


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        messages = body.get("messages", [])
        prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        model = body.get("model", self.server.conf["model"])
        n = self.server.conf["n"]
        temp = body.get("temperature", self.server.conf["temperature"])

        if self.server.conf["demo"]:
            responses = DEMO_RESPONSES[:n]
        else:
            responses = asyncio.run(call_llm_n(
                prompt, n, model, temp, self.server.conf["api_key"], self.server.conf["base_url"]
            ))

        claims = score_claims(responses[0], responses)
        result = {
            "id": f"confab-{hashlib.md5(prompt.encode()).hexdigest()[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": annotate_response(claims)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "confab_metadata": {"samples": n, "claims": [{"text": c.text, "confidence": c.confidence, "level": c.level} for c in claims]},
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, fmt, *args):
        print(f"[proxy] {args[0]}")


def serve(conf: dict):
    """Start the proxy server. `conf` keys: model, n, temperature, demo, api_key, base_url, port."""
    server = HTTPServer(("0.0.0.0", conf["port"]), _Handler)
    server.conf = conf

    print(f"🚀 Confab proxy listening on http://localhost:{conf['port']}")
    print(f"   Model: {conf['model']} | Samples: {conf['n']} | Temp: {conf['temperature']}")
    print(f"   Demo: {'ON' if conf['demo'] else 'OFF'}")
    print(f"\n   curl http://localhost:{conf['port']}/v1/chat/completions -d '{{...}}'")
    print(f"   Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Proxy stopped.")
        server.shutdown()
