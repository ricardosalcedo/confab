"""OpenAI-compatible ASGI proxy that annotates responses with confidence."""

import hashlib
import json
import time
from typing import Any

from confab.engine import score_claims
from confab.llm import DEMO_RESPONSES
from confab.providers import call_provider_n

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False


def _build_app(conf: dict[str, Any]) -> Any:
    """Build the Starlette ASGI app."""

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": "0.3.0"})

    async def completions(request: Request) -> JSONResponse | StreamingResponse:
        body: dict[str, Any] = await request.json()
        messages = body.get("messages", [])
        prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        model = body.get("model", conf["model"])
        n: int = conf["n"]
        temp = body.get("temperature", conf["temperature"])
        stream = body.get("stream", False)

        if conf["demo"]:
            responses = DEMO_RESPONSES[:n]
        else:
            responses = await call_provider_n(
                prompt,
                n,
                model,
                temp,
                api_key=conf.get("api_key", ""),
                base_url=conf.get("base_url", ""),
                provider=conf.get("provider", "auto"),
            )

        claims = score_claims(responses[0], responses)
        metadata = {
            "samples": n,
            "claims": [{"text": c.text, "confidence": c.confidence, "level": c.level} for c in claims],
        }

        if stream:
            return _stream_response(responses[0], claims, metadata, model)

        # Return raw response + metadata (not annotated)
        result: dict[str, Any] = {
            "id": f"confab-{hashlib.md5(prompt.encode()).hexdigest()[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": responses[0]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "confab_metadata": metadata,
        }
        return JSONResponse(result)

    def _stream_response(raw: str, claims: list, metadata: dict, model: str) -> StreamingResponse:
        """SSE stream: chunks of raw response, then final metadata event."""

        async def event_generator():
            # Stream the raw response in chunks
            words = raw.split(" ")
            chunk_size = 5
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i : i + chunk_size])
                if i > 0:
                    chunk_text = " " + chunk_text
                chunk = {
                    "id": "confab-stream",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk with finish_reason
            final_chunk = {
                "id": "confab-stream",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "confab_metadata": metadata,
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", completions, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
        ]
    )
    return app


def serve(conf: dict[str, Any]) -> None:
    """Start the proxy server."""
    if not HAS_STARLETTE:
        # Fallback to legacy http.server proxy
        _serve_legacy(conf)
        return

    import uvicorn

    app = _build_app(conf)

    print(f"\U0001f680 Confab proxy listening on http://localhost:{conf['port']}")
    print(f"   Model: {conf['model']} | Samples: {conf['n']} | Temp: {conf['temperature']}")
    print(f"   Provider: {conf.get('provider', 'auto')} | Demo: {'ON' if conf['demo'] else 'OFF'}")
    print(f"\n   POST http://localhost:{conf['port']}/v1/chat/completions")
    print(f"   GET  http://localhost:{conf['port']}/health")
    print("   Ctrl+C to stop.\n")

    uvicorn.run(app, host="0.0.0.0", port=conf["port"], log_level="warning")


def _serve_legacy(conf: dict[str, Any]) -> None:
    """Legacy http.server fallback when starlette is not installed."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return

            import asyncio

            body: dict[str, Any] = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            messages = body.get("messages", [])
            prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            model = body.get("model", conf["model"])
            n: int = conf["n"]
            temp = body.get("temperature", conf["temperature"])

            if conf["demo"]:
                responses = DEMO_RESPONSES[:n]
            else:
                responses = asyncio.run(
                    call_provider_n(
                        prompt,
                        n,
                        model,
                        temp,
                        api_key=conf.get("api_key", ""),
                        base_url=conf.get("base_url", ""),
                    )
                )

            claims = score_claims(responses[0], responses)
            result: dict[str, Any] = {
                "id": f"confab-{hashlib.md5(prompt.encode()).hexdigest()[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": responses[0]},
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

    server = HTTPServer(("0.0.0.0", conf["port"]), Handler)
    print(f"\U0001f680 Confab proxy (legacy) on http://localhost:{conf['port']}")
    print("   Install starlette+uvicorn for async/streaming: pip install confab-llm[proxy]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
