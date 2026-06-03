#!/usr/bin/env python3
"""Confab — Inline hallucination confidence via self-consistency.

Send a prompt N times, compare claims across responses, annotate with confidence scores.
"""

import argparse
import asyncio
import json
import hashlib
import os
import re
import sqlite3
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, field
from pathlib import Path

# Optional deps — graceful fallback
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# --- Config ---

DEFAULT_N = 5
DEFAULT_TEMP = 0.8
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PORT = 8080
DB_PATH = Path(os.environ.get("CONFAB_DB", Path.home() / ".confab" / "history.db"))

# --- Database ---

def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.execute("""CREATE TABLE IF NOT EXISTS checks (
        id INTEGER PRIMARY KEY,
        timestamp TEXT DEFAULT (datetime('now')),
        command TEXT,
        prompt TEXT,
        model TEXT,
        samples INTEGER,
        elapsed REAL,
        claims_json TEXT,
        verdict TEXT
    )""")
    db.commit()
    return db


def save_check(command: str, prompt: str, model: str, samples: int, elapsed: float, claims: list, verdict: str = None):
    db = get_db()
    claims_json = json.dumps([{"text": c.text, "confidence": c.confidence, "level": c.level, "support": c.support_count, "total": c.total_samples} for c in claims]) if claims else "[]"
    db.execute("INSERT INTO checks (command, prompt, model, samples, elapsed, claims_json, verdict) VALUES (?,?,?,?,?,?,?)",
               (command, prompt, model, samples, elapsed, claims_json, verdict))
    db.commit()
    db.close()

# --- Data Types ---

@dataclass
class Claim:
    text: str
    confidence: float  # 0.0 - 1.0
    support_count: int
    total_samples: int

    @property
    def level(self) -> str:
        if self.confidence >= 0.8:
            return "high"
        elif self.confidence >= 0.5:
            return "medium"
        return "low"

@dataclass
class CheckResult:
    prompt: str
    annotated: str
    claims: list
    model: str
    samples: int
    elapsed: float

# --- Demo Mode ---

DEMO_RESPONSES = [
    "The Python programming language was created by Guido van Rossum and first released in 1991. It was named after Monty Python's Flying Circus. Python 3.0 was released in 2008. The language is dynamically typed and supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    "Python was created by Guido van Rossum and released in 1991. It was inspired by the BBC show Monty Python's Flying Circus. Python 3 was released in December 2008. It is a dynamically typed language that supports object-oriented, procedural, and functional programming styles.",
    "The Python language was developed by Guido van Rossum, with its first release in 1991. The name comes from Monty Python's Flying Circus. Python 3.0 came out in 2008. Python is dynamically typed and multi-paradigm, supporting OOP, procedural, and functional approaches.",
    "Python was created by Guido van Rossum in 1991. It's named after Monty Python. Python 3 was released in 2008. The language uses dynamic typing and supports multiple paradigms including object-oriented and functional programming.",
    "Guido van Rossum created Python, first released in 1991. The name derives from Monty Python's Flying Circus. Python 3.0 was released in October 2008. It features dynamic typing and supports procedural, object-oriented, and functional programming paradigms.",
]

DEMO_VERIFY_RESPONSES = [
    "This claim is SUPPORTED. Guido van Rossum created Python, first released in February 1991.",
    "This claim is SUPPORTED. Python 1.0 was released in January 1994, but the first version (0.9.0) was released in February 1991.",
    "This claim is SUPPORTED. Multiple reliable sources confirm this fact.",
]

# --- LLM Client ---

async def call_llm(prompt: str, model: str, temperature: float, api_key: str, base_url: str) -> str:
    if not HAS_HTTPX:
        print("ERROR: httpx required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def call_llm_n(prompt: str, n: int, model: str, temperature: float, api_key: str, base_url: str) -> list:
    done = 0
    async def _call_with_progress():
        nonlocal done
        result = await call_llm(prompt, model, temperature, api_key, base_url)
        done += 1
        print(f"\r   ⏳ {done}/{n} responses received", end="", flush=True)
        return result
    tasks = [_call_with_progress() for _ in range(n)]
    results = await asyncio.gather(*tasks)
    print()  # newline after progress
    return results

# --- Self-Consistency Engine ---

def extract_claims(text: str) -> list:
    """Split text into sentence-level claims. Handles prose, bullets, numbered lists."""
    claims = []
    # Split into lines first (handles bullet/numbered lists)
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        # Strip bullet/number prefix
        line = re.sub(r'^[\-\*•]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        if not line:
            continue
        # Split on sentence boundaries within the line
        # Handle colon-separated claims (e.g. "Created by: Guido van Rossum")
        if ':' in line and len(line.split(':')) == 2:
            claims.append(line.strip())
        else:
            sentences = re.split(r'(?<=[.!?])\s+', line)
            for s in sentences:
                s = s.strip()
                if len(s) > 10:
                    claims.append(s)
    return claims


def normalize_claim(claim: str) -> str:
    """Normalize for fuzzy comparison."""
    return re.sub(r'[^a-z0-9 ]', '', claim.lower()).strip()


def score_claims(primary_response: str, all_responses: list) -> list:
    """Score each claim in primary response by how many other responses support it."""
    claims = extract_claims(primary_response)
    scored = []
    
    # Normalize all responses for comparison
    other_texts = [normalize_claim(r) for r in all_responses[1:]]
    n = len(all_responses)
    
    for claim_text in claims:
        norm = normalize_claim(claim_text)
        # Extract meaningful content words (skip stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'was', 'were', 'are', 'it', 'its', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'by', 'that', 'this', 'from', 'on', 'as'}
        words = [w for w in norm.split() if w not in stopwords and len(w) > 2]
        
        if not words:
            words = norm.split()
        
        # Count how many other responses contain these key words
        support = 0
        for other in other_texts:
            matches = sum(1 for w in words if w in other)
            # If >50% of content words appear, consider it supported
            if matches / max(len(words), 1) > 0.5:
                support += 1
        
        # +1 for the primary response itself
        total_support = support + 1
        confidence = total_support / n
        
        scored.append(Claim(
            text=claim_text,
            confidence=confidence,
            support_count=total_support,
            total_samples=n,
        ))
    
    return scored


def annotate_response(claims: list) -> str:
    """Produce annotated markdown output."""
    lines = []
    for c in claims:
        icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}[c.level]
        lines.append(f"{icon} [{c.level} {c.confidence:.0%}] {c.text}")
    return '\n'.join(lines)

# --- Commands ---

async def cmd_check(args):
    """Run self-consistency check on a prompt."""
    prompt = args.prompt
    # Support stdin: confab check -
    if prompt == "-":
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("ERROR: No input received from stdin", file=sys.stderr)
            sys.exit(1)
    
    n = args.n
    model = args.model
    temperature = args.temperature
    
    print(f"⚡ Running {n} samples with {model} (temp={temperature})...\n")
    start = time.time()
    
    if args.demo:
        responses = DEMO_RESPONSES[:n]
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
            sys.exit(1)
        responses = await call_llm_n(prompt, n, model, temperature, api_key, args.base_url)
    
    elapsed = time.time() - start
    claims = score_claims(responses[0], responses)
    annotated = annotate_response(claims)
    
    print(f"📋 Prompt: {prompt}\n")
    print(f"{'─' * 60}")
    print(annotated)
    print(f"{'─' * 60}")
    print(f"\n⏱  {elapsed:.1f}s | {n} samples | {len(claims)} claims extracted")
    
    # Summary stats
    high = sum(1 for c in claims if c.level == "high")
    med = sum(1 for c in claims if c.level == "medium")
    low = sum(1 for c in claims if c.level == "low")
    print(f"📊 Confidence: {high} high, {med} medium, {low} low")
    
    # Save to history
    save_check("check", prompt, model, n, elapsed, claims)
    
    if args.json:
        result = {
            "prompt": prompt,
            "model": model,
            "samples": n,
            "elapsed": elapsed,
            "claims": [{"text": c.text, "confidence": c.confidence, "level": c.level, "support": c.support_count} for c in claims],
        }
        print(f"\n{json.dumps(result, indent=2)}")


async def cmd_verify(args):
    """Verify a single claim using cross-model check."""
    claim = args.claim
    if claim == "-":
        claim = sys.stdin.read().strip()
    model = args.model
    cross_model = getattr(args, 'cross_model', None)
    
    print(f"🔍 Verifying: {claim}\n")
    
    verify_prompt = f"""Verify whether the following claim is true or false. Be specific and cite your reasoning.

Claim: {claim}

Respond with:
- SUPPORTED: if the claim is factually correct
- REFUTED: if the claim is factually incorrect
- UNCERTAIN: if you cannot determine the truth

Then explain briefly."""
    
    start = time.time()
    
    if args.demo:
        response = DEMO_VERIFY_RESPONSES[0]
        cross_response = DEMO_VERIFY_RESPONSES[1] if cross_model else None
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
            sys.exit(1)
        response = await call_llm(verify_prompt, model, 0.0, api_key, args.base_url)
        cross_response = None
        if cross_model:
            print(f"   🔄 Cross-checking with {cross_model}...")
            cross_response = await call_llm(verify_prompt, cross_model, 0.0, api_key, args.base_url)
    
    elapsed = time.time() - start
    
    # Determine verdict
    def get_verdict(text):
        t = text.upper()
        if "SUPPORTED" in t: return "SUPPORTED"
        if "REFUTED" in t: return "REFUTED"
        return "UNCERTAIN"
    
    verdict = get_verdict(response)
    icon = {"SUPPORTED": "🟢", "REFUTED": "🔴", "UNCERTAIN": "🟡"}[verdict]
    
    print(f"{icon} Verdict ({model}): {verdict}")
    print(f"\n{response}")
    
    if cross_model and cross_response:
        cross_verdict = get_verdict(cross_response)
        cross_icon = {"SUPPORTED": "🟢", "REFUTED": "🔴", "UNCERTAIN": "🟡"}[cross_verdict]
        print(f"\n{cross_icon} Cross-check ({cross_model}): {cross_verdict}")
        print(f"\n{cross_response}")
        
        if verdict != cross_verdict:
            print(f"\n⚠️  Models DISAGREE — treat with caution")
        else:
            print(f"\n✅ Models AGREE")
        verdict = f"{verdict}/{cross_verdict}"
    
    # Save to history
    save_check("verify", claim, model, 1, elapsed, [], verdict)
    
    if args.json:
        result = {'claim': claim, 'verdict': verdict, 'explanation': response}
        if cross_model and cross_response:
            result['cross_model'] = cross_model
            result['cross_explanation'] = cross_response
        print(f"\n{json.dumps(result, indent=2)}")


def cmd_history(args):
    """Show past checks and verifications."""
    db = get_db()
    limit = args.limit if hasattr(args, 'limit') else 20
    
    if hasattr(args, 'clear') and args.clear:
        db.execute("DELETE FROM checks")
        db.commit()
        print("🗑  History cleared.")
        db.close()
        return
    
    rows = db.execute("SELECT id, timestamp, command, prompt, model, samples, elapsed, claims_json, verdict FROM checks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    db.close()
    
    if not rows:
        print("No history yet. Run `confab check` or `confab verify` first.")
        return
    
    print(f"📜 Last {len(rows)} entries (newest first)\n")
    for row in rows:
        id_, ts, cmd, prompt, model, samples, elapsed, claims_json, verdict = row
        prompt_short = prompt[:60] + "..." if len(prompt) > 60 else prompt
        
        if cmd == "check":
            claims = json.loads(claims_json)
            high = sum(1 for c in claims if c["level"] == "high")
            med = sum(1 for c in claims if c["level"] == "medium")
            low = sum(1 for c in claims if c["level"] == "low")
            print(f"  #{id_} [{ts}] check ({model}, n={samples}, {elapsed:.1f}s)")
            print(f"     {prompt_short}")
            print(f"     → {high}🟢 {med}🟡 {low}🔴")
        elif cmd == "verify":
            icon = {"SUPPORTED": "🟢", "REFUTED": "🔴", "UNCERTAIN": "🟡"}.get(verdict.split("/")[0] if verdict else "", "❓")
            print(f"  #{id_} [{ts}] verify ({model})")
            print(f"     {prompt_short}")
            print(f"     → {icon} {verdict}")
        print()


class ConfabProxyHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible proxy that adds confidence annotations."""
    
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        
        # Extract the last user message as the prompt
        messages = body.get("messages", [])
        prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        
        model = body.get("model", self.server.conf_model)
        n = self.server.conf_n
        temperature = body.get("temperature", self.server.conf_temperature)
        
        if self.server.conf_demo:
            responses = DEMO_RESPONSES[:n]
        else:
            api_key = self.server.conf_api_key
            base_url = self.server.conf_base_url
            responses = asyncio.run(call_llm_n(prompt, n, model, temperature, api_key, base_url))
        
        claims = score_claims(responses[0], responses)
        annotated = annotate_response(claims)
        
        # Return in OpenAI format
        result = {
            "id": f"confab-{hashlib.md5(prompt.encode()).hexdigest()[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": annotated},
                "finish_reason": "stop",
            }],
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
    
    def log_message(self, format, *args):
        print(f"[proxy] {args[0]}")


def cmd_proxy(args):
    """Start OpenAI-compatible proxy with confidence annotations."""
    port = args.port
    
    server = HTTPServer(("0.0.0.0", port), ConfabProxyHandler)
    server.conf_model = args.model
    server.conf_n = args.n
    server.conf_temperature = args.temperature
    server.conf_demo = args.demo
    server.conf_api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    server.conf_base_url = args.base_url
    
    if not args.demo and not server.conf_api_key:
        print("ERROR: Set OPENAI_API_KEY or pass --api-key (or use --demo)", file=sys.stderr)
        sys.exit(1)
    
    print(f"🚀 Confab proxy listening on http://localhost:{port}")
    print(f"   Model: {args.model} | Samples: {args.n} | Temp: {args.temperature}")
    print(f"   Demo mode: {'ON' if args.demo else 'OFF'}")
    print(f"\n   Usage: curl http://localhost:{port}/v1/chat/completions -d '{{...}}'")
    print(f"   Press Ctrl+C to stop.\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Proxy stopped.")
        server.shutdown()

# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        prog="confab",
        description="Inline hallucination confidence via self-consistency.",
    )
    parser.add_argument("--version", action="version", version="confab 0.2.0")
    
    # Global options
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", "-k", help="API key (or set OPENAI_API_KEY)")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("-n", type=int, default=DEFAULT_N, help=f"Number of samples (default: {DEFAULT_N})")
    parser.add_argument("--temperature", "-t", type=float, default=DEFAULT_TEMP, help=f"Sampling temperature (default: {DEFAULT_TEMP})")
    parser.add_argument("--demo", action="store_true", help="Use canned responses (no API key needed)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    sub = parser.add_subparsers(dest="command")
    
    # check
    p_check = sub.add_parser("check", help="Run self-consistency check on a prompt")
    p_check.add_argument("prompt", help="The prompt to check (use '-' for stdin)")
    
    # verify
    p_verify = sub.add_parser("verify", help="Verify a single claim")
    p_verify.add_argument("claim", help="The claim to verify (use '-' for stdin)")
    p_verify.add_argument("--cross-model", help="Cross-check with a second model (e.g. gpt-4o)")
    
    # proxy
    p_proxy = sub.add_parser("proxy", help="Start OpenAI-compatible proxy")
    p_proxy.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    
    # history
    p_hist = sub.add_parser("history", help="Review past checks and verifications")
    p_hist.add_argument("--limit", "-l", type=int, default=20, help="Number of entries to show")
    p_hist.add_argument("--clear", action="store_true", help="Clear all history")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "check":
        asyncio.run(cmd_check(args))
    elif args.command == "verify":
        asyncio.run(cmd_verify(args))
    elif args.command == "proxy":
        cmd_proxy(args)
    elif args.command == "history":
        cmd_history(args)


if __name__ == "__main__":
    main()
