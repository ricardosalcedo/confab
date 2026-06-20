"""CLI: argparse + command dispatch."""

import argparse
import asyncio
import json
import os
import sys
import time

import confab
from confab.db import clear_history, get_history, save_check
from confab.engine import annotate_response, score_claims
from confab.llm import DEMO_RESPONSES, DEMO_VERIFY_RESPONSES, call_llm, call_llm_n
from confab.proxy import serve as proxy_serve

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_N = 5
DEFAULT_TEMP = 0.8
DEFAULT_PORT = 8080


def _read_input(value: str) -> str:
    """Read from stdin if value is '-', otherwise return as-is."""
    if value == "-":
        text = sys.stdin.read().strip()
        if not text:
            print("ERROR: No input from stdin", file=sys.stderr)
            sys.exit(1)
        return text
    return value


def _require_api_key(args) -> str:
    key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: Set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)
    return key


def _get_verdict(text: str) -> str:
    t = text.upper()
    if "SUPPORTED" in t:
        return "SUPPORTED"
    if "REFUTED" in t:
        return "REFUTED"
    return "UNCERTAIN"


# --- Commands ---

async def cmd_check(args):
    """Run self-consistency check."""
    prompt = _read_input(args.prompt)
    n, model, temp = args.n, args.model, args.temperature

    print(f"⚡ Running {n} samples with {model} (temp={temp})...\n")
    start = time.time()

    if args.demo:
        responses = DEMO_RESPONSES[:n]
    else:
        responses = await call_llm_n(prompt, n, model, temp, _require_api_key(args), args.base_url)

    elapsed = time.time() - start
    claims = score_claims(responses[0], responses)
    annotated = annotate_response(claims)

    print(f"📋 Prompt: {prompt}\n")
    print(f"{'─' * 60}")
    print(annotated)
    print(f"{'─' * 60}")
    high = sum(1 for c in claims if c.level == "high")
    med = sum(1 for c in claims if c.level == "medium")
    low = sum(1 for c in claims if c.level == "low")
    print(f"\n⏱  {elapsed:.1f}s | {n} samples | {len(claims)} claims")
    print(f"📊 Confidence: {high} high, {med} medium, {low} low")

    save_check("check", prompt, model, n, elapsed, claims)

    if args.json:
        data = {
            "prompt": prompt, "model": model, "samples": n, "elapsed": elapsed,
            "claims": [
                {"text": c.text, "confidence": c.confidence, "level": c.level, "support": c.support_count}
                for c in claims
            ],
        }
        print(f"\n{json.dumps(data, indent=2)}")


async def cmd_verify(args):
    """Verify a single claim, optionally with cross-model check."""
    claim = _read_input(args.claim)
    model = args.model
    cross_model = getattr(args, "cross_model", None)

    print(f"🔍 Verifying: {claim}\n")

    verify_prompt = (
        f"Verify whether the following claim is true or false. Be specific.\n\n"
        f"Claim: {claim}\n\n"
        f"Respond with SUPPORTED, REFUTED, or UNCERTAIN, then explain briefly."
    )

    start = time.time()

    if args.demo:
        response = DEMO_VERIFY_RESPONSES[0]
        cross_response = DEMO_VERIFY_RESPONSES[1] if cross_model else None
    else:
        api_key = _require_api_key(args)
        response = await call_llm(verify_prompt, model, 0.0, api_key, args.base_url)
        cross_response = None
        if cross_model:
            print(f"   🔄 Cross-checking with {cross_model}...")
            cross_response = await call_llm(verify_prompt, cross_model, 0.0, api_key, args.base_url)

    elapsed = time.time() - start
    verdict = _get_verdict(response)
    icons = {"SUPPORTED": "🟢", "REFUTED": "🔴", "UNCERTAIN": "🟡"}

    print(f"{icons[verdict]} Verdict ({model}): {verdict}\n\n{response}")

    if cross_model and cross_response:
        cv = _get_verdict(cross_response)
        print(f"\n{icons[cv]} Cross-check ({cross_model}): {cv}\n\n{cross_response}")
        print(f"\n{'⚠️  Models DISAGREE' if verdict != cv else '✅ Models AGREE'}")
        verdict = f"{verdict}/{cv}"

    save_check("verify", claim, model, 1, elapsed, [], verdict)

    if args.json:
        r = {"claim": claim, "verdict": verdict, "explanation": response}
        if cross_response:
            r["cross_model"] = cross_model
            r["cross_explanation"] = cross_response
        print(f"\n{json.dumps(r, indent=2)}")


def cmd_history(args):
    """Show or clear history."""
    if getattr(args, "clear", False):
        clear_history()
        print("🗑  History cleared.")
        return

    rows = get_history(getattr(args, "limit", 20))
    if not rows:
        print("No history yet. Run `confab check` or `confab verify` first.")
        return

    print(f"📜 Last {len(rows)} entries (newest first)\n")
    for r in rows:
        prompt_short = r["prompt"][:60] + ("..." if len(r["prompt"]) > 60 else "")
        if r["command"] == "check":
            claims = json.loads(r["claims_json"])
            h = sum(1 for c in claims if c["level"] == "high")
            m = sum(1 for c in claims if c["level"] == "medium")
            lo = sum(1 for c in claims if c["level"] == "low")
            print(f"  #{r['id']} [{r['timestamp']}] check ({r['model']}, n={r['samples']}, {r['elapsed']:.1f}s)")
            print(f"     {prompt_short}")
            print(f"     → {h}🟢 {m}🟡 {lo}🔴\n")
        else:
            v = (r["verdict"] or "").split("/")[0]
            icon = {"SUPPORTED": "🟢", "REFUTED": "🔴", "UNCERTAIN": "🟡"}.get(v, "❓")
            print(f"  #{r['id']} [{r['timestamp']}] verify ({r['model']})")
            print(f"     {prompt_short}")
            print(f"     → {icon} {r['verdict']}\n")


def cmd_proxy(args):
    """Start proxy server."""
    conf = {
        "model": args.model, "n": args.n, "temperature": args.temperature,
        "demo": args.demo, "port": args.port,
        "api_key": args.api_key or os.environ.get("OPENAI_API_KEY", ""),
        "base_url": args.base_url,
    }
    if not conf["demo"] and not conf["api_key"]:
        print("ERROR: Set OPENAI_API_KEY or pass --api-key (or use --demo)", file=sys.stderr)
        sys.exit(1)
    proxy_serve(conf)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(prog="confab", description="Inline hallucination confidence via self-consistency.")
    parser.add_argument("--version", action="version", version=f"confab {confab.__version__}")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", "-k", help="API key (or set OPENAI_API_KEY)")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("-n", type=int, default=DEFAULT_N, help=f"Samples (default: {DEFAULT_N})")
    parser.add_argument("--temperature", "-t", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--demo", action="store_true", help="Canned responses, no API key needed")
    parser.add_argument("--json", action="store_true", help="JSON output")

    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("check", help="Self-consistency check on a prompt")
    p.add_argument("prompt", help="Prompt to check (use '-' for stdin)")

    p = sub.add_parser("verify", help="Verify a single claim")
    p.add_argument("claim", help="Claim to verify (use '-' for stdin)")
    p.add_argument("--cross-model", help="Second model for cross-verification")

    p = sub.add_parser("proxy", help="OpenAI-compatible proxy with confidence")
    p.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)

    p = sub.add_parser("history", help="Review past checks")
    p.add_argument("--limit", "-l", type=int, default=20)
    p.add_argument("--clear", action="store_true")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {"check": cmd_check, "verify": cmd_verify, "proxy": cmd_proxy, "history": cmd_history}
    handler = dispatch[args.command]

    if asyncio.iscoroutinefunction(handler):
        asyncio.run(handler(args))
    else:
        handler(args)
