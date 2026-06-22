# Confab

[![CI](https://github.com/ricardosalcedo/confab/actions/workflows/ci.yml/badge.svg)](https://github.com/ricardosalcedo/confab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)](https://github.com/ricardosalcedo/confab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**Add hallucination confidence to any LLM app. One URL change.**

Send a prompt to an LLM N times, compare claims across responses, and score consistency. Claims that appear in most responses get high confidence; inconsistent claims get flagged.

Works as a CLI, a drop-in proxy, or a Python/JS SDK.

## Install

```bash
pip install confab-llm

# With proxy support (starlette + uvicorn)
pip install confab-llm[proxy]

# Optional providers
pip install boto3  # for AWS Bedrock
```

## Quick Start

```bash
# Demo mode (no API key needed)
confab --demo check "Who invented Python and when?"

# With real model
confab check "What year was the first iPhone released?"

# More samples = more reliable
confab -n 10 check "Explain quantum entanglement"

# Use embeddings for smarter scoring
confab --scoring accurate check "When was Python created?"

# Verify a single claim
confab verify "The Great Wall of China is visible from space"
```

Output:
```
🟢 [high 100%] Python was created by Guido van Rossum in 1991.
🟡 [medium 60%] It was first released in February 1991.
🔴 [low 20%] The first version was 0.9.0.
```

## Proxy Mode

Drop-in replacement for the OpenAI API that adds confidence metadata:

```bash
confab proxy --port 8080
confab proxy --model claude-3-5-sonnet --provider anthropic
```

Then point your app at it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me about Mars"}]}'
```

Response includes the raw LLM response plus `confab_metadata`:

```json
{
  "choices": [{"message": {"content": "Mars is the fourth planet..."}}],
  "confab_metadata": {
    "samples": 5,
    "claims": [
      {"text": "Mars is the fourth planet from the Sun.", "confidence": 1.0, "level": "high"},
      {"text": "It has two moons.", "confidence": 0.8, "level": "high"},
      {"text": "Mars was discovered in 1610.", "confidence": 0.2, "level": "low"}
    ]
  }
}
```

Streaming is supported with `"stream": true` — metadata arrives in the final SSE chunk.

## Python SDK

### OpenAI Wrapper

```python
from openai import OpenAI
from confab.middleware import ConfabClient

client = ConfabClient(OpenAI(), n=5, scoring="fast")
result = client.chat("What is Python?")

print(result.content)              # raw response
print(result.confidence)           # average confidence 0-1
print(result.claims)               # list of Claim objects
print(result.low_confidence_claims) # claims to double-check
```

### As a Library

```python
from confab import score, score_claims, extract_claims

# Quick word-overlap scoring
claims = score_claims(primary_response, all_responses)

# Configurable backend
claims = await score(response, responses, backend="accurate")  # embeddings
claims = await score(response, responses, backend="nli")       # LLM judge
claims = await score(response, responses, backend="fast")      # word overlap
```

## JavaScript Client

```bash
npm install confab-client
```

```javascript
import { confab } from 'confab-client';

const client = confab({ baseUrl: 'http://localhost:8080' });
const result = await client.chat('What is Python?');

console.log(result.content);    // raw response
console.log(result.claims);     // [{text, confidence, level}]
console.log(result.confidence); // average 0-1

// Streaming
for await (const chunk of client.chatStream('Tell me about Mars')) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

## Commands

| Command | Description |
|---------|-------------|
| `confab check [prompt]` | Self-consistency check on a prompt |
| `confab verify [claim]` | Verify a single claim |
| `confab proxy` | Start OpenAI-compatible proxy with confidence |
| `confab history` | Review past checks |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model, -m` | gpt-4o-mini | Model to use |
| `--provider` | auto | Provider: auto, openai, anthropic, ollama, bedrock |
| `-n` | 5 | Number of consistency samples |
| `--temperature, -t` | 0.8 | Sampling temperature |
| `--scoring` | fast | Scoring backend: fast, accurate, nli |
| `--api-key, -k` | $OPENAI_API_KEY | API key |
| `--base-url` | OpenAI | API base URL |
| `--demo` | off | Use canned responses |
| `--json` | off | Machine-readable output |

## Scoring Backends

| Backend | Method | Cost | Best For |
|---------|--------|------|----------|
| `fast` | Word overlap | 0 extra calls | Quick checks, local models |
| `accurate` | Embedding similarity | 1 API call | Production, balanced cost/accuracy |
| `nli` | LLM-as-judge entailment | N×claims calls | High-stakes, max accuracy |

## Providers

Auto-detected from model name:

| Provider | Models | Config |
|----------|--------|--------|
| OpenAI | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| Anthropic | claude-3-5-sonnet, claude-3-opus | `ANTHROPIC_API_KEY` |
| Ollama | llama3, mistral, phi | Local (no key needed) |
| Bedrock | us.anthropic.*, amazon.* | AWS credentials |
| Any OpenAI-compatible | — | Set `--base-url` |

## How It Works

1. Send the same prompt N times with temperature > 0
2. Extract sentence-level claims from the primary response
3. Score each claim against all N responses using the chosen backend
4. Tag: 🟢 high (≥80%), 🟡 medium (≥50%), 🔴 low (<50%)

## Project Structure

```
confab/
├── engine.py       — claim extraction, annotation
├── scoring.py      — 3 backends: fast, accurate (embeddings), nli (LLM judge)
├── providers.py    — OpenAI, Anthropic, Ollama, Bedrock + auto-detect + retries
├── proxy.py        — ASGI proxy (starlette) + SSE streaming + legacy fallback
├── middleware.py   — Python SDK: ConfabClient wrapper, LangChain callback
├── llm.py          — base async LLM client + demo fixtures
├── db.py           — SQLite history
└── cli.py          — CLI commands + arg parsing
clients/
└── js/             — JavaScript/TypeScript client for the proxy
tests/              — 78+ tests, 90%+ coverage
```

## License

MIT
