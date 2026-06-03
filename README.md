# Confab

Inline hallucination confidence for LLMs via self-consistency.

Send a prompt to an LLM N times, compare claims across responses, and annotate with confidence scores. Claims that appear consistently get high confidence; claims that vary get low confidence.

## Install

```bash
pip install httpx  # only runtime dependency

# Or run in --demo mode with zero deps
python3 confab.py --demo check "your prompt"
```

## Usage

### Check a prompt

```bash
# Demo mode (no API key needed)
confab --demo check "Who invented Python and when?"

# With a real model
confab check "What year was the first iPhone released?"

# More samples = more reliable
confab -n 10 check "Explain quantum entanglement"

# Pipe from stdin
echo "Is the sky blue?" | confab check -
cat question.txt | confab check -
```

Output:
```
🟢 [high 100%] Python was created by Guido van Rossum in 1991.
🟡 [medium 60%] It was first released in February 1991.
🔴 [low 20%] The first version was 0.9.0.
```

### Verify a single claim

```bash
confab verify "The Great Wall of China is visible from space"

# Cross-model verification (ask two models, compare verdicts)
confab verify "Python was released in 1991" --cross-model gpt-4o

# From stdin
echo "Earth is flat" | confab verify -
```

### History

All checks and verifications are stored locally in `~/.confab/history.db`:

```bash
confab history            # review past checks
confab history --limit 5  # show last 5
confab history --clear    # delete all history
```

### Proxy mode

Drop-in replacement for OpenAI API that adds confidence annotations:

```bash
confab proxy --port 8080

# Then point your app at it:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me about Mars"}]}'
```

Response includes `confab_metadata` with per-claim confidence scores.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model, -m` | gpt-4o-mini | Model to use |
| `-n` | 5 | Number of consistency samples |
| `--temperature, -t` | 0.8 | Sampling temperature |
| `--api-key, -k` | $OPENAI_API_KEY | API key |
| `--base-url` | OpenAI | API base URL (for local models) |
| `--cross-model` | — | Second model for cross-verification (verify only) |
| `--demo` | off | Use canned responses |
| `--json` | off | Machine-readable output |

## How it works

1. Send the same prompt N times with temperature > 0
2. Extract sentence-level claims from the primary response
3. Check how many of the N responses contain each claim's key content words
4. Score: `support_count / N` → confidence percentage
5. Tag each claim: 🟢 high (≥80%), 🟡 medium (≥50%), 🔴 low (<50%)

## Use as a library

```python
from confab import score_claims, call_llm, Claim

# Score claims from pre-existing responses
claims = score_claims(responses[0], responses)
for claim in claims:
    print(f"{claim.level}: {claim.text} ({claim.confidence:.0%})")
```

## Project structure

```
confab.py           → CLI entrypoint
confab/
  __init__.py       → version + public API
  __main__.py       → python -m confab
  engine.py         → claim extraction, scoring, annotation
  llm.py            → async LLM client + demo fixtures
  db.py             → SQLite history
  proxy.py          → OpenAI-compatible HTTP proxy
  cli.py            → argparse + command dispatch
```

## Works with any OpenAI-compatible API

```bash
# Ollama
confab --base-url http://localhost:11434/v1 -m llama3 check "..."

# Together AI
confab --base-url https://api.together.xyz/v1 -k $TOGETHER_KEY check "..."

# Any OpenAI-compatible endpoint
confab --base-url http://your-server/v1 check "..."
```

## Environment

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Default API key |
| `CONFAB_DB` | Custom path for history database (default: `~/.confab/history.db`) |

## License

MIT
