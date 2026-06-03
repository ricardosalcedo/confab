# Confab

Inline hallucination confidence for LLMs via self-consistency.

Send a prompt to an LLM N times, compare claims across responses, and annotate with confidence scores. Claims that appear consistently get high confidence; claims that vary get low confidence.

## Install

```bash
# Just need httpx for API calls
pip install httpx

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

# Cross-model verification (ask two models)
confab verify "Python was released in 1991" --cross-model gpt-4o

# From stdin
echo "Earth is flat" | confab verify -
```

### History

```bash
# Review past checks
confab history

# Show last 5
confab history --limit 5

# Clear history
confab history --clear
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

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model, -m` | gpt-4o-mini | Model to use |
| `-n` | 5 | Number of consistency samples |
| `--temperature, -t` | 0.8 | Sampling temperature |
| `--api-key, -k` | $OPENAI_API_KEY | API key |
| `--base-url` | OpenAI | API base URL (for local models) |
| `--cross-model` | - | Second model for cross-verification |
| `--demo` | off | Use canned responses |
| `--json` | off | Machine-readable output |

## How it works

1. Send the same prompt N times with temperature > 0
2. Extract sentence-level claims from the primary response
3. Check how many of the N responses support each claim
4. Score: `support_count / N` → confidence percentage
5. Tag each claim: 🟢 high (≥80%), 🟡 medium (≥50%), 🔴 low (<50%)

## Works with any OpenAI-compatible API

```bash
# Ollama
confab --base-url http://localhost:11434/v1 -m llama3 check "..."

# Together AI
confab --base-url https://api.together.xyz/v1 -k $TOGETHER_KEY check "..."

# Any OpenAI-compatible endpoint
confab --base-url http://your-server/v1 check "..."
```

## License

MIT
