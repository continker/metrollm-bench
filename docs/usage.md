# Pointing the runner at an LLM endpoint

`harness/runner.py` speaks the OpenAI chat-completions interface, so any
OpenAI-compatible endpoint works. You select the provider with three flags:

```
--llm-url    base URL (default http://localhost:8080/v1)
--llm-key    API key / bearer token (default "local")
--llm-model  model or deployment name (default "qwen3.5")
```

The runner auto-detects a few providers from the URL and adjusts the request
body accordingly (token-limit field, reasoning-effort, thinking controls), so in
most cases setting the three flags above is all you need.

Tier-2 judging is separate: the scorer calls Claude Haiku and reads
`ANTHROPIC_API_KEY` from the environment or a `.env` file (see `.env.example`).

## Local llama-server (default)

Serve a GGUF with [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
`llama-server` on port 8080, then run the benchmark against it:

```bash
# terminal 1: tool server
uv run python -m harness.mock_server --system marta --port 8100

# terminal 2: the benchmark, against the local model
uv run python -m harness.runner \
  --cases cases/marta_cases.json --system marta \
  --llm-url http://localhost:8080/v1 --llm-key local --llm-model qwen3.5 \
  --output results/marta_local.json
```

For a local llama-server the runner sends `max_completion_tokens` and, when
`--no-thinking` is set, disables the model's thinking mode via
`chat_template_kwargs`.

## OpenAI API

```bash
uv run python -m harness.runner \
  --cases cases/marta_cases.json --system marta \
  --llm-url https://api.openai.com/v1 --llm-key "$OPENAI_API_KEY" \
  --llm-model gpt-4o --output results/marta_openai.json
```

GPT-5-family models are sent `reasoning_effort` automatically.

## Azure OpenAI

Point `--llm-url` at your deployment endpoint; the runner detects `azure.com` in
the URL and switches to the Azure request shape. GPT-5 deployments are sent
`reasoning_effort`, and note that Azure GPT-5 deployments require
`--temperature 1.0` (temperature 0 is rejected).

```bash
uv run python -m harness.runner \
  --cases cases/marta_cases.json --system marta \
  --llm-url "https://<resource>.cognitiveservices.azure.com/openai/deployments/<deployment>?api-version=<ver>" \
  --llm-key "$AZURE_OPENAI_KEY" --llm-model <deployment> \
  --temperature 1.0 --output results/marta_azure.json
```

## Mistral API

```bash
uv run python -m harness.runner \
  --cases cases/marta_cases.json --system marta \
  --llm-url https://api.mistral.ai/v1 --llm-key "$MISTRAL_KEY" \
  --llm-model mistral-small-latest --output results/marta_mistral.json
```

## Useful flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--system` | `marta` | Which transit system the cases belong to |
| `--mock-url` | `http://localhost:8100` | Tool server URL |
| `--parallel` | `2` | Concurrent cases (raise for fast hosted APIs, mind rate limits) |
| `--temperature` | `0.0` | Sampling temperature (0 for reproducibility; 1.0 for Azure GPT-5) |
| `--thinking` / `--no-thinking` | thinking on | Toggle the model's thinking mode |
| `--max-tool-rounds` | `20` | Tool-call rounds before a case is cut off |
| `--max-tokens` | `4096` | Max tokens per response |
| `--limit` / `--case-ids` | — | Run a subset of cases (for quick checks) |
| `--extra-body-json` | — | JSON string shallow-merged into each request body (for provider-specific fields) |

## Scoring the run

```bash
uv run python -m harness.scorer \
  --system marta --results results/marta_local.json \
  --output results/marta_local_scored.json
```

The scorer runs the 14 deterministic Tier-1 components locally and the 8 Tier-2
components through the Haiku judge (cached per `(case, rubric)` under
`results/`). Set `ANTHROPIC_API_KEY` before scoring.
