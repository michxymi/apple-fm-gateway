# apple-fm-gateway (V1)

FastAPI gateway that exposes Apple's on-device Foundation Model (`apple_fm_sdk`) through:
- OpenAI-compatible endpoints (`/v1/models`, `/v1/chat/completions`)
- Anthropic-compatible endpoints for Claude Code (`/v1/messages`, `/v1/messages/count_tokens`)

## Requirements

- See [Foundation Models SDK for Python](https://github.com/apple/python-apple-fm-sdk/blob/main/README.md#requirements)
- Python `>=3.13`
- [`uv`](https://docs.astral.sh/uv/) installed

## Installation

```bash
git submodule update --force --init --recursive
uv sync --dev
```

## Run the Server

```bash
uv run fastapi dev
```

Base URL:

```text
http://127.0.0.1:8000
```

## OpenAI Compatibility

### Supported endpoints
- `GET /v1/models`
- `POST /v1/chat/completions`

### OpenAI example

```python
from openai import OpenAI

client = OpenAI(
    api_key="unused-local-key",
    base_url="http://127.0.0.1:8000/v1",
)

resp = client.chat.completions.create(
    model="apple.fm.system",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Explain dependency injection briefly."},
    ],
)

print(resp.choices[0].message.content)
```

## Claude Code Compatibility (Anthropic Gateway)

### Supported endpoints
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

### Claude Code setup

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=unused-local-token
export ANTHROPIC_MODEL=sonnet
```

Notes:
- Auth is not enforced in v1, but Claude Code expects an auth token env var.
- Model aliases `sonnet`, `haiku`, `opus`, and `claude-*` are mapped to backend model `apple.fm.system`.

### Anthropic non-stream example

```bash
curl -s http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "sonnet",
    "max_tokens": 256,
    "messages": [
      {
        "role": "user",
        "content": [{"type": "text", "text": "Say hello in one sentence."}]
      }
    ]
  }' | jq
```

### Anthropic streaming example

```bash
curl -N http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "haiku",
    "stream": true,
    "max_tokens": 256,
    "messages": [
      {
        "role": "user",
        "content": [{"type": "text", "text": "Write a short haiku about debugging."}]
      }
    ]
  }'
```

### Count tokens example

```bash
curl -s http://127.0.0.1:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "sonnet",
    "system": "You are concise.",
    "messages": [
      {
        "role": "user",
        "content": [{"type": "text", "text": "Count these tokens."}]
      }
    ]
  }' | jq
```

## Compatibility Matrix (V1)

| Area | Status | Notes |
|---|---|---|
| `GET /v1/models` | Supported | Returns `apple.fm.system` |
| `POST /v1/chat/completions` | Supported | OpenAI-compatible |
| OpenAI stream (`stream=true`) | Supported | SSE chunks + `[DONE]`; optional final usage chunk via `stream_options.include_usage=true` |
| OpenAI `response_format.type=json_schema` | Supported | Non-stream only |
| `POST /v1/messages` | Supported | Anthropic-compatible message API |
| Anthropic stream (`stream=true`) | Supported | `message_start`, `content_block_*`, `message_delta`, `message_stop` |
| `POST /v1/messages/count_tokens` | Supported | Deterministic token estimate |
| Anthropic model aliases | Supported | `sonnet`, `haiku`, `opus`, `claude-*` -> `apple.fm.system` |
| Anthropic non-text content blocks | Not supported | Returns 400 in v1 |
| Tool/function calling | Not supported | Accepted/ignored for OpenAI; unsupported for Anthropic blocks |

## Limitations / Caveats (V1)
- Foundation Models SDK for Python doesn't have a PyPI package as of yet, so we are using submodules.
- Apple Foundation Models have a small context window (4096 tokens).
- Command line tools get rate limited in MacOS, [source](https://developer.apple.com/forums/thread/787737)
- Anthropic content blocks are text-only (`type="text"`).
- `/v1/messages/count_tokens` uses deterministic estimation (not backend-native token accounting).
- No auth enforcement in v1 (localhost/trusted setup).
- All Anthropic model aliases route to the same backend model: `apple.fm.system`.
- OpenAI usage fields use deterministic estimation (not backend-native token accounting).

## Error Shapes

- OpenAI endpoints return:

```json
{"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
```

- Anthropic endpoints return:

```json
{"type": "error", "error": {"type": "...", "message": "..."}}
```

## Tests

```bash
uv run pytest -q
```

Coverage includes:
- OpenAI endpoints + regression checks
- Anthropic non-stream/stream endpoints
- Count tokens behavior
- Model alias mapping
- Unsupported block validation
- Backend unavailable error mapping
