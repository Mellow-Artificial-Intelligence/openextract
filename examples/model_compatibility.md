# Model and HTTP adapter compatibility

OpenExtract uses HTTPX for provider requests and Pydantic for schemas and
validation. Install plain `openextract`; provider SDKs and extras are not needed.

## Configured models

Use `openextract.Model` for settings, API selection, endpoints, authentication,
timeouts, and retry budgets. Configured Pydantic AI model objects are no longer
accepted. Earlier SDK version-pair findings concerned the removed Pydantic AI
integration and do not constrain this HTTPX implementation.

The [configured-model example](configured_models.md) demonstrates Anthropic
`max_tokens` in both sync and async calls. Offline tests inspect the actual HTTP
request and validate a canned response using the original Pydantic schema.
They verify configuration propagation, not live access or extraction accuracy.

```bash
uv sync --dev --frozen
uv run --no-sync pytest tests/test_configured_models.py
```

## Provider settings and limits

| API | Settings placement | Structured output |
| --- | --- | --- |
| OpenAI Responses | Top-level request parameters | Strict JSON Schema |
| Chat Completions | Top-level request parameters | Forced tool by default; optional JSON Schema |
| Anthropic Messages | Top-level request parameters, including `max_tokens` | Forced extraction tool |
| Gemini | `generationConfig` | JSON Schema |
| Local Ollama | Top-level parameters; sampling inside `options` | JSON Schema |
| Cohere V2 Chat | Top-level request parameters | JSON Schema |
| Bedrock Converse | Top-level parameters; inference options inside `inferenceConfig` | Forced extraction tool |
| Decisions | Settings rejected | `bool`, `Literal`, and `Enum` fields only |

Settings use the provider's native spelling and are not checked against an SDK
settings type. Request structure and output-schema fields cannot be overridden.
Provider/model limits still apply; unsupported settings may produce `ModelError`.
Too small a token budget can truncate output and raise `SchemaValidationError`.

Strict JSON Schema output requests every declared field, including fields with
defaults, and rejects open-ended dictionaries. The original Pydantic model still
performs final validation. Tagged unions are converted to the provider's strict
schema format. All general adapters use JSON validation, including decoded
Anthropic and Bedrock tool inputs, so strict `date` and `Decimal` fields retain
the same behavior across providers. Schema validation cannot establish factual accuracy.

See the [provider matrix](../README.md#providers) for media and authentication
limits, including Bedrock bearer-token authentication and caller-supplied HTTPX
clients. AWS SDK credential discovery, Vertex AI credentials, and SDK-specific
configuration are not bundled.

## Optional live checks

The saved synthetic fixtures and opt-in matrix cover OpenAI, native Gemini,
native Anthropic, and Gemini through OpenRouter, in sync and async modes.
Gemini through OpenRouter uses JSON Schema output. Follow the
[live provider instructions](../CONTRIBUTING.md#live-provider-checks) to select
cases and supply credentials; ordinary CI requires no API keys.
