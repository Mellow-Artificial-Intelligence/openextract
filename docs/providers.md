---
layout: page
title: Provider capability matrix
---

# Provider capability matrix

Status values:

| Status | Meaning |
| ------ | ------- |
| **verified** | Exercised by repo examples or maintainer live smoke runs |
| **expected** | Supported through `pydantic-ai` / the provider SDK, not yet smoke-tested here |
| **unknown** | Possible via upstream, but openextract has not validated it |

Media capabilities inherit from the model and provider SDK. Prefer a
vision/audio-capable model identifier when extracting non-text media.

## Matrix

| Provider | Extra | Credentials | Example model | Text | Image | PDF | Audio | Video | Usage | Notes |
| -------- | ----- | ----------- | ------------- | ---- | ----- | --- | ----- | ----- | ----- | ----- |
| OpenAI | `openai` | `OPENAI_API_KEY` | `openai:gpt-5` | verified | verified | expected | expected | unknown | verified | Used by `examples/basic/local_file.py`, batch, retries |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `anthropic:claude-sonnet-4` | verified | verified | expected | unknown | unknown | expected | Used by bytes/async/receipt examples |
| Google (GLA) | `google` | `GEMINI_API_KEY` | `google-gla:gemini-2.5-pro` | expected | expected | expected | expected | unknown | expected | Prefix `google-gla` |
| Google (Vertex) | `google` | GCP ADC / Vertex config | `google-vertex:...` | expected | expected | expected | expected | unknown | expected | Prefix `google-vertex` |
| AWS Bedrock | `bedrock` | AWS creds / `AWS_BEARER_TOKEN` | `bedrock:anthropic.claude-sonnet-4-20250514-v1:0` | expected | expected | expected | unknown | unknown | expected | Region and model access are account-specific |
| xAI | `xai` | `XAI_API_KEY` | `xai:grok-4.3` | verified | verified | expected | verified | unknown | verified | Used by URL, usage, audio examples |
| Cohere | `cohere` | `CO_API_KEY` | `cohere:command-r-plus` | expected | unknown | unknown | unknown | unknown | expected | Multimodal support depends on model |
| Groq | `groq` | `GROQ_API_KEY` | `groq:llama-3.3-70b-versatile` | expected | unknown | unknown | unknown | unknown | expected | Model-dependent media support |
| Hugging Face | `huggingface` | `HF_TOKEN` | `huggingface:meta-llama/Llama-3.3-70B-Instruct` | expected | unknown | unknown | unknown | unknown | expected | Endpoint/model capabilities vary |
| Mistral | `mistral` | `MISTRAL_API_KEY` | `mistral:mistral-large-latest` | expected | expected | expected | unknown | unknown | expected | |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `openrouter:~typesafe/jev-latest` | expected | expected | expected | unknown | unknown | expected | OpenAI-compatible; Jev cookbook: `examples/advanced/openrouter_jev.py` |
| Cerebras | `openai` | `CEREBRAS_API_KEY` | `cerebras:llama3.1-70b` | expected | unknown | unknown | unknown | unknown | expected | OpenAI-compatible path |
| Ollama | `openai` | optional `OLLAMA_API_KEY` | `ollama:llama3` | expected | unknown | unknown | unknown | unknown | expected | Local server; uses `NativeOutput` path |
| Outlines | install `pydantic-ai-slim[outlines-*]` | backend-specific | `outlines:transformers/...` | expected | unknown | unknown | unknown | unknown | unknown | Local constrained decoding; not an openextract extra |

Install examples:

```bash
pip install 'openextract[openai]'
pip install 'openextract[anthropic]'
pip install 'openextract[all]'
```

Missing extras raise `ProviderNotInstalledError` with a provider-specific install
hint when the model prefix is known (`_PROVIDER_EXTRAS` in
`src/openextract/_agent.py`).

## Known gaps / follow-ups

- PDF/audio/video cells marked **expected** or **unknown** need live smoke
  coverage before they can be promoted to **verified**.
- Provider quirks (rate limits, max upload size, vision availability) are owned
  by upstream SDKs and can change without an openextract release.
- See [Live provider smoke tests](live-smoke.md) for the opt-in verification harness.

## Related

- [Troubleshooting](troubleshooting.md)
- [Examples README](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/examples/README.md)
