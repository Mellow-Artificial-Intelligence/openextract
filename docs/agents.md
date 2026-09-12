---
layout: page
title: For agents
description: Contract for coding agents generating or editing openextract integrations.
---

# For agents

This page is the integration contract for coding agents, IDE tools, and generated examples. Humans should start at the [Guide](guide.md). Machine-readable index: [llms.txt](llms.txt).

## Public surface

Import only names in `openextract.__all__`. Modules whose names start with `_` (`openextract._extract`, `_batch`, `_session`, `_cli`, …) are private and may change without deprecation.

Swarm surface (provisional): `extract_swarm*`, `SwarmMember`, `SwarmResult`, `SwarmReduce`, `reduce_outputs`, `resolve_swarm_members`.

Agent surface (provisional): `define_agent`, `define_remote_agent`, `DefinedAgent`, `RemoteAgent`, `flatten_agent`, `resolve_output_schema`, `load_agent`, `load_agents`, `load_agent_directory`, `RemoteAgentError`, and the `openextract.auth` helpers.

Stable enough to generate against: `extract`, `extract_async`, `extract_with_usage`, `extract_with_usage_async`, `Usage`, and the exception types. Provisional (still public, may evolve before 1.0): sessions, batch helpers, `ExtractionInput` / `ExtractionResult`, `ExtractionStyle`, CLI flags.

Canonical signatures: [API reference](api-reference.md). CI fails if those headings drift from the installed callables.

## Minimal call

```python
from pydantic import BaseModel
from openextract import extract


class Info(BaseModel):
    summary: str


result = extract(schema=Info, model="openai:gpt-5", input_file="doc.pdf")
```

Always define a real `pydantic.BaseModel` subclass. Do not ask the library for free-form JSON.

## Which API to generate

| Situation | Use |
| --- | --- |
| One input, sync code | `extract` |
| One input, async code | `extract_async` |
| Need token counts | `extract_with_usage` / `_async` |
| Same schema/model many times | `Extractor` / `AsyncExtractor` |
| Many inputs, want a list | `extract_many` (sync, **not** from a running loop) or `extract_many_async` |
| Many inputs, stream as done | `iter_extract_many_async` — yields `(input_index, result)` in **completion order** |
| Per-item usage / timing | `extract_many_with_results*` + `total_usage` |
| Per-field source spans (ExtractBench grounding) | `cite=True` on `extract_many_with_results*` or sessions (`Extractor` / `AsyncExtractor`). Read `ExtractionResult.citations`; session `extract()` still returns the schema instance. Swarm: `extract_swarm_with_results*` → `SwarmResult.citations` (per-agent cites stay on each agent). |
| Several agents on **one** input | `extract_swarm` / `extract_swarm_async` |
| A reusable specialist (model + style + instructions + schema) | `define_agent`, then `extract(agent, input_file)` or `agents=` |
| An extraction service over HTTP | `define_remote_agent` + `openextract.auth` |
| Agents shipped in a repository | `load_agent` / `load_agents` (directory, file, `module:attribute`) |
| Per-agent usage / failures from a swarm | `extract_swarm_with_results*` → `SwarmResult` |
| Shell / CI | `openextract` CLI; parse **stdout** only |
| A swarm or agent from the shell | `--swarm` / `--models` / `--agent` / `--agents` / `--reduce` |

## Input rules (common bugs)

- `bytes` and file-like objects **require** `media_type`. Missing it is `TypeError`, not an extraction error.
- Paths and URLs may omit `media_type`; it is guessed or taken from `Content-Type`.
- Mix paths, bytes, and URLs in one batch via `ExtractionInput` with per-item `media_type`.
- Default size cap is 50 MiB (`InputTooLargeError`). Do not read the file yourself “to be safe” unless the caller needs a smaller cap — pass `max_input_bytes`.
- The library does **not** load `.env`. Set credentials in the process environment, or load dotenv in the app/CLI layer.
- URL fetches block private/loopback hosts unless `OPENEXTRACT_ALLOW_PRIVATE_URLS` is set.

## Styles

Default `style='direct'`. `search` and `code` are **text-only** (`text/*`, JSON, XML, YAML). Do not emit `style='search'` for PDFs, images, audio, or video. Those styles need `pydantic-ai-harness` / `pydantic-ai-harness[codemode]` and raise `ProviderNotInstalledError` if the extra is missing. Do not combine `search`/`code` with an injected `agent=`.

## Errors to catch

Catch `ExtractionError` as the fallback. Prefer the specific subclass:

- `UrlFetchError` — fetch/SSRF
- `InputTooLargeError` — cap
- `SchemaValidationError` — output mismatch (tighten `instructions` / schema; this is not retried)
- `ModelError` — inspect `.retryable` before retrying yourself; the library already retries when `max_retries > 0`
- `ProviderNotInstalledError` — print the exception; it includes the install command
- `RemoteAgentError` — a remote agent endpoint failed; inspect `.status_code` and `.retryable`

Do not catch `Exception` and retry. Do not retry `SchemaValidationError` unless the user asked for it.

Invalid `max_retries`, `retry_backoff`, `retry_max_backoff`, or `max_concurrency` raise **`ValueError` before any model call**.

## Do not

- Import or patch `openextract._*` in application code.
- Call `extract_many` / `extract_many_with_results` from a running asyncio loop.
- Assume `extract_many` yields completion order — it returns **input order**. Completion order is only `iter_extract_many_async`.
- Treat CLI stderr as the payload. Exit `7` still has the batch JSON on stdout.
- Hard-code Chat Completions for OpenAI. `openai:` uses the Responses API; opt into `openai-chat:` only when required.
- Claim provider media support that the [capability matrix](providers.md) marks expected/unknown.
- Add provider SDKs as direct dependencies of the caller if `openextract[extra]` already pulls them.

## CLI if you shell out

```bash
openextract INPUT --schema package.mod:Class --model openai:gpt-5
```

| Exit | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Usage / bad `--schema` / invalid options |
| `2`–`6` | `UrlFetchError` … `ProviderNotInstalledError` |
| `7` | Partial batch failure (`--continue-on-error`) |

`--schema` is `module:ClassName`. For stdin, pass `-` and `--media-type`. Full contract: [CLI](cli.md).

## Tests without live providers

Prefer `pydantic_ai.models.test.TestModel` (or inject an `Agent`) so examples and unit tests do not need API keys:

```python
from pydantic_ai.models.test import TestModel
from openextract import extract

model = TestModel(custom_output_args={"summary": "ok"})
extract(schema=Info, model=model, input_file=b"hello", media_type="text/plain")
```

Repo examples that must run without credentials follow this pattern (`examples/advanced/reusable_sessions.py`, `examples/batch/stream_batch_extract.py`).

## Install extras (prefix → extra)

`openai`, `openai-chat`, `openai-responses`, `cerebras`, `ollama` → `openextract[openai]`. Also: `anthropic`, `google-gla` / `google-vertex` → `google`, `bedrock`, `cohere`, `groq`, `huggingface`, `mistral`, `openrouter`, `xai`. Unknown prefixes suggest `openextract[all]`.
