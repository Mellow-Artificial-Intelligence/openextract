<div align="center">

# openextract

**Extract structured data from documents, images, audio, and video using LLMs.**

[![PyPI version](https://img.shields.io/pypi/v/openextract.svg?logo=pypi&logoColor=white&color=4B8BBE)](https://pypi.org/project/openextract/)
[![Python versions](https://img.shields.io/pypi/pyversions/openextract.svg?logo=python&logoColor=white)](https://pypi.org/project/openextract/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://img.shields.io/pypi/dm/openextract.svg?color=blue)](https://pypi.org/project/openextract/)

[Documentation](https://mellow-artificial-intelligence.github.io/openextract/guide.html) &middot; [For agents](https://mellow-artificial-intelligence.github.io/openextract/agents.html) &middot; [PyPI](https://pypi.org/project/openextract/) &middot; [Changelog](CHANGELOG.md) &middot; [Issues](https://github.com/Mellow-Artificial-Intelligence/openextract/issues)

</div>

---

`openextract` turns any document, image, audio, or video file into a typed Pydantic model in a single function call. Point it at a local path or a URL, pass a schema, and get back a validated object you can use directly in your code.

The [guide](https://mellow-artificial-intelligence.github.io/openextract/guide.html) is the how-to. Coding agents should start at [For agents](https://mellow-artificial-intelligence.github.io/openextract/agents.html) or [llms.txt](https://mellow-artificial-intelligence.github.io/openextract/llms.txt).

## Features

- **Type-safe output.** Define your shape with Pydantic; get back a validated instance.
- **One function, many modalities.** Documents (PDF, DOCX), images, audio, and video.
- **Extraction styles.** Pass the document directly, search it with file tools, or write Python against the text via [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/).
- **Local files or URLs.** Pass a path or an `https://` URL &mdash; `openextract` handles fetching.
- **Bring your own model.** OpenAI, Anthropic, Google, AWS Bedrock, xAI, Cohere, Hugging Face, Groq, Cerebras, Mistral, and Ollama supported out of the box via [`pydantic-ai`](https://github.com/pydantic/pydantic-ai).
- **Explicit error handling.** Distinct exceptions for URL fetch, schema validation, and model errors.
- **100% test coverage**, enforced in CI.

## Installation

```bash
uv add openextract
```

Or with pip:

```bash
pip install openextract
```

Model calls require a provider SDK. Install the extra for the provider you use, for example `openextract[openai]`, `openextract[anthropic]`, or `openextract[all]` for every supported provider. Local PDF parse-then-extract (page text and word boxes for citations) needs `openextract[pdf]` or `openextract[all]`. Agentic `search` and `code` styles need [`pydantic-ai-harness`](https://pydantic.dev/docs/ai/harness/) (`pip install pydantic-ai-harness` / `pip install 'pydantic-ai-harness[codemode]'`). The base package ships `pydantic-ai-slim` without provider SDKs pre-installed. If the requested provider SDK is missing, `openextract` raises `ProviderNotInstalledError` with a provider-specific `pip install 'openextract[...]'` command when the model prefix is known.

Requires Python 3.12+.

## Quick start

```python
from pydantic import BaseModel
from openextract import extract


class PdfInfo(BaseModel):
    summary: str
    language: str


result = extract(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="https://example.com/document.pdf",
    instructions="Return a two-sentence summary and the document's primary language.",
)

print(result.summary)
print(result.language)
```

`result` is a fully-validated `PdfInfo` instance &mdash; not a dict, not a string.

## Extraction styles

`style` selects how the model inspects the input. The default, `direct`, is the
current behavior: the resolved media is passed to the LLM in one shot. For
**text** documents you can instead use agentic search or code execution, both
powered by [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/).

```python
from openextract import extract, ExtractionStyle

# Default: send the document bytes to the model.
extract(schema=PdfInfo, model="openai:gpt-5", input_file="notes.txt")

# Grep/read the text with sandboxed file tools (needs pydantic-ai-harness).
extract(
    schema=PdfInfo,
    model="openai:gpt-5",
    input_file="notes.txt",
    style="search",  # or ExtractionStyle.SEARCH
)

# Write Python against a workspace copy of the document (needs pydantic-ai-harness[codemode]).
extract(
    schema=PdfInfo,
    model="openai:gpt-5",
    input_file="notes.txt",
    style="code",
)
```

`search` and `code` require UTF-8 text (`text/*`, JSON, XML, YAML, and similar).
PDFs, Office documents, images, audio, and video stay on `direct`. Missing
packages raise `ProviderNotInstalledError` with a `pip install pydantic-ai-harness`
or `pip install 'pydantic-ai-harness[codemode]'` hint. The integration was
written against `pydantic-ai-harness` 0.18.x. The CLI flag is `--style`.

## Reusable sessions

For repeated extractions with the same schema and model, use `Extractor` or
`AsyncExtractor`. A session constructs one Pydantic AI agent, reuses its
provider and input-fetch HTTP clients, and closes them deterministically.

```python
from openextract import Extractor, RetryPolicy

with Extractor(
    schema=PdfInfo,
    model="openai:gpt-5",
    instructions="Extract the summary and primary language.",
    model_settings={"temperature": 0},
    timeout=30,
    retry_policy=RetryPolicy(max_retries=3),
) as extractor:
    first = extractor.extract("./reports/q3.pdf")
    second, usage = extractor.extract_with_usage("./reports/q4.pdf")
```

The async session is bound to the event loop that enters it and supports
concurrent calls on that loop:

```python
import asyncio
from openextract import AsyncExtractor

async def main() -> None:
    async with AsyncExtractor(PdfInfo, "openai:gpt-5") as extractor:
        q3, q4 = await asyncio.gather(
            extractor.extract("./reports/q3.pdf"),
            extractor.extract("./reports/q4.pdf"),
        )

asyncio.run(main())
```

`model` may also be a configured `pydantic_ai.models.Model`, preserving custom
providers, endpoints, credentials, and model defaults without string-prefix
routing. For advanced dependency injection, pass a fully configured
`pydantic_ai.Agent` as `agent=` instead of `model=`. Openextract revalidates the
agent output against `schema`; agent instructions, model settings, timeout, and
instrumentation must be configured on the injected agent itself.

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from openextract import Extractor

test_agent = Agent(
    TestModel(custom_output_args={"summary": "Test", "language": "en"}),
    output_type=PdfInfo,
)
with Extractor(PdfInfo, agent=test_agent) as extractor:
    assert extractor.extract(b"fixture", media_type="text/plain").language == "en"
```

`Extractor` is thread-bound and not thread-safe; use one per thread.
`AsyncExtractor` must be entered and used on one event loop, though calls may
overlap within that loop. Both classes must be used as context managers (or
closed explicitly with `close()` / `aclose()`). Set `instrument=True` to enable
Pydantic AI instrumentation, or pass an `InstrumentationSettings` instance.

## Usage

### Local files

```python
result = extract(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="./reports/q4.pdf",
)
```

### Bytes or file-like objects

```python
result = extract(schema=PdfInfo, model="xai:grok-4.3", input_file=pdf_bytes, media_type="application/pdf")
# A file-like object with .read() works too; pass media_type explicitly:
result = extract(schema=PdfInfo, model="xai:grok-4.3", input_file=open("q4.pdf", "rb"), media_type="application/pdf")
```

### Input size limits

Every input is capped at 50 MiB (`52_428_800` bytes) before a model call.
Local paths are checked before and during reading, URL bodies are streamed
through the cap even when `Content-Length` is missing or incorrect, and binary
streams are read in bounded chunks. Override the limit for one call with
`max_input_bytes`, or set `OPENEXTRACT_MAX_INPUT_BYTES` for the process:

```python
result = extract(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="./reports/large.pdf",
    max_input_bytes=100 * 1024 * 1024,
)
```

Oversized inputs raise `InputTooLargeError` before a model request. The CLI
exposes the same control as `--max-input-bytes` and reports the error with exit
code `5`.

### Retry on transient model errors

```python
result = extract(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="./reports/q4.pdf",
    max_retries=3,
)
```

`max_retries` defaults to `0` (single attempt) and must be a non-negative integer.
Only transient `ModelError` failures—timeouts, rate limits, and supported 5xx
responses—are retried; authentication, permission, and invalid-request failures
fail immediately. Delays use exponential backoff with up to 25% additive jitter,
bounded by `retry_max_backoff` (60 seconds by default). A valid provider
`Retry-After` value takes precedence but is still bounded. Both backoff values
must be finite and non-negative.

The input is resolved once before the first model attempt. Retries reuse the same
media bytes, prompt, and agent, so URLs and non-seekable streams are not fetched
or read again.

### Inspecting token usage

Use `extract_with_usage` when you want token counts alongside the extracted output (for cost tracking, logging, etc.).

```python
from openextract import extract_with_usage

result, usage = extract_with_usage(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="./reports/q4.pdf",
)

print(result.summary)
print(f"tokens: {usage.input_tokens} in / {usage.output_tokens} out / {usage.total_tokens} total")
```

`usage` is a frozen `Usage` dataclass with `input_tokens`, `output_tokens`, and `total_tokens` fields.

### Batch extraction

Every public API accepts a `pathlib.Path` (or any `os.PathLike`) directly, in
addition to `str` paths/URLs, `bytes`, and binary file-like objects. Batch
calls accept `ExtractionInput` items so heterogeneous inputs can carry their
own media type in one batch:

```python
from pathlib import Path
from openextract import ExtractionInput, extract_many

results = extract_many(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_files=[
        Path("./reports/q3.pdf"),
        ExtractionInput(source=b"...pdf bytes...", media_type="application/pdf"),
        ExtractionInput(source=b"...png bytes...", media_type="image/png"),
    ],
)
```

A batch-wide `media_type` still applies to any item that does not specify one.
`return_exceptions` is typed, so checkers infer `list[PdfInfo]` by default and
`list[PdfInfo | Exception]` when it is `True`.

For per-item token usage, attempt counts, timing, and sanitized source labels,
use `extract_many_with_results` and aggregate with `total_usage`:

```python
from openextract import extract_many_with_results, total_usage

results = extract_many_with_results(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_files=[Path("./reports/q3.pdf"), Path("./reports/q4.pdf")],
)

total = total_usage(results)
print(total.input_tokens, total.output_tokens, total.total_tokens)
```

Each `ExtractionResult` carries `output`, `usage`, `attempts`, `duration`,
`model`, `media_type`, and a sanitized `source` (never raw media, credentials,
or query strings). The async sibling is `extract_many_with_results_async`.

### Swarm extraction

`extract_many` scales across inputs. A swarm scales across *agents* on one
input: the file is loaded once, the agents run concurrently, and their outputs
are reduced into a single validated object.

```python
from openextract import SwarmMember, extract_swarm, extract_swarm_with_results

# Two models cross-checking each other, majority per field.
invoice = extract_swarm(
    schema=Invoice,
    agents=["openai:gpt-5.5", "anthropic:claude-opus-4-8"],
    input_file="invoice.pdf",
    reduce="vote",
)

# Per-agent instructions, plus per-agent usage and failures.
swarm = extract_swarm_with_results(
    schema=Invoice,
    agents=[
        SwarmMember("openai:gpt-5.5", instructions="Line items only."),
        SwarmMember("openai:gpt-5.5", instructions="Totals and dates only."),
    ],
    input_file="invoice.pdf",
)
print(swarm.output, swarm.usage, swarm.reduce)
```

`reduce` is `merge` (union lists, fill fields, the default), `vote` (majority
per field), or `first`. Agents that fail are reported in `swarm.agents`; only
an all-agent failure raises. `size=N` fans one agent out up to 16 ways.

### Agents

`define_agent` packages a model, style, instructions, and output schema behind
a description, and `subagents` compose several into one. An agent is accepted
anywhere a swarm takes `agents`:

```python
from openextract import define_agent, extract_swarm

line_items = define_agent("Line items", model="openai:gpt-5.5", instructions="Rows only.")
totals = define_agent("Totals", model="openai:gpt-5.5", instructions="Totals and dates.")
invoices = define_agent("Invoices", output_schema=Invoice, subagents=[line_items, totals])

invoice = extract_swarm(schema=Invoice, agents=invoices, input_file="invoice.pdf")
```

An agent works in `extract` too, and supplies the schema when it declares one:

```python
invoice = extract(invoices, "invoice.pdf")
```

A single-model agent runs as an ordinary one-shot call; an agent with subagents
or a remote endpoint fans out into a swarm and its outputs are merged.

Agents also load from disk or an import path, so a repository can ship them
next to the code — `load_agent("agents/invoices")` reads `agent.py`,
`subagents/`, and `instructions.md`; `load_agent("my_pkg.agents:invoices")`
imports one.

`define_remote_agent` points at an HTTP extraction service instead of a local
model, with per-request auth from `openextract.auth`:

```python
from openextract import define_remote_agent
from openextract.auth import bearer

remote = define_remote_agent(
    url="https://agents.example.com",
    description="Hosted invoice reader",
    auth=bearer(lambda: os.environ["AGENT_TOKEN"]),
)
```

### Streaming batch

`extract_many` waits for every item and returns a list in **input order**.
`iter_extract_many_async` yields `(input_index, result)` in **completion
order**, so you can start processing before the last item finishes. Inputs are
consumed lazily and at most `max_concurrency` items are in flight.

```python
import asyncio
from openextract import iter_extract_many_async

async def main() -> None:
    async for index, result in iter_extract_many_async(
        schema=PdfInfo,
        model="xai:grok-4.3",
        input_files=[Path("./reports/q3.pdf"), Path("./reports/q4.pdf")],
        return_exceptions=True,
    ):
        if isinstance(result, Exception):
            print(f"{index} failed: {result}")
        else:
            print(index, result.summary)

asyncio.run(main())
```

Runnable comparison of input order vs completion order:
[`examples/batch/stream_batch_extract.py`](examples/batch/stream_batch_extract.py).

### Choosing a model

`model` follows the `pydantic-ai` provider prefix convention:

| Provider     | Example identifier                                       | Install extra |
| ------------ | -------------------------------------------------------- | ------------- |
| OpenAI       | `openai:gpt-5`                                           | `openextract[openai]` |
| Anthropic    | `anthropic:claude-sonnet-4`                              | `openextract[anthropic]` |
| Google       | `google-gla:gemini-2.5-pro`                              | `openextract[google]` |
| AWS Bedrock  | `bedrock:anthropic.claude-sonnet-4-20250514-v1:0`        | `openextract[bedrock]` |
| xAI          | `xai:grok-4.3`                                           | `openextract[xai]` |
| Cohere       | `cohere:command-r-plus`                                  | `openextract[cohere]` |
| Hugging Face | `huggingface:meta-llama/Llama-3.3-70B-Instruct`          | `openextract[huggingface]` |
| Groq         | `groq:llama-3.3-70b-versatile`                           | `openextract[groq]` |
| Cerebras     | `cerebras:llama3.1-70b`                                  | `openextract[openai]` |
| Mistral      | `mistral:mistral-large-latest`                           | `openextract[mistral]` |
| OpenRouter   | `openrouter:anthropic/claude-sonnet-4`                   | `openextract[openrouter]` |
| Outlines     | `outlines:transformers/meta-llama/Llama-3.2-1B-Instruct` | Install the matching `pydantic-ai-slim[outlines-*]` backend |
| Ollama       | `ollama:llama3`                                          | `openextract[openai]` |

OpenAI identifiers using the concise `openai:` prefix are routed through the
Responses API by default. Use `openai-responses:` to select it explicitly or
`openai-chat:` to force the legacy Chat Completions API.

Ollama and Cerebras work via the `openai`-compatible code path &mdash; no dedicated extra is required for either.

Set the corresponding provider credentials in your environment (e.g.
`XAI_API_KEY` for xAI). The CLI and bundled examples load `.env` files; the
Python library leaves environment configuration to the host application.

OpenRouter and Cerebras are openai-compatible (they go through the `openai` client under the hood), so their errors are already classified via the existing openai path &mdash; no separate exception handling is needed.

Outlines runs models locally (via HuggingFace transformers, llama-cpp, MLX, vLLM, or SGLang) and enforces JSON-schema-conforming output at the token level. Install it separately alongside the backend you want, for example `pip install pydantic-ai-slim[outlines-transformers]`.

### Command line

`openextract` ships with a CLI for one-shot extractions from the shell.

```bash
openextract ./reports/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --instructions "Pull totals and line items." \
  --output json
```

Batch multiple files (JSON array output, `--max-concurrency` in-flight at once):

```bash
openextract ./invoices/a.pdf ./invoices/b.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --max-concurrency 8
```

Stream a large batch as JSONL — each record is written the moment its input
finishes, with progress on stderr, so downstream tooling can start consuming
immediately:

```bash
openextract ./invoices/*.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --output jsonl --continue-on-error --progress \
  | jq -c 'select(.result) | .result'
```

Mix media types in one run with a JSONL manifest (`source` required;
`media_type` and `name` optional per line):

```bash
cat > inputs.jsonl <<'EOF'
{"source": "./invoices/a.pdf", "media_type": "application/pdf", "name": "invoice-a"}
{"source": "https://example.com/report", "media_type": "text/html"}
EOF
openextract --manifest inputs.jsonl \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --output jsonl
```

Token usage (single file or batch; batches add per-item and aggregate usage):

```bash
openextract ./reports/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --usage
```

Swarm one input across several models, or extract with an agent:

```bash
openextract ./reports/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --models openai:gpt-5.5,anthropic:claude-opus-4-8 \
  --reduce vote

# --schema is optional when the agent declares an output_schema
openextract ./reports/q4.pdf --agent ./agents/invoices
```

Read from stdin:

```bash
cat ./reports/q4.pdf | openextract - \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --media-type application/pdf
```

- `input_file` accepts one or more paths/URLs, or `-` for stdin (`--media-type` required for stdin).
- `--schema` is a Python import path of the form `module:ClassName` resolving to a Pydantic model.
- `--model` is a `pydantic-ai` model identifier.
- `--instructions` is optional natural-language guidance.
- `--style` is `direct` (default), `search` (file tools on text), or `code`
  (write Python against text). `search` needs `pydantic-ai-harness`; `code`
  needs `pydantic-ai-harness[codemode]`.
- `--media-type` sets MIME type for stdin, overrides guessing for paths/URLs,
  and is the fallback for manifest entries without their own.
- `--manifest` reads inputs from a JSONL file with per-item media types and
  display names; mutually exclusive with positional inputs.
- `--usage` prints `result` and `usage` for a single input; batches report
  per-item usage plus an aggregate.
- `--cite` requests per-field citations (`{field, quote, page}`, optional
  `bbox`) in JSON/jsonl via the same `cite=True` library path.
- `--output` is `json` (default), `jsonl` (one record per completed input,
  written incrementally in completion order with an `index` field), or `repr`.
- `--max-concurrency` bounds in-flight extractions for batches (default 5).
- `--progress` reports per-item batch completion on stderr only.
- `--max-retries`, `--retry-backoff`, and `--retry-max-backoff` match the Python
  API retry behavior.
- `--max-input-bytes` overrides the 50 MiB per-input cap.
- `--continue-on-error` (batch only) keeps processing when an input fails; each
  failure is emitted inline as `{"input", "error", "error_type"}` and the command
  exits `7` if any input failed. Without it, a batch aborts on the first failure.
- `--swarm N`, `--models a,b`, `--agent SPEC`, `--agents SPEC,SPEC`, and
  `--reduce merge|vote|first` run several agents over a **single** input and
  fold their outputs; `--schema` is optional when an agent declares an
  `output_schema`.

Concurrency, retry, and size options are validated before any model call.

Exit codes: `0` success, `2` URL fetch error, `3` schema validation error, `4` model error,
`5` other extraction error, `6` missing provider extra, `7` partial batch failure
(`--continue-on-error`), `8` remote agent failure, `130` interrupted, `141`
broken pipe, `1` any other failure (including missing or bad
`--schema` / `--model` and invalid manifests).

Extraction errors and progress are written to stderr; successful JSON, JSONL
records, usage payloads, and `--continue-on-error` batch arrays are written to
stdout. Missing provider extras
exit `6` and include the same install hint as the Python API, for example
`pip install 'openextract[xai]'`. Partial batch failures with `--continue-on-error`
still print the full batch array to stdout, write a warning to stderr, and exit `7`.

Full stdout/stderr/exit-code contracts: [docs/cli.md](docs/cli.md).
Provider capability matrix: [docs/providers.md](docs/providers.md).
Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md).

## Examples

Runnable scripts live in [`examples/`](examples/), grouped by use case (local files, bytes, URLs, images, batch, streaming batch, async, retries, CLI, and more). See [examples/README.md](examples/README.md) for the full table.

```bash
# Run all fixture-based examples (uses OpenAI, Anthropic, and xAI — see examples/README.md)
uv run python -m examples.run_all

# Single example with the bundled sample image
uv run python -m examples.basic.local_file --fixture
```

[See the examples/ directory](examples/) for the full source.

### Error handling

```python
from openextract import (
    extract,
    InputTooLargeError,
    UrlFetchError,
    SchemaValidationError,
    ModelError,
    ProviderNotInstalledError,
    ExtractionError,
)

try:
    result = extract(schema=PdfInfo, model="xai:grok-4.3", input_file=url)
except UrlFetchError:
    ...  # The URL could not be fetched
except InputTooLargeError:
    ...  # The input exceeded the configured byte limit
except SchemaValidationError:
    ...  # The model's output did not match your schema
except ProviderNotInstalledError:
    ...  # The provider extra isn't installed (e.g. pip install openextract[xai])
except ModelError as exc:
    # Structured fields are populated when the provider exposes them.
    print(exc.provider, exc.status_code, exc.retryable, exc.retry_after)
except ExtractionError:
    ...  # Any other extraction failure (base class)
```

All `openextract` exceptions inherit from `ExtractionError`, so you can catch it as a single fallback if you prefer.

## API reference

The canonical public API reference lives in
[docs/api-reference.md](docs/api-reference.md) (also on the
[docs site](https://mellow-artificial-intelligence.github.io/openextract/api-reference.html)).
CI verifies every documented function signature against the installed package so
signature drift fails the build.

## Public API stability

`openextract.__all__` is the public Python API surface. Modules and helpers whose
names start with `_`, including `openextract._extract`, `openextract._batch`,
`openextract._session`, and `openextract._cli`, are internal implementation
details. The CLI command is also user-facing and follows the compatibility notes
below even though it is not exported from `__all__`.

| API | Status for 1.0 | Notes |
| --- | --- | --- |
| `ExtractionStyle` | Provisional | `direct`, `search`, or `code` extraction strategy. `search`/`code` are text-only and require `pydantic-ai-harness`. |
| `extract` | Stable | Primary synchronous API. Signature, return type, media input forms (`str`, `os.PathLike`, `bytes`, file-like, `ExtractionInput`), retry behavior, and public exception categories are intended to carry into 1.0 unchanged. `style` is additive. |
| `extract_async` | Stable | Async sibling of `extract`; same input contract and retry behavior, with `Agent.run` instead of `run_sync`. |
| `extract_with_usage` | Stable | Usage-returning sync API. The `(output, Usage)` tuple shape is stable; exact token values depend on provider reporting. |
| `extract_with_usage_async` | Stable | Async sibling of `extract_with_usage`; same tuple shape and retry behavior. |
| `extract_many` | Provisional | Batch return ordering, option validation, and `return_exceptions` semantics are intended to remain. Calling it from a running event loop raises `RuntimeError`; use `extract_many_async` in async code. Accepts `os.PathLike` and per-item `ExtractionInput` media types. |
| `extract_many_async` | Provisional | Async batch API with the same return shape, option constraints, and per-item retry behavior as `extract_many`. |
| `iter_extract_many_async` | Provisional | Bounded async iterator yielding `(input_index, result)` pairs in completion order. |
| `extract_many_with_results` | Provisional | Batch API returning per-item `ExtractionResult` objects (output, usage, attempts, duration, model/media metadata, sanitized source). |
| `extract_many_with_results_async` | Provisional | Async sibling of `extract_many_with_results`. |
| `total_usage` | Provisional | Sum `Usage` across batch `ExtractionResult` objects. |
| `ExtractionInput` | Provisional | Frozen input contract wrapping a media source with optional per-item `media_type` and safe `name`. |
| `ExtractionResult` | Provisional | Frozen generic result contract; never retains raw media, credentials, or provider internals. Additive `citations` when `cite=True`. |
| `Citation` | Provisional | Per-field source span (`field`, `quote`, `page`, optional normalized `bbox`). `as_dict()` is the JSON shape (`bbox` as four floats or `null`). `as_field_citation()` maps to ExtractBench `FieldCitation`. |
| `Usage` | Stable | Frozen dataclass with `input_tokens`, `output_tokens`, and `total_tokens`. New fields, if ever needed, should be additive. |
| `ExtractionError` | Stable | Base class for all public `openextract` exceptions. Catch this for a broad fallback. |
| `UrlFetchError` | Stable | Raised for URL fetch and URL safety failures. Message wording may improve, but the exception type is stable. |
| `InputTooLargeError` | Stable | Raised before a model call when resolved media exceeds the configured per-input byte limit. |
| `SchemaValidationError` | Stable | Raised when model output cannot be validated against the requested schema. |
| `ModelError` | Stable | Raised for provider/model API failures, with `provider`, `status_code`, `retryable`, and `retry_after` metadata where available. |
| `ProviderNotInstalledError` | Stable | Raised when the requested model provider extra is missing. Install hints may become more specific as providers are added. |
| `openextract` CLI | Provisional | The command, core flags, JSON output, stderr error reporting, provider-install exit code `6`, partial-batch exit code `7`, and remote-agent exit code `8` are intended to remain. |

No pre-1.0 signature changes are currently proposed for stable symbols.

## Compatibility and deprecation policy

`openextract` follows semantic-versioning intent, with extra care while the
project is still pre-1.0:

- **Public API:** `openextract.__all__` is the public Python API. The documented
  CLI arguments and exit codes, supported optional extras, documented environment
  variables, and documented input/output behavior are also user-facing
  compatibility surfaces.
- **Private API:** modules, functions, classes, and constants whose names start
  with `_` are internal unless they are explicitly documented here. They may
  change without a deprecation period.
- **Patch releases:** should fix bugs, documentation, packaging, provider error
  classification, or security issues without intentionally breaking public API.
- **Minor releases before 1.0:** may make breaking public API changes when they
  are needed for correctness, security, or a clearer long-term contract. These
  changes must be called out in `CHANGELOG.md` as breaking changes.
- **Major releases after 1.0:** are the normal place for breaking public API
  removals or incompatible behavior changes.

Deprecated public APIs should remain available until at least the next minor
release before `1.0`, unless keeping them would create a security, correctness,
or maintenance risk. After `1.0`, deprecated public APIs should remain available
until the next major release. Deprecations should be documented in
`CHANGELOG.md` with the replacement path and the earliest expected removal
version when that is known.

Provider behavior depends partly on `pydantic-ai` and provider SDKs. Upstream
model availability, credential requirements, supported media types, token usage
reporting, and provider-specific error shapes can change outside an
`openextract` release. `openextract` aims to keep its own public contract stable,
but provider-specific compatibility notes may be updated as upstream behavior
changes.

Python support follows `requires-python` in `pyproject.toml`; the current
supported versions are Python 3.12 and 3.13, both exercised in CI. Python 3.10
and 3.11 are not supported: retaining the 3.12 minimum avoids carrying syntax
compatibility changes for versions outside the project's declared support
window. Dropping support for a Python minor version is a breaking change and
should be announced in `CHANGELOG.md`.

## Security

### URL fetching and SSRF

When `input_file` is an `http://` or `https://` URL, `openextract` fetches it
with host validation, redirect re-checks, and configurable timeout/redirect
limits. Summary:

- Supported schemes: `http://`, `https://`
- Non-public hosts (private, loopback, link-local/metadata, multicast, reserved)
  are refused unless `OPENEXTRACT_ALLOW_PRIVATE_URLS` is set
- Hosts are re-validated at every redirect hop
- `OPENEXTRACT_URL_TIMEOUT` (default `30`) and `OPENEXTRACT_MAX_REDIRECTS`
  (default `10`) tune fetch behavior
- `OPENEXTRACT_MAX_INPUT_BYTES` (default `52428800`) caps each resolved input,
  including streamed URL bodies with missing or incorrect length headers

Full model, remaining risk boundaries (including DNS rebinding), and reporting
process: [SECURITY.md](SECURITY.md#url-input-security-model).

## Development

```bash
git clone https://github.com/Mellow-Artificial-Intelligence/openextract.git
cd openextract
uv sync --dev

uv run pytest --cov=openextract            # tests + coverage
uv run ruff check .                        # lint (Astral ruff)
uv run ruff format --check .               # format check
uv run ty check                            # types (Astral ty)
```

CI runs the test suite on every PR and fails if total coverage drops below 100%.

To score extraction quality on [ExtractBench](https://github.com/run-llama/ExtractBench)
with any model:

```bash
uv run python scripts/extractbench.py --model openai:gpt-5 --test
```

See [docs/extractbench.md](docs/extractbench.md) for how to run it and the
latest local 6-doc smoke status.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

## Roadmap

The project roadmap lives in the [GitHub Wiki](https://github.com/Mellow-Artificial-Intelligence/openextract/wiki/Roadmap).

## License

[MIT](LICENSE) &copy; Cole McIntosh
