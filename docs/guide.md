---
layout: page
title: Guide
description: How to extract structured data with openextract — install, inputs, styles, sessions, batch, errors, and CLI.
---

# Guide

`openextract` turns a document, image, audio file, or video into a **validated Pydantic model**. You bring a schema, a model identifier, and media; you get a typed object back.

This page is the how-to. Use [API reference](api-reference.md) for signatures, [For agents](agents.md) if you are generating code against the library, and [llms.txt](llms.txt) for a machine-readable contract.

## Install

```bash
uv add openextract
# or
pip install openextract
```

The base package does **not** install provider SDKs. Add the extra for the model you call:

```bash
pip install 'openextract[openai]'
pip install 'openextract[anthropic]'
pip install 'openextract[xai]'
pip install 'openextract[all]'
```

Requires **Python 3.12+**. Missing extras raise `ProviderNotInstalledError` with a `pip install 'openextract[...]'` hint.

## First extraction

```python
from pydantic import BaseModel
from openextract import extract


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str


invoice = extract(
    schema=Invoice,
    model="openai:gpt-5",
    input_file="./invoices/acme.pdf",
    instructions="Extract vendor, total, and currency.",
)

print(invoice.vendor, invoice.total)
```

`invoice` is an `Invoice` instance — not a dict, not a JSON string.

Set provider credentials in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, …). The **CLI and bundled examples** load `.env`; the **Python library does not**.

## Inputs

Every extract API accepts:

| Form | Notes |
| --- | --- |
| Path string or `pathlib.Path` | MIME type is guessed from the name; override with `media_type`. |
| `http://` or `https://` URL | Fetched with SSRF host checks. See [SECURITY.md](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/SECURITY.md#url-input-security-model). |
| `bytes` or a binary file-like object | **`media_type` is required.** |
| `ExtractionInput` | Per-item `media_type` and optional safe `name` for batch diagnostics. |

```python
from pathlib import Path
from openextract import ExtractionInput, extract

extract(schema=Invoice, model="openai:gpt-5", input_file=Path("bill.pdf"))
extract(schema=Invoice, model="openai:gpt-5", input_file=pdf_bytes, media_type="application/pdf")
extract(
    schema=Invoice,
    model="openai:gpt-5",
    input_file=ExtractionInput(pdf_bytes, media_type="application/pdf", name="bill.pdf"),
)
```

Each input is capped at **50 MiB** (`52428800` bytes) before a model call. Override with `max_input_bytes` or `OPENEXTRACT_MAX_INPUT_BYTES`. Oversized inputs raise `InputTooLargeError`.

Private/loopback URL hosts are refused unless `OPENEXTRACT_ALLOW_PRIVATE_URLS` is set. Tune `OPENEXTRACT_URL_TIMEOUT` (default `30`) and `OPENEXTRACT_MAX_REDIRECTS` (default `10`).

## Extraction styles

`style` selects how the model inspects the media:

| Style | Behavior | Requirements |
| --- | --- | --- |
| `direct` (default) | Send the resolved media to the model in one shot. | Any supported modality. |
| `search` | Grep/read a **text** document with sandboxed file tools. | `pydantic-ai-harness`; UTF-8 text only. |
| `code` | Write Python against a workspace copy of the **text**. | `pydantic-ai-harness[codemode]`; UTF-8 text only. |

```python
extract(schema=Invoice, model="openai:gpt-5", input_file="notes.txt", style="search")
```

PDFs, Office files, images, audio, and video stay on `direct`. The CLI flag is `--style`. Written against `pydantic-ai-harness` 0.18.x.

## Sessions

For repeated calls with the same schema and model, use `Extractor` / `AsyncExtractor`. One agent and HTTP client are built on enter and closed on exit.

```python
from openextract import Extractor, RetryPolicy

with Extractor(
    Invoice,
    "openai:gpt-5",
    instructions="Extract vendor, total, and currency.",
    retry_policy=RetryPolicy(max_retries=3),
) as extractor:
    q3 = extractor.extract("./invoices/q3.pdf")
    q4, usage = extractor.extract_with_usage("./invoices/q4.pdf")
```

`Extractor` is thread-bound and not thread-safe. `AsyncExtractor` is bound to one event loop; concurrent awaits on that loop are fine. Pass a configured `pydantic_ai.models.Model` as `model=`, or a fully configured `Agent` as `agent=` (mutually exclusive with `model=`; not combinable with `search`/`code`).

## Retries and usage

`max_retries` defaults to `0`. Only **transient** `ModelError` values retry (timeouts, rate limits, supported 5xx). Auth and invalid-request failures fail immediately. Backoff is exponential with jitter, capped by `retry_max_backoff` (default 60s). A bounded provider `Retry-After` wins when present.

The input is resolved **once**; retries reuse the same bytes, prompt, and agent.

```python
from openextract import extract_with_usage

invoice, usage = extract_with_usage(
    schema=Invoice,
    model="openai:gpt-5",
    input_file="bill.pdf",
    max_retries=3,
)
print(usage.input_tokens, usage.output_tokens, usage.total_tokens)
```

## Citations

`cite=True` asks the model for per-field source evidence (a quote and
1-indexed page). PDFs are parsed locally first; long documents are chunked by
page so the model never sees the whole parse as one prompt.

One-page windows remap `page=1` / missing pages onto the document page the
model actually saw. If the quote is paraphrased or the page is wrong, the page
is backfilled from the extracted field value. `bbox` is parser-backed only —
attached when the local parser matches the span, never invented. Boxes need
`openextract[pdf]`.

`extract()` still returns the schema instance. Read citations from
`ExtractionResult.citations` on `extract_many_with_results*` /
`extract_swarm_with_results*`.

```python
from openextract import extract_many_with_results

results = extract_many_with_results(
    schema=Invoice, model="openai:gpt-5", input_files=["bill.pdf"], cite=True
)
for citation in results[0].citations:
    print(citation.as_dict())
```

Default is off: no extra instructions or schema wrap. Citations never retain
raw media or credentials. See [ExtractBench](extractbench.md) for how these
map onto grounding scores.

## Batch

| API | Returns | Order | Use when |
| --- | --- | --- | --- |
| `extract_many` / `extract_many_async` | `list[T]` | Input order | You want the full batch. |
| `iter_extract_many_async` | `(index, result)` as items finish | Completion order | Large/generator inputs; start work early. |
| `extract_many_with_results*` | `list[ExtractionResult[T]]` | Input order | Per-item usage, attempts, duration, sanitized source. |

```python
from openextract import extract_many, iter_extract_many_async, total_usage

results = extract_many(
    schema=Invoice,
    model="openai:gpt-5",
    input_files=["a.pdf", "b.pdf"],
    max_concurrency=5,
    return_exceptions=True,
)

async for index, item in iter_extract_many_async(
    schema=Invoice,
    model="openai:gpt-5",
    input_files=paths,
    return_exceptions=True,
):
    ...
```

`extract_many` **cannot** run inside an active event loop (`RuntimeError`); use `extract_many_async` or the iterator. Default fail-fast cancels outstanding work. `return_exceptions=True` keeps going and puts errors in the result position. `total_usage(results)` sums successful `ExtractionResult` usage.

See [`examples/batch/stream_batch_extract.py`](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/examples/batch/stream_batch_extract.py).

## Swarms

`extract_many` scales across inputs; a swarm scales across *agents* on **one**
input. The file is loaded once, the agents run concurrently, and their outputs
are reduced into a single validated object.

```python
from openextract import SwarmMember, extract_swarm, extract_swarm_with_results

# Three passes of one model, merged.
invoice = extract_swarm(
    schema=Invoice, agents="openai:gpt-5", input_file="invoice.pdf", size=3
)

# Two models cross-checking each other, majority per field.
invoice = extract_swarm(
    schema=Invoice,
    agents=["openai:gpt-5", "anthropic:claude-opus-4-8"],
    input_file="invoice.pdf",
    reduce="vote",
)

# Per-agent instructions and styles, plus per-agent diagnostics.
swarm = extract_swarm_with_results(
    schema=Invoice,
    agents=[
        SwarmMember("openai:gpt-5", instructions="Line items only.", style="search"),
        SwarmMember("openai:gpt-5", instructions="Totals and dates only."),
    ],
    input_file="invoice.txt",
)
print(swarm.output, swarm.usage, swarm.reduce)
```

`reduce` is `merge` (union lists, fill fields), `vote` (majority per field), or
`first`. A swarm survives partial failure: agents that raised appear in
`swarm.agents`, and only every agent failing raises. `size` is capped at 16,
and `max_concurrency` defaults to `min(5, agents)`.

Reach for a swarm when one pass under-recalls — a long document, a wide schema,
or a result worth cross-checking with a second model. One document one model
handles well does not need one.

See [`examples/advanced/swarm_extract.py`](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/examples/advanced/swarm_extract.py).

## Agents

A swarm member is a model plus instructions plus a style. An agent packages
that once so callers import a specialist instead of re-describing it, and
`subagents` compose several into one.

```python
from pydantic import BaseModel
from openextract import define_agent, extract_swarm

class Invoice(BaseModel):
    vendor: str
    total: float

line_items = define_agent("Line items", model="openai:gpt-5", instructions="Rows only.")
totals = define_agent("Totals", model="openai:gpt-5", instructions="Totals and dates.")
invoices = define_agent("Invoices", output_schema=Invoice, subagents=[line_items, totals])

invoice = extract_swarm(schema=Invoice, agents=invoices, input_file="invoice.pdf")
```

An agent works in `extract` too. When it declares an `output_schema`, the schema
argument is redundant and can be dropped:

```python
from openextract import extract

invoice = extract(invoices, "invoice.pdf")          # agent supplies model + schema
invoice = extract(Invoice, line_items, "invoice.pdf")  # explicit schema, agent as the model
```

A single-model agent runs as an ordinary one-shot call. An agent with subagents
(or a remote endpoint) fans out into a swarm and its outputs are merged, so the
same call scales with the agent rather than the call site.

Agents also load from disk, so a repository can ship them next to the code:

```text
agents/invoices/
  agent.py          # agent = define_agent(...)
  instructions.md   # used when agent.py declares none
  subagents/
    line_items.py
    totals.py
```

```python
from openextract import load_agent, load_agents

invoices = load_agent("agents/invoices")        # directory
totals = load_agent("agents/totals.py")          # file
imported = load_agent("my_package.agents:invoices")   # module:attribute
both = load_agents("agents/invoices,agents/receipts")
```

### Remote agents

`define_remote_agent` points at an HTTP endpoint instead of a local model. It
posts the JSON Schema, the base64 media, and the style, and reads back
`{"output": ...}`.

```python
import os

from openextract import define_remote_agent, extract_swarm
from openextract.auth import bearer

remote = define_remote_agent(
    url=lambda: os.environ["INVOICE_AGENT_URL"],
    description="Hosted invoice reader",
    auth=bearer(lambda: os.environ["INVOICE_AGENT_TOKEN"]),
)

invoice = extract_swarm(schema=Invoice, agents=["openai:gpt-5", remote], input_file="invoice.pdf")
```

`url`, `headers`, and `auth` may be callables (sync or async) resolved per
request, so rotated credentials are picked up without redefining the agent.
Transport failures and 408/409/425/429/5xx responses raise a retryable
`RemoteAgentError` and honour `max_retries`. The endpoint host goes through the
same SSRF allowlist as document URLs; a local agent server needs
`OPENEXTRACT_ALLOW_PRIVATE_URLS=1`.

## Errors

All public exceptions subclass `ExtractionError`.

| Exception | Typical cause | CLI exit |
| --- | --- | --- |
| `UrlFetchError` | Network, HTTP, or SSRF refusal | `2` |
| `SchemaValidationError` | Model output did not match the schema | `3` |
| `ModelError` | Provider/model API failure (`provider`, `status_code`, `retryable`, `retry_after`) | `4` |
| `InputTooLargeError` | Over the byte cap | `5` |
| `ProviderNotInstalledError` | Missing extra or harness package | `6` |
| `ValueError` | Bad options (`max_retries`, `max_concurrency`, missing `media_type` for bytes) | `1` |

```python
from openextract import ExtractionError, ModelError, UrlFetchError, extract

try:
    extract(schema=Invoice, model="openai:gpt-5", input_file=url)
except UrlFetchError:
    ...
except ModelError as exc:
    print(exc.provider, exc.status_code, exc.retryable, exc.retry_after)
except ExtractionError:
    ...
```

## CLI

```bash
openextract ./bill.pdf \
  --schema mypkg.schemas:Invoice \
  --model openai:gpt-5 \
  --instructions "Extract vendor, total, and currency."
```

- Success goes to **stdout**; errors to **stderr**.
- `--schema` is `module:ClassName` on `PYTHONPATH`.
- Batch: pass multiple paths. `--continue-on-error` emits per-item errors inline and exits `7` if any failed.
- `--usage` is single-input only.
- `--cite` adds a `citations` array to JSON/jsonl (`field`, `quote`, `page`,
  optional `bbox`). Same as `cite=True` on the Python API.
- `--style`, `--max-retries`, `--max-input-bytes` match the Python API.

Full stdout/stderr/exit-code contract: [CLI](cli.md).

## Models

`model` uses the `pydantic-ai` prefix convention (`openai:gpt-5`, `anthropic:claude-sonnet-4`, `xai:grok-4.3`, `ollama:llama3`, …). `openai:` routes through the **Responses API** by default; use `openai-chat:` for Chat Completions.

Capability matrix and credentials: [Providers](providers.md).

## What to read next

- [For agents](agents.md) — public surface, do-nots, and a decision tree for generated code.
- [API reference](api-reference.md) — every public signature (CI-checked).
- [Troubleshooting](troubleshooting.md) — extras, URLs, retries, batch choice.
- [Examples](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/examples/README.md) — runnable scripts.
- [Changelog](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/CHANGELOG.md)
