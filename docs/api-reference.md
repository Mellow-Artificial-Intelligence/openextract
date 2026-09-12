---
layout: page
title: API reference
---

# API reference

This is the canonical reference for the public Python API. CI compares every
function heading below with the installed callable signature; update this page
in the same change as any public signature.

How-to: [Guide](guide.md). Integration contract for generated code: [For agents](agents.md).

## Extraction

### `Extractor(schema, model=None, instructions=None, *, style='direct', agent=None, model_settings=None, timeout=None, instrument=False, retry_policy=None, max_input_bytes=None, url_timeout=None, cite=False)`

Reusable synchronous extraction session. Enter it with `with`; then call
`extract(input_file, *, media_type=None)` or
`extract_with_usage(input_file, *, media_type=None)`. The agent, model provider
client, and URL-fetch client are constructed once and closed on context exit.

`model` accepts either a known model string or a configured
`pydantic_ai.models.Model`. `style` selects how the model inspects the input
(`direct`, `search`, or `code`; see [Common arguments](#common-arguments)) and
applies to every call in the session; `search` and `code` sessions build their
harness agent and temporary workspace once on enter and remove both on exit.
As an advanced alternative, `agent` accepts a fully
configured Pydantic AI `Agent`; it is mutually exclusive with `model`, and its
output is revalidated against `schema`. Non-`direct` styles and `cite=True`
cannot be combined with an injected `agent`. `cite=True` wraps the session
schema so the model returns per-field citations; `extract` still returns the
schema instance.

### `AsyncExtractor(schema, model=None, instructions=None, *, style='direct', agent=None, model_settings=None, timeout=None, instrument=False, retry_policy=None, max_input_bytes=None, url_timeout=None, cite=False)`

Async session counterpart. Enter it with `async with`; then await `extract` or
`extract_with_usage`. It shares one async HTTP client and one agent across
calls, including concurrent calls made on the entering event loop. Close it
manually with `aclose()` only when a context manager is impractical.

### `RetryPolicy(max_retries=0, backoff=1.0, max_backoff=60.0)`

Frozen session retry configuration. Only transient `ModelError` failures are
retried. The backoff and bounded provider `Retry-After` behavior match the
one-shot function arguments.

### ExtractionStyle

`direct` (default) sends resolved media to the model in one shot. `search` and
`code` are text-only agentic styles powered by
[Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/). Pass the enum
(`ExtractionStyle.SEARCH`) or the string (`"search"`). Non-text inputs and a
missing harness extra fail before the model call.

### `extract(schema, model, input_file=None, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

`model` also accepts an [agent](#agents), and an agent that declares an
`output_schema` can be passed as `schema` instead — `extract(invoices, "doc.pdf")`
is the same call as `extract(Invoice, invoices, "doc.pdf")`. An agent that
resolves to one local model runs as a normal one-shot call using the agent's
model, instructions, and style; an agent with subagents or a remote endpoint
runs as a [swarm](#swarm) and its outputs are reduced. Omitting `input_file`
raises `ValueError`; the parameter is optional only so the agent form can shift
it left. The same applies to the three siblings below.

Extract one input synchronously and return an instance of `schema`.
`cite=True` asks the model for per-field source spans (page and quote). PDFs
are parsed locally; long documents are extracted in page windows and merged.
Boxes come from parser spans, not the model. The return type stays the schema
instance; citations land on [`ExtractionResult`](#extractionresult) from the
`*_with_results` APIs.

### `extract_async(schema, model, input_file=None, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

Async counterpart to `extract`. It uses `Agent.run` and returns an instance of
`schema`.

### `extract_with_usage(schema, model, input_file=None, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

Extract one input synchronously and return `(output, Usage)`. It has the same
retry behavior as `extract`; `Usage` describes the successful model call.

### `extract_with_usage_async(schema, model, input_file=None, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

Async counterpart to `extract_with_usage`; returns `(output, Usage)`.

### `extract_many(schema, model, input_files, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

Run concurrent extractions from synchronous code. Results preserve input order.
When `return_exceptions=True`, per-item exceptions appear in the result list.
Do not call this function from a running event loop; use
`extract_many_async` instead.

### `extract_many_async(schema, model, input_files, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

Async counterpart to `extract_many`; it has the same arguments, result ordering,
and per-item retry behavior.

### `iter_extract_many_async(schema, model, input_files, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

Return an async iterator of `(input_index, result)` pairs in **completion
order**. Inputs are consumed lazily, at most `max_concurrency` items are
scheduled, and results are available before the complete batch finishes.
Simultaneous completions are yielded in input-index order.

This is the streaming counterpart to `extract_many_async`, which waits for every
item and returns a list in **input order**. Keep the original `input_files`
sequence (or a name on `ExtractionInput`) if you need to map `input_index` back
to a path.

With `return_exceptions=False`, the first failure cancels and awaits outstanding
work before being raised. With `return_exceptions=True`, item exceptions are
yielded in the result position and streaming continues.

```python
async for index, result in iter_extract_many_async(
    schema=PdfInfo,
    model="openai:gpt-5",
    input_files=paths,
    return_exceptions=True,
    max_concurrency=5,
):
    if isinstance(result, Exception):
        print(f"{index} failed: {result}")
    else:
        print(index, result.summary)
```

### `extract_many_with_results(schema, model, input_files, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

Run a batch and return per-item [`ExtractionResult`](#extractionresult) objects
instead of bare schema instances. It has the same arguments, input ordering,
concurrency, and retry semantics as `extract_many`, and each result carries
token usage, attempt count, duration, model/media metadata, and a sanitized
source label. With `return_exceptions=True`, failed items appear as
`Exception` values in place. Use [`total_usage`](#total_usageresults) to
aggregate token usage across the returned results.

### `extract_many_with_results_async(schema, model, input_files, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

Async counterpart to `extract_many_with_results`; it has the same arguments,
result ordering, and per-item retry behavior.

### `total_usage(results)`

Sum token usage across batch extraction results, for example the list returned
by `extract_many_with_results` or `extract_many_with_results_async`. Returns a
single [`Usage`](#usage) whose fields are the totals of the successful items.

## Swarm

A swarm runs several agents over one input and reduces their outputs. The input
is fetched and decoded once, so a swarm costs one load and N model calls. Each
agent is told its position in the swarm so it works independently instead of
assuming a peer covered a section.

Use a swarm when one pass under-recalls: a long document, a schema with many
optional fields, or a job worth cross-checking with a second model. One
document that one model handles well does not need one.

### `SwarmMember`

`SwarmMember(model, instructions=None, style=None)` is one agent, where `model`
is a model identifier, a configured pydantic-ai `Model`, or a `RemoteAgent`. Its
`instructions` and `style` override the swarm-wide values, so a `search` reader
and a `direct` reader can share the same swarm. A bare model identifier or a
configured pydantic-ai `Model` is accepted anywhere a `SwarmMember` is.

### `SwarmResult`

Returned by `extract_swarm_with_results*`. `output` is the reduced instance,
`agents` holds each agent's [`ExtractionResult`](#extractionresult) or the
exception it raised in agent order, `usage` sums the successful agents, and
`reduce` is the strategy that produced `output`.

### `resolve_swarm_members(agents, size=None)`

Expand the `agents` argument into one `SwarmMember` per agent. A single agent
plus `size` fans it out `size` times (1..16); a list is used as-is and `size`
may not contradict its length. [Defined agents](#agents) are flattened first,
so a parent with subagents contributes one member per leaf.

### `extract_swarm(schema, agents, input_file, instructions=None, *, size=None, style='direct', reduce='merge', media_type=None, max_input_bytes=None, max_concurrency=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

Run the agents concurrently over one input and return the reduced schema
instance. `max_concurrency` defaults to `min(5, agents)`. Agent failures are
tolerated as long as one agent succeeds; if every agent fails, the first
failure is raised. Raises `RuntimeError` from a running event loop.

### `extract_swarm_async(schema, agents, input_file, instructions=None, *, size=None, style='direct', reduce='merge', media_type=None, max_input_bytes=None, max_concurrency=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, cite=False)`

Async counterpart to `extract_swarm`.

### `extract_swarm_with_results(schema, agents, input_file, instructions=None, *, size=None, style='direct', reduce='merge', media_type=None, max_input_bytes=None, max_concurrency=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, on_agent_start=None, on_agent=None, cite=False)`

Same run, returning a [`SwarmResult`](#swarmresult). `on_agent_start(index,
total)` and `on_agent(index, total, result)` report progress as agents start
and finish.

### `extract_swarm_with_results_async(schema, agents, input_file, instructions=None, *, size=None, style='direct', reduce='merge', media_type=None, max_input_bytes=None, max_concurrency=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0, on_agent_start=None, on_agent=None, cite=False)`

Async counterpart to `extract_swarm_with_results`.

## Agents

An agent packages what a swarm member would otherwise repeat at every call
site — model, style, instructions, and output schema — so a caller imports a
specialist instead of re-specifying it. Agents compose: a parent with
`subagents` flattens into one swarm member per leaf, so an agent is accepted
anywhere `agents` is.

### `define_agent(description, *, model=None, style=None, instructions=None, output_schema=None, subagents=())`

Define a local agent. `description` is required. `model` is optional only when
`subagents` is non-empty (a pure group). `output_schema` must be a
`pydantic.BaseModel` subclass. Returns a frozen `DefinedAgent`.

### `define_remote_agent(url, description, *, auth=None, headers=None, path='/extract', output_schema=None)`

Define an agent served over HTTP. Returns a frozen `RemoteAgent`. `url` may be
a callable (sync or async) so a rotating deployment URL is resolved per
request; `headers` and `auth` likewise. See [Remote agent
protocol](#remote-agent-protocol).

### `flatten_agent(agent)`

Expand one agent into the `SwarmMember` list it contributes: its own model
first (when it has one), then each subagent's members, depth first.

### `resolve_output_schema(agent)`

Return the agent's `output_schema`, or raise `ValueError` naming the agent when
it declared none.

### `load_agent(spec)`

Load one agent from a directory, a Python file, or a `module:attribute` path. A
`DefinedAgent` / `RemoteAgent` passes through unchanged. A file or module must
expose the agent as a module-level name (`agent` for files, the named attribute
for `module:attribute`).

### `load_agents(spec)`

Load several agents from a comma-separated string or a sequence of specs.

### `load_agent_directory(directory)`

Load an agent from a directory laid out as:

```text
invoices/
  agent.py          # defines `agent = define_agent(...)`
  instructions.md   # used when the agent declared no instructions
  subagents/
    line_items.py   # each defines `agent = ...`
    totals/         # nested directories load recursively
```

Entries beginning with `.` or `_` are skipped and subagents load in sorted
order. A directory with subagents but no `agent.py` becomes a group agent named
after the directory; a directory with exactly one subagent resolves to it.

### Remote agent protocol

`RemoteAgent` posts to `url + path` with
`{"schema": <JSON Schema>, "input": {"data": <base64>, "mediaType": <MIME>}, "instructions": ..., "style": ...}`
and reads `{"output": ..., "usage": {...}}` — a bare object is treated as the
output, and `{"error": "..."}` (at any status) raises `RemoteAgentError`.
`usage` is accepted in either `inputTokens` or `input_tokens` spelling. HTTP
408/409/425/429/5xx and transport failures are retryable and honour
`max_retries`.

The agent URL goes through the same SSRF host allowlist as document URLs, so a
local agent server needs `OPENEXTRACT_ALLOW_PRIVATE_URLS=1`.

### `openextract.auth`

Outbound auth header providers for `define_remote_agent(auth=...)`. Each is
resolved per request, so a refreshed credential is picked up without redefining
the agent.

| Helper | Header |
| --- | --- |
| `bearer(token)` | `Authorization: Bearer <token>` |
| `basic((username, password))` | `Authorization: Basic <base64>` |
| `vercel_oidc()` | Bearer token read from `VERCEL_OIDC_TOKEN` |

`bearer` and `basic` accept the value itself or a callable (sync or async)
returning it.

## Swarm reduce

A swarm runs several agents over one input and folds their outputs into a
single result. The fold strategy is `SwarmReduce`; the reducers are public so a
caller can combine outputs it gathered itself.

### SwarmReduce

`merge` (default) unions list fields and fills each scalar field from the first
agent that produced a value. `vote` keeps the most frequent non-empty value per
field, breaking ties toward the earlier agent; list fields have no majority, so
they fall back to `merge`. `first` returns the first successful agent's output
untouched. Pass the enum (`SwarmReduce.VOTE`) or the string (`"vote"`).

### `normalize_reduce(reduce='merge')`

Return a valid `SwarmReduce` for an enum member or string, or raise
`ValueError` naming the allowed strategies.

### `reduce_outputs(values, reduce='merge')`

Fold a sequence of same-schema Pydantic model instances into one instance.
`merge` and `vote` reduce the dumped payloads and re-validate the combined
value, so the return value always satisfies the schema; a combination that no
longer validates raises `SchemaValidationError`. An empty `values` raises
`ValueError`.

## Choosing a batch API

| API | Returns | Order | When to use |
| --- | --- | --- | --- |
| `extract_many` / `extract_many_async` | `list[T]` (or exceptions in place) | Input order | You want the full batch before continuing. |
| `iter_extract_many_async` | `(input_index, result)` as items finish | Completion order | Large or generator inputs; start work before the last item completes. |
| `extract_many_with_results` / `_async` | `list[ExtractionResult[T]]` | Input order | Per-item usage, attempts, duration, and sanitized source labels. |

`extract_many` and `extract_many_with_results` raise `RuntimeError` from a
running event loop; use the `_async` siblings or the iterator instead. See
[`examples/batch/stream_batch_extract.py`](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/examples/batch/stream_batch_extract.py)
for a side-by-side run.

## Input and result contracts

### `ExtractionInput`

A frozen dataclass wrapping a single media source with optional per-item media
metadata. Passing one to any extraction API (or mixing them into a batch) is
equivalent to passing the raw source directly.

| Field | Type | Description |
| --- | --- | --- |
| `source` | `str \| os.PathLike[str] \| bytes \| BinaryIO` | Local path, HTTP(S) URL, `Path`, raw `bytes`, or binary file-like object. |
| `media_type` | `str \| None` | Per-item MIME type. Required for `bytes` and file-like sources when no batch-wide override is supplied. |
| `name` | `str \| None` | Optional safe source label recorded on `ExtractionResult.source`. |

Batch item media types resolve per item: an `ExtractionInput.media_type` wins
over the batch-wide `media_type` argument, which remains the fallback for raw
items.

### `Citation`

A frozen dataclass for one field's source evidence. Mapped onto ExtractBench
`FieldCitation` via `Citation.as_field_citation()` (`field` → `field_path`,
`quote` → `reference_text`, plus `page` and optional `bbox`).

| Field | Type | Description |
| --- | --- | --- |
| `field` | `str` | Dotted schema path (`vendor`, `lines[0].qty`). |
| `quote` | `str \| None` | Verbatim source span, when present. |
| `page` | `int \| None` | 1-indexed page. Required to emit an ExtractBench citation. |
| `bbox` | `tuple[float, float, float, float] \| None` | Normalized COCO `(x, y, width, height)` in `[0, 1]`. Kept only when a local parser matched the span; never invented. |

`as_dict()` is the JSON shape for CLI and other consumers:
`{field, quote, page, bbox}` with `bbox` as a list of four floats or `null`.
Quote-only citations are included. `as_field_citation()` returns `None` when
`page` is missing (those citations stay on `ExtractionResult` but cannot be
scored by ExtractBench).

### `ExtractionResult`

A frozen, generic dataclass returned by `extract_many_with_results*`. It never
retains raw media, credentials, query strings, fragments, or provider
internals; `source` is sanitized.

| Field | Type | Description |
| --- | --- | --- |
| `output` | `T` | The validated schema instance. |
| `usage` | [`Usage`](#usage) | Token usage from the successful model call. |
| `attempts` | `int` | Model-call attempts including retries; always `>= 1` on success. |
| `duration` | `float` | Wall-clock seconds for the item, including retries. |
| `model` | `str \| None` | Model identifier that produced the output, when known. |
| `media_type` | `str \| None` | Media type requested for the item, when provided. |
| `source` | `str \| None` | Sanitized source label; `None` for unnamed bytes/file-like inputs. |
| `warnings` | `tuple[str, ...]` | Extensible diagnostics channel; currently always empty. |
| `citations` | `tuple[Citation, ...]` | Per-field source spans when `cite=True`; empty otherwise. |

## Common arguments

| Argument | Type | Description |
| --- | --- | --- |
| `schema` | `type[BaseModel]` | Pydantic model class describing the desired output. |
| `model` | `str \| Model` | `pydantic-ai` model identifier or configured model instance. |
| `input_file` | `str \| os.PathLike[str] \| bytes \| BinaryIO \| ExtractionInput` | Local path, HTTP(S) URL, `Path`, bytes, binary file-like object, or `ExtractionInput`. |
| `instructions` | `str \| None` | Optional model guidance. |
| `style` | `ExtractionStyle \| str` | How the model inspects the input. `direct` (default) sends media in one shot. `search` gives the model sandboxed file tools (read/grep) against a text document. `code` lets the model write Python against a text document via [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/). `search` needs `pydantic-ai-harness`; `code` needs `pydantic-ai-harness[codemode]`. |
| `media_type` | `str \| None` | Required for bytes and file-like inputs without a per-item type; overrides inference for paths and URLs. Item-level `ExtractionInput.media_type` wins in batch calls. |
| `max_input_bytes` | `int \| None` | Per-input byte cap; `None` uses `OPENEXTRACT_MAX_INPUT_BYTES` or the 50 MiB default. |
| `max_retries` | `int` | Extra attempts after transient `ModelError`; defaults to `0`. |
| `retry_backoff` | `float` | Base seconds for exponential backoff with up to 25% jitter. |
| `retry_max_backoff` | `float` | Maximum delay, including provider `Retry-After`; defaults to `60.0`. |
| `cite` | `bool` | Ask the model for per-field citations. Default `False` (no extra instructions or schema wrap). Citations attach to `ExtractionResult`; `extract()` still returns the schema instance. |

Batch functions also accept:

| Argument | Type | Description |
| --- | --- | --- |
| `input_files` | `Iterable[str \| os.PathLike[str] \| bytes \| BinaryIO \| ExtractionInput]` | One input per extraction; items may carry their own media type. |
| `max_concurrency` | `int` | Positive maximum number of in-flight extractions; defaults to `5`. |
| `return_exceptions` | `bool` | Return per-item exceptions in place instead of failing fast. |

## `Usage`

`Usage` is a frozen dataclass returned by the two usage helpers.

| Field | Type | Description |
| --- | --- | --- |
| `input_tokens` | `int` | Prompt tokens consumed. |
| `output_tokens` | `int` | Completion tokens consumed. |
| `total_tokens` | `int` | Total reported tokens. |

## Configuration

The Python library reads provider configuration from the existing process
environment and does not load `.env` files. The `openextract` CLI and bundled
examples load `.env` explicitly as an application-level convenience.

Session `model_settings` are passed directly to Pydantic AI using its typed
`ModelSettings` contract. `timeout` is an explicit shortcut for the model
request timeout and overrides a `timeout` entry in `model_settings`.
`url_timeout` separately controls URL input fetching. `instrument=True` enables
Pydantic AI instrumentation; an `InstrumentationSettings` instance provides
fine-grained control.

`Extractor` is bound to the thread that enters it and is not thread-safe.
`AsyncExtractor` is bound to one event loop; concurrent calls within that loop
are supported. Neither session can be reopened after it is closed.
