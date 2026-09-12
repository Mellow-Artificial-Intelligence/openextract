# openextract examples

Runnable scripts showing common ways to use openextract. Each example prints JSON from a validated Pydantic model.

Examples use **OpenAI**, **Anthropic**, and **xAI** so you can see how provider identifiers map to real calls. Set the matching API keys in `.env`; the shared example helper loads it before selecting a model.

| Provider   | Model identifier              | Environment variable   |
| ---------- | ----------------------------- | ---------------------- |
| OpenAI     | `openai:gpt-5.5`              | `OPENAI_API_KEY`       |
| Anthropic  | `anthropic:claude-opus-4-8`  | `ANTHROPIC_API_KEY`    |
| xAI        | `xai:grok-4.3`                | `XAI_API_KEY`          |

Install provider extras as needed: `openextract[openai]`, `openextract[anthropic]`, `openextract[xai]`, or `openextract[all]`.

`openai:` model identifiers use the Responses API by default. Use
`openai-chat:` only when a model specifically requires Chat Completions.

Set `OPENEXTRACT_MODEL` to override every example with a single model (useful for CI or one-off testing).

If a provider SDK is missing, the Python API raises `ProviderNotInstalledError`
with a `pip install 'openextract[...]'` hint for known model prefixes. The CLI
prints the same error to stderr and exits `6`.

## Prerequisites

```bash
uv sync --dev
```

## Run everything

```bash
uv run python -m examples.run_all
```

`advanced/error_handling.py` runs without API keys. The rest need the API key for their assigned provider (see table below), or set `OPENEXTRACT_MODEL` to run all with one model.

## Examples by use case

| Directory | Script | Provider | What it demonstrates |
| --------- | ------ | -------- | -------------------- |
| `basic/` | `local_file.py` | OpenAI | `extract()` with a local path (`--fixture` uses the sample image) |
| `basic/` | `bytes_input.py` | Anthropic | `extract()` with `bytes` + `media_type` |
| `basic/` | `url_extract.py` | xAI | `extract()` with a public HTTPS URL |
| `images/` | `document_summary.py` | xAI | Summarize a document page image |
| `images/` | `receipt_extraction.py` | Anthropic | Receipt-style fields from an image |
| `documents/` | `invoice_extraction.py` | Anthropic | Invoice schema from PDF or image (`--fixture`) |
| `batch/` | `batch_extract.py` | OpenAI | Concurrent `extract_many()` |
| `advanced/` | `swarm_extract.py` | OpenAI + Anthropic | Parallel agents on one input via `extract_swarm_with_results()` |
| `batch/` | `stream_batch_extract.py` | TestModel | `iter_extract_many_async` completion order vs `extract_many` input order |
| `async/` | `async_extract.py` | Anthropic | `extract_async()` |
| `advanced/` | `extract_with_usage.py` | xAI | `extract_with_usage()` and token counts |
| `advanced/` | `extract_with_citations.py` | TestModel | `cite=True` on `extract_with_usage` / `extract_many_with_results` |
| `advanced/` | `retry_extract.py` | OpenAI | `max_retries` / `retry_backoff` |
| `advanced/` | `reusable_sessions.py` | TestModel | Sync/async sessions and dependency-injected agents |
| `advanced/` | `extraction_styles.py` | TestModel | `style='direct'` vs `search` / `code` |
| `advanced/` | `error_handling.py` | — | Catching `UrlFetchError` (no model call) |
| `audio/` | `meeting_notes.py` | xAI | Audio → structured meeting notes (bring your own file) |
| `cli/` | `schemas.py` | — | Pydantic models for CLI `--schema` |

### Quick commands

```bash
# OpenAI — local file
uv run python -m examples.basic.local_file --fixture

# Anthropic — bytes + explicit media type
uv run python -m examples.basic.bytes_input

# xAI — public URL
uv run python -m examples.basic.url_extract

# xAI — token usage
uv run python -m examples.advanced.extract_with_usage --fixture

# TestModel — cite=True (no API key; parser-backed bbox needs openextract[pdf])
uv run python -m examples.advanced.extract_with_citations

# OpenAI — batch
uv run python -m examples.batch.batch_extract

# Anthropic via CLI (from repo root)
PYTHONPATH=. uv run openextract examples/fixtures/document_page.png \
  --schema examples.cli.schemas:DocumentInfo \
  --model anthropic:claude-opus-4-8 \
  --instructions "Two-sentence summary and primary language."

# Batch via CLI, keeping per-item failures in the JSON output
PYTHONPATH=. uv run openextract examples/fixtures/document_page.png missing.png \
  --schema examples.cli.schemas:DocumentInfo \
  --model xai:grok-4.3 \
  --continue-on-error
```

With `--continue-on-error`, successful items and per-item errors are written as
one JSON array on stdout. If any item fails, the CLI also writes a warning to
stderr and exits `7`; without the flag, the batch stops at the first failure.

### Audio

```bash
uv run python -m examples.audio.meeting_notes /path/to/meeting.mp3
```

Requires `XAI_API_KEY` (or override with `OPENEXTRACT_MODEL`).

### Your own PDF invoice

```bash
uv run python -m examples.documents.invoice_extraction ./invoices/acme-q4.pdf
```

Uses Anthropic by default. Vision-capable models work with the image fixture; PDFs may need provider-specific balance or limits.
