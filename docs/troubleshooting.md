---
layout: page
title: Troubleshooting
---

# Troubleshooting

Short recovery steps for common setup and runtime failures. Exception names and
CLI exit codes match `src/openextract/_cli.py` and `src/openextract/exceptions.py`.

## Missing provider SDK extras

**Symptoms**

- Python: `ProviderNotInstalledError`
- CLI: stderr `error: ...`, exit code `6`

**Cause**

The base package does not install provider SDKs. Calling a model whose extra is
missing raises this error.

**Next step**

Install the hinted extra, for example:

```bash
pip install 'openextract[openai]'
# or
pip install 'openextract[xai]'
# or every provider
pip install 'openextract[all]'
```

`style='search'` needs `pydantic-ai-harness`; `style='code'` needs
`pydantic-ai-harness[codemode]`. The styles integration was written against
`pydantic-ai-harness` 0.18.x, which requires `pydantic-ai-slim` 2.x; the
repository's locked development environment cannot resolve it, so CI does not
exercise the live harness integration.

## Missing provider credentials

**Symptoms**

- Python: `ModelError` (or a provider SDK auth error wrapped as `ModelError`)
- CLI: exit code `4`

**Cause**

The provider SDK is installed, but no API key / cloud credentials are available.

**Next step**

Set the provider environment variable. The CLI and bundled examples load
`.env`; library callers should load application configuration explicitly. For
example: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`. See the [provider
matrix](providers.md) for the credential column.

## URL fetch failures

**Symptoms**

- Python: `UrlFetchError`
- CLI: exit code `2`

**Common causes**

- Host refused as non-public (SSRF protection)
- HTTP error / timeout / too many redirects
- DNS failure

**Next step**

1. Confirm the URL is `http://` or `https://` and publicly reachable.
2. For intentional localhost/internal fetches, either set
   `OPENEXTRACT_ALLOW_PRIVATE_URLS=1` (understand the risk) or fetch bytes
   yourself and pass `bytes` / a file-like object with `media_type`.
3. Tune `OPENEXTRACT_URL_TIMEOUT` / `OPENEXTRACT_MAX_REDIRECTS` if needed.
4. Read [SECURITY.md](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/SECURITY.md#url-input-security-model).

## Schema validation failures

**Symptoms**

- Python: `SchemaValidationError`
- CLI: exit code `3`

**Cause**

The model returned output that could not be validated against your Pydantic
schema. The exception message lists each failing field path and the type that
was expected, for example `lines[0].qty: expected int, got str`.
`exc.errors` is a tuple of `{field, expected, received}` mappings.

**Next step**

- Read the field paths in the error; those are the values to fix.
- Tighten `instructions` and keep the schema fields extractable from the media.
- Prefer capable models for the modality (for example vision for images).
- Retry with `max_retries` if the failure is intermittent model drift.

## Model API failures and retries

**Symptoms**

- Python: `ModelError`
- CLI: exit code `4`

**Cause**

Provider/model API failure (auth, rate limit, server error, unsupported media).

**Next step**

```python
extract(..., max_retries=3, retry_backoff=1.0, retry_max_backoff=60.0)
```

CLI:

```bash
openextract ./file.pdf --schema pkg:Model --model openai:gpt-5 \
  --max-retries 3 --retry-backoff 1.0 --retry-max-backoff 60.0
```

Retries apply only when `ModelError.retryable` is true. Rate limits, transient
transport failures, and supported 5xx responses retry; authentication,
permission, and invalid-request failures do not. Provider `Retry-After` values
take precedence over exponential backoff but remain bounded by
`retry_max_backoff`. Invalid option values raise `ValueError` (CLI exit `1`).

## Input exceeds the configured size limit

`InputTooLargeError` means openextract stopped reading an input before calling
the model. The default limit is 50 MiB per input. If a larger input is expected,
pass `max_input_bytes=...`, use `--max-input-bytes`, or set
`OPENEXTRACT_MAX_INPUT_BYTES`. Keep the smallest practical limit for untrusted
paths, URLs, streams, and batch jobs.

## CLI schema import errors

**Symptoms**

- CLI: stderr `error: ...`, exit code `1`

**Cause**

`--schema` must be `module:ClassName` pointing at a Pydantic `BaseModel`
subclass importable from the current `PYTHONPATH`.

**Next step**

```bash
# from the repo root, for the bundled example schema
PYTHONPATH=. openextract ./file.png \
  --schema examples.cli.schemas:DocumentInfo \
  --model xai:grok-4.3
```

Confirm the module imports cleanly: `python -c "import examples.cli.schemas"`.

## Batch partial failures

**Symptoms**

- CLI with `--continue-on-error`: exit code `7`, stderr warning, stdout still has
  the full JSON array
- Python `extract_many(..., return_exceptions=True)`: exceptions appear in-place

**Next step**

Inspect per-item `error` / `error_type` entries. Fix the failing inputs, or omit
`--continue-on-error` / `return_exceptions` to fail fast on the first error.

## Choosing a batch API

- `extract_many` / `extract_many_async` wait for the full batch and return
  results in **input order**.
- `iter_extract_many_async` yields `(input_index, result)` in **completion
  order**, consumes generators lazily, and never schedules more than
  `max_concurrency` items. Use it when you want to start processing before the
  last item finishes.
- `extract_many_with_results*` keep input order and add per-item usage,
  attempts, duration, and a sanitized source label.

Default fail-fast cancels outstanding work after the first error.
`return_exceptions=True` puts per-item exceptions in the result position and
continues. See [API reference](api-reference.md#choosing-a-batch-api).

## Sync batch inside an async event loop

**Symptoms**

- `RuntimeError: extract_many() cannot be called from a running event loop...`

**Next step**

Use the async API:

```python
results = await extract_many_async(schema=..., model=..., input_files=...)
```

## Related

- [Guide](guide.md)
- [For agents](agents.md)
- [CLI contracts](cli.md)
- [Provider matrix](providers.md)
- [README error handling](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/README.md#error-handling)
