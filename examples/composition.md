# Compose extraction with ordinary Python

These runnable scripts keep preprocessing, reuse, concurrency, and error handling
in your application. Each successful extraction returns an ordinary `Invoice`;
there is no batch or session API to learn.

## Setup

From a repository checkout with Python 3.12+, install the locked development
environment and set your own Anthropic key:

```bash
uv sync --dev --frozen
export ANTHROPIC_API_KEY='your-api-key'
```

On PowerShell use `$env:ANTHROPIC_API_KEY = 'your-api-key'`. The scripts target
`claude-sonnet-4-5`; running them makes paid provider requests. They use
`AnthropicModelSettings(max_tokens=1024)` with the verified Pydantic AI 2.56.0 /
Anthropic SDK 1.13.0 pair. See [configured models](configured_models.md) for setup
outside a checkout, settings limits, and endpoint routing, and the
[compatibility record](model_compatibility.md) for allowed combinations that fail.
No dependency remedy is applied here. This `Invoice` is for general extraction,
not the narrower decision endpoints.

## Preprocess text and bind reusable policy

Run the complete [composition.py](composition.py) example:

```bash
uv run --no-sync python examples/composition.py
```

`prepare_text` normalizes CRLF line endings, then calls `text.encode("utf-8")`.
Names, amounts, and ambiguous currency remain available to the model. Adapt this
function to preprocessing you own; OpenExtract does not choose OCR, redaction,
or document conversion for you. A source string always means a path or URL, so
pass encoded bytes for inline text.

`extract_text` uses `functools.partial` to bind `Invoice`, a configured model, and
interpretation instructions. It then supplies the prepared bytes and known media
type. Reuse the function with another text and the same model within the same
application lifecycle. It returns an `Invoice`, not a custom result envelope.
`max_tokens` limits generated output; too small a budget can truncate a response.
The fictional source's intended fields are Acme Paper Co., `Decimal("120.00")`,
and USD. Those are source-derived expectations, not measured provider output.

## Bound async work and handle each file

Run the complete [async_composition.py](async_composition.py) example, which
imports the schema and preprocessing from its neighboring script:

```bash
uv run --no-sync python examples/async_composition.py
uv run --no-sync python examples/async_composition.py examples/fixtures/invoice.txt missing.txt
```

The default finds the supplied fixture relative to the script, even outside the
repository directory. The second command deliberately includes a missing file
to demonstrate application-owned failure handling. User-supplied paths are
relative to the current working directory. Inputs must be UTF-8 text files.

`extract_files` applies a semaphore around both reading/preprocessing and the
provider call. Its default permits at most two files in flight. Local reads in
OpenExtract's `extract_async` are currently synchronous: this recipe instead uses
`asyncio.to_thread(read_source, path)` and passes the resulting bytes to
`extract_async`. For small files you can also preload bytes before starting async
work. CPU-heavy preprocessing may need a process pool chosen by your application.

Results stay in input order even when requests finish out of order. Successful
entries are `Invoice` instances. `OSError`, `UnicodeError`, and OpenExtract's
`ExtractionError` subclasses become per-item exception values; `main` shows where
to log, retry, or persist them. It does not catch every exception: programming
errors fail the TaskGroup and cancel sibling tasks. Cancellation is not converted
to a per-item error; cancelling the caller propagates and cancels outstanding
async work. A thread already reading a file cannot be forcibly stopped by task
cancellation and may finish its read in the background.

This example is for a finite list: it creates one task per path, bounds active
work rather than queued tasks, and retains all results in memory. For very large
inputs, use your own bounded queue or chunks. Concurrency is not a rate limiter;
choose limits and retry policy for your provider. Remote fetching, timeouts,
logging, cancellation policy, and storage remain caller responsibilities. The
example closes its configured model's client in the same event loop that used it;
do not share an async network client across independently created loops.

## Offline verification

```bash
uv run --no-sync pytest tests/test_composition.py
```

Tests check preprocessing, real extraction wiring with a local model, configured
model construction, result order, bounded concurrency, expected per-file
failures, unexpected-error cleanup, and cancellation. They also execute both
script entry points with provider calls replaced locally. No real key or paid
request is needed. These checks establish wiring and application behavior, not
extraction quality. No live quality check was run; verify factual output against
your own documents.
