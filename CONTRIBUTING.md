# Contributing to openextract

```bash
git clone https://github.com/Mellow-Artificial-Intelligence/openextract.git
cd openextract
uv sync --dev

uv run pytest --cov=openextract   # 100% coverage is enforced
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Create a branch from `main`, keep changes focused, and open a pull request. PRs
need a code owner's approval and a green `CI` check.

## Design rules

**Add useful control without adding concepts users must learn.** The
[canonical roadmap](https://github.com/Mellow-Artificial-Intelligence/openextract/wiki/Roadmap)
keeps OpenExtract a small extraction primitive: schema + model + source → a
validated instance of that schema. Prefer small explicit functions and share
preparation or validation when the behavior is actually the same.

Use existing hooks first: Pydantic fields and validators express output meaning
and constraints, `instructions` guide interpretation, and `media_type` overrides
input detection. See the [fine-detail examples](examples/fine_details.md).
Use `openextract.Model` for provider settings and connection options; see the
[configured example](examples/configured_models.md) and
[compatibility limits](examples/model_compatibility.md). Guidance and type
validation do not guarantee factual extraction accuracy.

### Follow the current call paths

In [`src/openextract/_extract.py`](src/openextract/_extract.py), simple and
customized calls use the same loading and routing functions. Adding instructions
or a media override does not change the result type or introduce a result wrapper.
`extract` and `extract_async` both return the requested schema instance.

- General extraction shares model resolution, `_prepare`, and `_validate`, with
  sync/async HTTP requests in `_transport.py`. Model strings beginning with
  `openai:` select Responses; `ollama:` selects native structured output.
  `Model(api=...)` explicitly overrides the default API.
- Loading shares media detection: explicit override, file/URL extension, URL
  content type, then byte sniffing. Inline text must be encoded to bytes; strings
  are paths or URLs. Async URL fetching uses an async HTTP client, but local file
  reads inside `extract_async` are synchronous. Callers can preload bytes or
  offload file reads when needed.
- Explicit decision-model strings select their dedicated request builders. The
  [OpenRouter adapter](src/openextract/_decisions.py) and
  [OpenAI adapter](src/openextract/_openai_decisions.py) each share their request
  preparation and response validation between sync and async calls. Both return
  schema instances, but support only `bool`, `Literal`, and `Enum` fields and
  text/image input. Their wire formats, credentials, and response checks differ;
  OpenAI also enforces its 1–400 field limit. Keep those differences explicit
  in their builders and parsers; share only HTTP transport and final validation.

### When a public option earns its place

A proposal should show a concrete user need that schemas, instructions, configured
models, media overrides, or ordinary Python composition cannot serve clearly.
Describe the smallest change, its provider/version limits, its sync/async behavior,
and the regression it could introduce. Reuse a helper only when it removes real
duplication without hiding provider differences; a speculative abstraction is not
a cleanup.

Fetching policy, preprocessing, reusable functions or partials, concurrency,
cancellation, and per-item result handling belong to the caller. A useful recipe
does not by itself justify a batch API, session object, plugin framework, or
orchestration dependency in the core.

### Check contracts, not implementation shape

[`tests/test_extract.py`](tests/test_extract.py) checks loading and media priority,
model-string routing, typed results, error translation, and both decision routes
using mock HTTP transports. The simple/customized sync/async
cases verify that instructions and overrides reach the model while preserving the
schema result; invalid-output cases verify validation in both entry points.
[`tests/test_quickstart.py`](tests/test_quickstart.py) and
[`tests/test_fine_details.py`](tests/test_fine_details.py) execute published examples
and check deterministic schema behavior. These offline checks measure wiring and
contracts, not provider extraction quality.

For a change, identify the affected path and reuse these checks. Add a behavioral
case only for a meaningful missing contract or concrete regression risk; do not
freeze private helper structure in tests. Tests support the user experience,
rather than becoming a separate evaluation product or a reason to expand the API.

## Provider adapters

Model requests use HTTPX; do not add provider SDK dependencies. Keep request
encoding and response parsing in `_providers.py`, connection defaults in
`models.py`, and retries/HTTP errors in `_transport.py`. Decisions schema mapping
remains in its dedicated modules. Validate final output using the original
Pydantic schema.

Use `httpx.MockTransport` to test complete requests and responses in both sync
and async flows. Cover media encoding, unsupported inputs, refusals, truncated
output, and provider-specific structured output. Update the README's provider
matrix whenever capabilities or authentication change. Default tests run offline;
real HTTPX transports are blocked unless a test is marked `live` and `--live` is
explicitly provided. Offline tests isolate provider credentials and retry delays;
live tests retain the caller's environment and normal transport behavior.

## Live provider checks

The frozen corpus in `tests/fixtures/invoices/` covers PDF, PNG, JPEG, multi-page
PDFs, and clean/degraded image-only scans. Expected invoices and checksums are
checked in with the fixtures. Generation and PDF-rendering tools are not needed
to run the tests.

The live suite has 62 cases. The invoice matrix has 56 cases: seven inputs, four
provider routes, and sync/path plus async/bytes extraction. Six schema regression
cases cover tagged unions and strict dates/decimals through OpenAI Responses,
OpenAI Chat JSON Schema, and Anthropic, in sync and async calls. Model IDs and
token limits are declared in `tests/test_live_providers.py` and
`tests/test_live_schemas.py`. Each case checks exact output with HTTP and
validation retries disabled, so failures remain visible.

| Route | Credentials |
| --- | --- |
| OpenAI Responses (`gpt-4.1-mini`) | `OPENAI_API_KEY` |
| Native Gemini (`gemini-2.5-flash`) | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Native Anthropic (`claude-haiku-4-5-20251001`) | `ANTHROPIC_API_KEY` |
| Gemini through OpenRouter (`google/gemini-2.5-flash`) | `OPENROUTER_API_KEY` |

Live API calls can incur provider charges. Load credentials from your environment
or an ignored local `.env` file, then opt in explicitly:

```bash
# All live checks; .env is loaded by uv, not by the library or pytest.
uv run --env-file .env pytest --live -m live --no-cov -v

# One provider and one fixture, in both sync and async modes.
uv run --env-file .env pytest --live -m live --no-cov -k "openrouter_gemini and degraded_scanned_pdf" -v

# Offline CI equivalent: never selects live tests, and needs no API keys.
uv run pytest -m "not live" --cov=openextract --cov-report=term-missing
```

Omit `--env-file .env` if credentials are already exported. Without `--live`,
live-marked tests are skipped even when credentials are present or `-m live` is
selected. With `--live`, a missing key fails the selected case with the required
environment-variable names. Use `-k` to select the providers you have configured.
Provider errors retain the library's `ModelError` details; output differences
fail the normal Pydantic-model equality assertion.

OpenRouter checks use JSON Schema output and require supporting endpoints. PDF
cases explicitly select native PDF processing through the `file-parser` plugin.
CI excludes the live marker and does not load `.env` or configure provider secrets.

## Releases

Bump `version` in `pyproject.toml` and move the `[Unreleased]` notes in
`CHANGELOG.md` into a dated section. After CI passes on `main`,
`.github/workflows/release.yml` publishes to PyPI and creates the GitHub
release, but only when that version is new.

## Dependency embargo

`[tool.uv] exclude-newer` in `pyproject.toml` refuses any package version
published after that date, which protects against short-lived compromised
releases. Dependencies are never upgraded automatically. To upgrade, move the
date to at least 24 hours ago, run `uv lock --upgrade`, and open a pull request.
