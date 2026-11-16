# Contributing to openextract

```bash
git clone https://github.com/Mellow-Artificial-Intelligence/openextract.git
cd openextract
uv sync --dev

uv run pytest               # offline; 100% coverage is enforced
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Create a branch from `main`, keep the change focused, and open a pull request. PRs
need a code owner's approval and a green `CI` check.

## Design

openextract is one small primitive: schema + model + source → a validated instance
of that schema. Keep it that way.

- **Add control without adding concepts.** Pydantic fields and validators express
  meaning and constraints, `instructions` guide one call, `media_type` overrides
  detection, and `Model` holds provider settings. A new public option needs a
  concrete need that none of these can serve.
- **Leave orchestration to the caller.** Fetching policy, preprocessing, batching,
  concurrency, and per-item result handling are ordinary Python around `extract`.
- **Speak HTTP directly.** Model requests use HTTPX. Do not add provider SDKs or
  other runtime dependencies.
- **Share code only when behavior is the same.** Keep provider differences explicit
  in their own builders and readers.

## Layout

| Module | Responsibility |
| --- | --- |
| `_extract.py` | `extract` and `extract_async`: resolve, load, request, validate, retry |
| `models.py` | `Model`, the provider table, and `resolve` |
| `_content.py` | Text, paths, URLs, and bytes become content with a media type |
| `_providers.py` | General structured-output APIs |
| `_decisions.py` | Decisions APIs for closed-set schemas |
| `_openrouter_extractions.py` | OpenRouter's typed field API |
| `_schemas.py` | JSON Schema adapted to provider dialects |
| `_transport.py` | POST with bounded retries |
| `_diagnostics.py` | Every error message |

The three adapter modules share one interface: `request(schema, model, content,
instructions)` returns a path and a JSON body, and `parse(schema, model, payload)`
returns the validated result. To support a new API, add it to `API` and the provider
table in `models.py`, write its builder and reader, and register them.

## Tests

Tests check contracts through `extract` and `extract_async`, not the shape of private
helpers. The `provider` fixture in `tests/conftest.py` answers every HTTPX request
from a queue and records what was sent; `run` calls both entry points so one test
covers sync and async. Real HTTP is blocked, provider variables are cleared, and
retry delays are skipped.

For a new adapter, cover the request body, media encoding, unsupported inputs,
refusals, and truncated output, and update the provider table in the README.

## Live provider checks

`tests/test_live.py` holds 62 opt-in cases that call real providers and can incur
charges: seven frozen invoices (PDF, PNG, JPEG, multi-page, and scanned) through four
routes in sync and async, plus strict-type and tagged-union checks. Each compares
every field with retries disabled.

| Route | Credentials |
| --- | --- |
| OpenAI Responses | `OPENAI_API_KEY` |
| Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini through OpenRouter | `OPENROUTER_API_KEY` |

```bash
# Everything. uv loads .env; the library and pytest never do.
uv run --env-file .env pytest --live -m live --no-cov -v

# One provider and one fixture, in both modes.
uv run --env-file .env pytest --live -m live --no-cov -k "openrouter_gemini and degraded_scanned_pdf" -v
```

Without `--live` these cases are skipped even when credentials are present. With it,
a missing key fails the selected case and names the variable to set.

## Releases

Bump `version` in `pyproject.toml`, run `uv lock`, and move the `[Unreleased]` notes
in `CHANGELOG.md` into a dated section. After CI passes on `main`,
`.github/workflows/release.yml` publishes to PyPI and creates the GitHub release,
but only when that version is new.

## Dependencies

`[tool.uv] exclude-newer` in `pyproject.toml` refuses any package version published
after that date, which protects against short-lived compromised releases.
Dependencies are never upgraded automatically. To upgrade, move the date to at least
24 hours ago, run `uv lock --upgrade`, and open a pull request.

The `package` CI job builds the wheel and source distribution, installs each with
only the locked runtime dependencies, and runs `scripts/check_package.py`. That
script fails when the wheel, the installed package, or the installed runtime outgrows
its limit, or when the set of dependencies changes, then imports the package and
runs a first extraction with networking blocked. Raise a limit only in a pull request
that shows the measurements and the reason.
