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
Configured Pydantic AI models are accepted, but settings support depends on the
dependency version and provider; see the [verified example](examples/configured_models.md)
and [compatibility findings](examples/model_compatibility.md). Guidance and type validation do not guarantee
factual extraction accuracy.

### Follow the current call paths

In [`src/openextract/_extract.py`](src/openextract/_extract.py), simple and
customized calls use the same loading and routing functions. Adding instructions
or a media override does not change the result type or introduce a result wrapper.
`extract` and `extract_async` both return the requested schema instance.

- General extraction shares `_build_agent` and `_model_errors`, then uses the
  Agent's sync or async run method. Model strings beginning with `openai:` select
  Responses; `ollama:` selects native structured output. A configured Model is
  passed through rather than rewritten by the string-routing rules.
- Loading shares media detection: explicit override, file/URL extension, URL
  content type, then byte sniffing. Inline text must be encoded to bytes; strings
  are paths or URLs. Async URL fetching uses an async HTTP client, but local file
  reads inside `extract_async` are synchronous. Callers can preload bytes or
  offload file reads when needed.
- Explicit decision-model strings branch before Agent construction. The
  [OpenRouter adapter](src/openextract/_decisions.py) and
  [OpenAI adapter](src/openextract/_openai_decisions.py) each share their request
  preparation and response validation between sync and async calls. Both return
  schema instances, but support only `bool`, `Literal`, and `Enum` fields and
  text/image input. Their wire formats, credentials, and response checks differ;
  OpenAI also enforces its 1–400 field limit. Keep those differences explicit
  instead of forcing both providers through a registry or universal adapter.

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
using local models or mock HTTP transports. The simple/customized sync/async
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
