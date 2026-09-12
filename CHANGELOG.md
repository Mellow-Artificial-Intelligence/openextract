# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Breaking changes are called out explicitly in the relevant release section.
Deprecations should be listed with the replacement path and expected removal
timing when that is known.

## [Unreleased]

### Added
- `Citation.as_dict()` returns a JSON-stable `{field, quote, page, bbox}`
  payload (`bbox` is a list of four floats or `null`). Boxes stay
  parser-backed; `as_field_citation()` ExtractBench mapping is unchanged.
- CLI `--cite` requests per-field citations via the same `cite=True` library
  path (parse-then-extract when the PDF extra is available). JSON and JSONL
  include a `citations` array of `{field, quote, page}` objects with optional
  `bbox`. An injected agent still raises `ValueError`, as in the library.
- Example and guide notes for `cite=True`:
  `examples/advanced/extract_with_citations.py` shows `extract_with_usage` /
  `extract_many_with_results`. Citation docs spell out parser-backed boxes
  (never invented) and the ExtractBench `FieldCitation` mapping.
- `SwarmResult.citations` when `cite=True`: reduced per-field spans that
  support `output`. `first` keeps the leading agent's citations; `merge` and
  `vote` keep the first citation whose agent's field value equals `output`.
  Per-agent citations stay on each `ExtractionResult`. `extract_swarm*` still
  returns the schema instance.

### Changed
- ExtractBench docs: latest smoke notes that `0.13.1+` includes page/word
  grounding fixes (#220); the published 6-doc numbers remain the pre-fix
  `0.13.0` run (page F1 0.24). A fresh smoke has not been published; the
  gate is still unmet until that run.
- Docs: citation guidance in the guide and agent contract now covers page
  remap, value-backfill, and parser-backed boxes (`openextract[pdf]`).

### Fixed
- Release publish accepts hatchling wheels that emit Metadata-Version 2.5
  (`pypa/gh-action-pypi-publish` v1.14.2 / Twine 7). 0.13.1 build succeeded
  but PyPI upload failed on the older action pin.

## [0.13.1] - 2026-09-11

### Fixed
- ExtractBench page/word grounding: one-page windows remap model `page=1` /
  missing pages onto the document page the prompt actually contained, and
  citation grounding locates evidence from the extracted field value when the
  quote is paraphrased or the hinted page is wrong. Parser boxes now match
  numeric/punctuation variants (`$1,234.00` vs `1234`) and prefer the value
  span. Model field paths that drop array indexes (`qty` → `lines[0].qty`) are
  rebound onto the extracted output. Boxes are still never invented.

### Removed
- ExtractBench per-SHA smoke lab log (`docs/extractbench-smokes.md`). Latest
  6-doc smoke status now lives in `docs/extractbench.md`.

### Security
- Bump locked `pyasn1` from 0.6.3 to 0.6.4 to address CVE-2026-59884
  (unbounded BER tag parsing / DoS). (#217)

## [0.13.0] - 2026-09-04

### Changed
- ExtractBench per-file timeout is ``max(1800, window-pool wait)`` for a
  long-doc window count, and timeout retries are
  disabled. Veralto (41) and long (66) are not killed at 1800s × 3.
- ExtractBench window-pool wait is
  ``timeout × (ceil(windows / concurrency) + 2)``, not
  ``timeout × (batches + 1)``. Pueblo's 11-window run is not killed at
  960s. A pool timeout merges finished windows (and in-flight ones that
  complete quickly) instead of discarding them, then returns immediately
  instead of waiting on cancelled workers. Hung windows are still not
  retried; empty-tool-call / output-retry failures are retried at the
  window (pydantic-ai ``output_retries=3``, plus one extra attempt) so
  Veralto is not killed by a 1-retry cap.
- Empty-text / scanned PDFs (Bianco) are rendered to compact grayscale page
  images (long edge ≤ 768px) locally and never uploaded for OpenRouter
  document-parse (400 rate-limit). Boxes stay parser-backed; none are
  invented when the scan has no word spans.
- OpenRouter usage falls back to provider ``prompt_tokens`` /
  ``completion_tokens`` when pydantic-ai ``RequestUsage.extract`` leaves
  zeros (live GLM-5.3-flash / W14). Counts are never invented.
- Parse-then-extract windows page-indexed PDF text under a 12k-character
  budget (was 80k) and one page per window, and splits a single oversized page so
  each model call fits GLM-class context. Slide decks that fit 80k as one
  prompt (Veralto) now split. Citations are collected from every window before
  reduce, then missing pages are backfilled from parse values (including
  ``1,234`` / ``1234.0`` forms). Bounding boxes remain parser-backed only.
- PDF parse takes a process-wide lock. pypdfium2 is not thread-safe; concurrent
  ExtractBench workers no longer crash native PDFium.
- Usage capture asks OpenRouter for native usage via both `openrouter_usage`
  and `extra_body.usage`, then reads `prompt_tokens` / `native_tokens_*` from
  `result.usage`, nested `details`, `response`, and `all_messages`. Default-zero
  pydantic-ai `input_tokens` is still skipped. Counts are never invented.

## [0.12.0] - 2026-08-31

### Added
- Opt-in per-field provenance: `cite=True` on extract APIs, sessions, batch, and
  swarms asks the model for source spans without changing `extract()`'s return
  type. New public `Citation` (`field`, `quote`, `page`, optional normalized
  `bbox`) attaches to `ExtractionResult.citations`. `Citation.as_field_citation()`
  maps onto ExtractBench `FieldCitation`. Boxes are kept only when the model
  supplies a page-normalized COCO span; they are never invented. Default
  `cite=False` leaves prompts and return types unchanged.
- `scripts/extractbench.py` now emits real `field_citations` (citations on by
  default; `--no-cite` to disable) so page-level grounding can score when the
  model returns a page, and word-level grounding can score when it also returns
  a normalized box.

### Changed
- CI wall-clock time roughly halved. Change detection moved out of a gating job
  into the `.github/actions/detect-jobs` composite action that each job runs for
  itself, so lint, test, and package now start immediately instead of waiting on
  a separate runner. The test suite runs under `pytest -n auto` (new
  `pytest-xdist` dev dependency), and the coverage threshold is enforced by the
  same pytest process rather than a follow-up `coverage report` step.
- The docs workflow no longer serializes pull-request builds behind a global
  `pages` concurrency group; only the deploy job queues on `pages`, and
  superseded pull-request doc builds are cancelled.

## [0.11.0] - 2026-08-20

### Added
- CLI swarm and agent flags: `--swarm N`, `--models a,b`, `--agent SPEC`,
  `--agents SPEC,SPEC`, and `--reduce merge|vote|first`. `--schema` is now
  optional when an agent declares an `output_schema`, and `--usage` on a swarm
  reports the agent count and reduce strategy. Remote agent failures exit `8`.
- Agents are accepted by `extract`, `extract_async`, `extract_with_usage`, and
  `extract_with_usage_async` in the `model` position, and an agent declaring an
  `output_schema` can be passed as `schema` — `extract(agent, input_file)`.
  A single-model agent runs as a one-shot call; an agent with subagents or a
  remote endpoint fans out into a swarm and its outputs are reduced.
- Importable extract agents: `define_agent` / `define_remote_agent` package a
  model, style, instructions, and `output_schema` behind a description, and
  `subagents` compose them. `load_agent` / `load_agents` /
  `load_agent_directory` load them from a directory (`agent.py`,
  `subagents/`, `instructions.md`), a Python file, or `module:attribute`.
  Agents are accepted anywhere a swarm takes `agents`.
- Remote agents over HTTP with `RemoteAgentError` (retried on transient
  statuses and transport failures) and per-request auth providers in
  `openextract.auth`: `bearer`, `basic`, and `vercel_oidc`.
- Swarms: `extract_swarm` / `extract_swarm_async` run several agents over one
  input and return the reduced result, and
  `extract_swarm_with_results*` additionally report each agent's
  `ExtractionResult`, the summed usage, and the reduce strategy. Agents are a
  model identifier, a configured pydantic-ai `Model`, or a `SwarmMember` with
  per-agent `instructions` / `style`; `size` fans one agent out up to 16 ways.
  The input is loaded once for the whole swarm.
- Swarm reduce strategies: `SwarmReduce` (`merge`, `vote`, `first`),
  `normalize_reduce`, and `reduce_outputs` fold several same-schema outputs
  into one validated instance.
- CLI batch ergonomics for large workflows: `--max-concurrency` (validated
  before any model call), `--output jsonl` for incremental completion-order
  records with an `index` field, `--progress` reporting on stderr only,
  `--manifest` for JSONL per-input `source`/`media_type`/`name` configuration,
  and `--usage` on batches with per-item plus aggregate token usage via the
  rich result API. The default JSON array output and exit codes `0`-`7` are
  unchanged; Ctrl-C now exits `130` and a closed stdout pipe exits `141`, with
  cancellation, ordering, and partial-failure contracts documented in
  `docs/cli.md`.
- `scripts/extractbench.py` runs [ExtractBench](https://github.com/run-llama/ExtractBench)
  through openextract with any `pydantic-ai` model identifier (`--model openai:gpt-5 --test`).
- Extraction styles: `style='direct'` (default) still sends media to the model
  in one shot; `style='search'` uses Pydantic AI Harness `FileSystem` tools
  (read, regex search, glob) on text documents; `style='code'` uses Harness
  `CodeMode` so the model can write Python against a workspace copy of the
  text. Available on every extract API, reusable sessions, and as `--style`
  on the CLI. Install `pydantic-ai-harness` for search and
  `pydantic-ai-harness[codemode]` for code execution.
- Typed input and result contracts: `ExtractionInput` supports direct
  `os.PathLike` sources with per-item `media_type` and an optional safe
  `name`, and `ExtractionResult[T]` carries output, usage, attempts, timing,
  model/media metadata, and warnings without retaining raw media or secrets.
- `Path`/`os.PathLike` now works directly in every public API, and batch calls
  accept heterogeneous inputs with per-item media types in a single run.
- `extract_many_with_results` / `extract_many_with_results_async` return
  per-item `ExtractionResult` diagnostics, and `total_usage` aggregates token
  usage across batch results.
- `Literal[True/False]` overloads on `extract_many*` and
  `iter_extract_many_async` so type checkers infer `list[T]` versus
  `list[T | Exception]` from `return_exceptions`.
- Reusable `Extractor` and `AsyncExtractor` sessions with deterministic agent
  and HTTP-client cleanup, configured Pydantic AI model or agent injection,
  model settings/timeouts/instrumentation, and typed `RetryPolicy` support.
- `InputTooLargeError`, a 50 MiB default per-input cap, the
  `OPENEXTRACT_MAX_INPUT_BYTES` environment variable, `max_input_bytes` on all
  extraction APIs, and `--max-input-bytes` on the CLI.
- Drift-checked canonical API reference, supported-Python CI matrix, and wheel
  and source-distribution install smoke tests.
- `iter_extract_many_async()` streams `(input_index, result)` pairs in
  completion order without waiting for the full batch.
- Streaming-batch example and docs comparing `iter_extract_many_async`
  (completion order) with `extract_many` (input order).
- GitHub Pages now builds a documentation site: user [guide](docs/guide.md),
  [agent contract](docs/agents.md), rendered API/CLI/provider pages, and
  [`llms.txt`](docs/llms.txt).

### Changed
- Split the internal extraction implementation into focused modules (`_types`,
  `_config`, `_media`, `_errors`, `_retry`, `_agent`, `_session`, `_batch`) so
  input loading, retries, sessions, and batch execution are no longer one file.
  Public APIs are unchanged.
- CI skips jobs that the change set cannot affect, cancels outdated pull-request
  runs, caches uv downloads, and collects coverage on Python 3.12 only. Releases
  publish only after CI succeeds on `main`, and docs deploy only when `docs/`
  changes.
- `openai:` model identifiers now use the OpenAI Responses API by default;
  `openai-chat:` remains available as an explicit Chat Completions opt-in.
- Batch execution now schedules at most `max_concurrency` inputs, consumes
  generator inputs lazily, and cancels and awaits outstanding work on fail-fast
  errors while preserving input order in the existing list APIs.
- Python API calls no longer load `.env` into process-wide state. The CLI and
  bundled examples continue to load `.env` explicitly.
- Examples run as `python -m examples.<module>` without mutating `sys.path`.
- Package import defers Pydantic AI runtime modules, and model-error
  classification inspects the raised exception's MRO without importing
  unrelated provider SDKs.

### Security
- Paths, URLs, bytes, file-like objects, stdin, and batch inputs now fail before
  a model call when they exceed the configured cap. URL bodies are streamed and
  bounded even when `Content-Length` is missing or incorrect.

## [0.10.0] - 2026-08-01

### Added
- Explicit `RuntimeError` when `extract_many()` is called from a running event
  loop, directing callers to `extract_many_async`.
- Maintainer docs: CLI stdout/stderr/exit-code contracts, provider capability
  matrix, troubleshooting guide, live smoke harness notes, release checklist,
  and an input-size-limits design proposal.
- Opt-in live provider smoke test (`OPENEXTRACT_LIVE_SMOKE=1`) for a
  representative OpenAI image path.
- Expanded `SECURITY.md` URL input security model (schemes, host validation,
  redirects, env configuration, and non-guarantees).

### Changed
- Async extraction now keeps disk, DNS, and file-like reads off the event loop
  and reuses one HTTP client across each batch.
- Model retries now reuse the original media payload, prompt, and agent instead
  of reading or fetching the input and rebuilding the agent on every attempt.
- Model retries now distinguish transient failures from permanent provider
  errors, honor bounded `Retry-After` values, and expose provider, status,
  retryability, and retry-after metadata on `ModelError`.
- Added `retry_max_backoff` to all extraction APIs and
  `--retry-max-backoff` to the CLI.

### Security
- Async URL fetching retains redirect-by-redirect public-host and SSRF
  validation while using async clients and offloaded DNS resolution.

## [0.9.0] - 2026-07-12

### Added
- Compatibility and deprecation policy documenting the public API surface,
  pre-1.0 breaking-change expectations, provider compatibility limits, and
  Python version support policy.
- `ProviderNotInstalledError` (a subclass of `ExtractionError`) raised with an
  actionable `pip install openextract[...]` hint when a model is requested whose
  provider extra is not installed. The CLI reports it with exit code `6`.
- `--continue-on-error` flag on the `openextract` CLI: in batch mode, keep
  processing remaining inputs when one fails, emit per-item errors inline, and
  exit `7` if any input failed (default remains abort-on-first-failure).
- README public API stability audit covering every symbol exported from
  `openextract.__all__`, plus CLI stability notes and pre-1.0 follow-ups.
- Release-readiness documentation for provider install errors, extra-specific
  install hints, and CLI partial-batch failure stdout/stderr and exit-code
  behavior.

### Changed
- `max_retries`, `retry_backoff`, and `max_concurrency` now fail early with
  deterministic `ValueError` messages when invalid values are provided.

## [0.8.0] - 2026-06-02

### Added
- Environment variables `OPENEXTRACT_URL_TIMEOUT` and `OPENEXTRACT_MAX_REDIRECTS`
  to configure HTTP timeout and redirect limits when fetching URLs.
- Ship `py.typed` for PEP 561 type checker support.
- Run [ty](https://docs.astral.sh/ty/) on `src/openextract` in CI (Astral toolchain with uv and ruff).
- `max_retries` and `retry_backoff` on `extract_async`, `extract_with_usage`,
  `extract_with_usage_async`, `extract_many`, and `extract_many_async` (per-item
  for batch), matching sync `extract()` retry semantics.
- Optional dependency extras: `openai`, `anthropic`, `google`, `bedrock`,
  `cohere`, `groq`, `huggingface`, `mistral`, `openrouter`, `xai`, `logfire`,
  and `all`.
- Examples `batch_invoices.py` and `extract_with_usage.py` for concurrent batch
  extraction and token usage reporting.
- CLI accepts multiple `input_file` arguments for batch extraction via `extract_many`.
- `--usage`, `--media-type`, and stdin (`-`) support on the `openextract` command.

### Changed
- **Breaking:** `pip install openextract` no longer bundles every provider SDK.
  Install a provider extra for model calls (e.g. `openextract[openai]` or `openextract[all]`).
- Expand README API reference to document `extract_async`, `extract_many`,
  `extract_with_usage`, and related public APIs.

## [0.7.0] - 2026-05-22
### Added
- Add `extract_with_usage_async`: async counterpart to `extract_with_usage` that
  returns `(output, Usage)` alongside the token counts for the model call.
- Add `media_type` keyword argument to `extract_many` and `extract_many_async`:
  the batch functions previously hard-coded `None`, making it impossible to process
  `bytes` or file-like inputs that require an explicit MIME type; the new parameter
  is applied uniformly to every item in the batch.
- Add `--max-retries` and `--retry-backoff` flags to the `openextract` CLI:
  exposes the existing retry logic (already available via the Python API) to
  command-line users.

### Security
- URL fetching now refuses hosts that resolve to non-public addresses
  (private/loopback/link-local/multicast/reserved, IPv4 and IPv6, including
  IPv4-mapped IPv6 and the `169.254.169.254` cloud-metadata endpoint) to
  reduce SSRF risk when callers pass untrusted URLs. The check is re-applied
  at every redirect hop; set `OPENEXTRACT_ALLOW_PRIVATE_URLS=1` to opt out.
  **Breaking** for callers that previously fetched `localhost`/internal
  hosts via `extract()`.
- Pinned all GitHub Actions to commit SHAs.

## [0.6.0] - 2026-05-16
- Add Anthropic provider support (extra `pydantic-ai-slim[anthropic]`; `anthropic.APIError` wrapped as `ModelError`).
- Add AWS Bedrock provider support (extra `pydantic-ai-slim[bedrock]`; `botocore.exceptions.ClientError` wrapped as `ModelError`).
- Add xAI (Grok) provider support (extra `pydantic-ai-slim[xai]`).
- Add Cohere provider support (extra `pydantic-ai-slim[cohere]`; `cohere.core.api_error.ApiError` wrapped as `ModelError`).
- Add Hugging Face provider support (extra `pydantic-ai-slim[huggingface]`; `huggingface_hub.errors.HfHubHTTPError` wrapped as `ModelError`).
- Add Groq provider support (extra `pydantic-ai-slim[groq]`; `groq.APIError` wrapped as `ModelError`).
- Add Mistral provider support (extra `pydantic-ai-slim[mistral]`; `mistralai.client.errors.mistralerror.MistralError` wrapped as `ModelError`).
- Add OpenRouter provider support (extra `pydantic-ai-slim[openrouter]`; openai-compatible, so the existing `openai.APIError` classification applies).
- Document Cerebras provider support (`cerebras:` prefix; openai-compatible — no dedicated extra needed).
- Document Outlines provider support (`outlines:` prefix; install separately with a backend extra such as `pydantic-ai-slim[outlines-transformers]`).
- Clarify Ollama support: it works via the `openai`-compatible code path; `pydantic-ai-slim` does not publish a dedicated `ollama` extra, so no separate dependency is required.

## [0.5.0] - 2026-05-16
- Add `extract_async` for async extraction using `Agent.run`.
- Add `extract_many` and `extract_many_async` for concurrent batch extraction with configurable concurrency and optional exception capture.
- Accept raw `bytes` or any binary file-like object as `extract`'s `input_file`; new keyword-only `media_type` parameter for explicit MIME typing (required for `bytes`/file-like inputs, optional override for `str`).
- Add optional `max_retries` and `retry_backoff` keyword arguments to `extract()` for retrying transient `ModelError` failures with exponential backoff and jitter. Default behavior is unchanged (no retries).
- Add `extract_with_usage` and a `Usage` dataclass that surface model token counts (input, output, total) alongside the extracted output.
- Add `openextract` command-line interface (`openextract <file> --schema module:Class --model openai:gpt-5`) with structured exit codes.
- Add `examples/` directory with runnable scripts for invoice, receipt, and meeting-notes extraction.
- Reorganize `examples/` into use-case folders (basic, images, documents, batch, async, advanced, audio, CLI), add bundled fixtures, `run_all.py`, and smoke tests.
- Replace the substring-based exception classifier in `extract()` with typed provider-error matching against `pydantic_ai.exceptions.ModelAPIError`, `openai.APIError`, and `google.genai.errors.APIError`. Behavior change: an arbitrary exception whose message merely mentions "model" is no longer promoted to `ModelError`; it is now wrapped as `ExtractionError` unless the exception type is a subclass of a known provider error.

## [0.4.0] - 2026-05-16
- Accept `http://` URLs in addition to `https://`; previously, plain-HTTP URLs were silently treated as local file paths.
- Raise `UrlFetchError` on non-2xx responses; previously, the HTML error body was passed to the LLM as media bytes.
- Follow HTTP redirects when fetching URLs and apply a 30-second timeout.
- Fall back to the response `Content-Type` header when the URL has no recognizable extension (e.g., `/download?id=42`).
- Reach 100% test coverage and enforce the threshold in CI.
- Remove `configure_logging` from `__all__` — it was never defined, breaking `from openextract import *`.
- Fix `extract()` docstring (`url` → `input_file`) and add type hints to `_get_media`.

## [0.3.2] - 2026-05-05
- Add Ollama model support.

## [0.2.0] - 2026-01-11
- Landing page redesign and security updates.

## [0.1.4] - 2025-12-21
- Restructure project as installable Python package.
- Add tests and error handling.
- Initial commit: media extraction utility with pydantic-ai.

## [0.1.2] - 2025-09-13
- Add bytes-only vision API.
- Render PDFs to images.
- Support multimodal messaging.

## [0.1.1] - 2025-09-10
- Merge pull request #12 from Mellow-Artificial-Intelligence/new-release.

[Unreleased]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.13.1...HEAD
[0.13.1]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.3.1...v0.3.2
[0.2.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.2...v0.1.4
[0.1.2]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Mellow-Artificial-Intelligence/openextract/releases/tag/v0.1.1
