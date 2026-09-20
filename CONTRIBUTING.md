# Contributing to openextract

Thanks for your interest in contributing to openextract!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Mellow-Artificial-Intelligence/openextract.git
   cd openextract
   ```

2. Install dependencies with uv (includes all provider extras for tests):
   ```bash
   uv sync --dev
   ```

3. Run tests (add `-n auto` to fan them across cores, as CI does):
   ```bash
   uv run pytest
   ```

4. Run linting and type checking ([Astral](https://astral.sh) toolchain: [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/), [ty](https://docs.astral.sh/ty/)):
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run ty check
   ```

## Making Changes

1. Create a new branch from `main`
2. Make your changes
3. Ensure tests pass and code is formatted
4. Submit a pull request

## Pull Request Guidelines

- PRs require approval from a code owner before merging
- Keep changes focused and atomic
- Update tests for new functionality
- Follow existing code style (enforced by ruff)

## Code Style

This project uses the Astral toolchain for Python quality:

- [uv](https://docs.astral.sh/uv/) — dependency and environment management
- [ruff](https://docs.astral.sh/ruff/) — linting and formatting
- [ty](https://docs.astral.sh/ty/) — type checking (`src/openextract` only)

Run before submitting:

```bash
uv run ruff check . --fix
uv run ruff format .
uv run ty check
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs lint, tests, and package smoke
tests. All three start in parallel the moment a run is created; each one calls
the `.github/actions/detect-jobs` composite action first and skips its own work
when the change set cannot affect it (for example a docs-only PR does not
install Python dependencies). Nothing waits on a gating job, and the test suite
runs under `pytest -n auto` across every runner core. The single required check
is the `CI` job, which fails unless lint, test, and package all succeeded.
If a test matrix cell fails, times out, or is killed mid-pytest by a
hosted-runner eviction (`The runner has received a shutdown signal`), the same
suite — including the 100% coverage gate on 3.12 — is run once more on a fresh
runner before `CI` fails. Test jobs time out at 5 minutes so a hang rematches
instead of sitting until a long budget expires. Outdated pull-request runs are
cancelled when a new commit is pushed. After CI
succeeds on `main`,
[`.github/workflows/release.yml`](.github/workflows/release.yml) publishes to
PyPI only when the version in `pyproject.toml` is new.

## Dependency embargo (24h)

We do **not** install any package version that has been published less than 24 hours ago. This protects against supply-chain attacks where a compromised release sits on PyPI for a few hours before being yanked.

The policy is enforced via uv's `exclude-newer` setting in `pyproject.toml`:

```toml
[tool.uv]
exclude-newer = "<YYYY-MM-DDTHH:MM:SSZ>"
```

A scheduled GitHub Actions workflow (`.github/workflows/embargo-bump.yml`) runs daily at 03:00 UTC and opens a PR that advances the cutoff to "yesterday 00:00 UTC". Merge those PRs as part of normal review. The workflow uses only first-party tooling (the `gh` CLI preinstalled on GitHub-hosted runners and `astral-sh/setup-uv`) — no third-party PR-creation actions, to keep the supply-chain surface for the embargo workflow itself minimal.

If you need to add a dependency that was published in the last 24 hours, hold off and wait the embargo out rather than bumping `exclude-newer` past the rolling window.

## Performance benchmarking (maintainers)

`scripts/bench.py` microbenchmarks openextract's local hot path — import cost, media loading, agent construction, and mocked extraction. Run it on the same machine before and after performance-sensitive changes:

```bash
uv run python scripts/bench.py
```

It deliberately does **not** measure model or network latency. See [docs/benchmarking.md](docs/benchmarking.md) for what it measures, when to run it, and how to read the output without overfitting to local noise.

## ExtractBench quality eval (any model)

To score openextract on LlamaIndex ExtractBench with any provider model:

```bash
uv run python scripts/extractbench.py --model openai:gpt-5 --test
```

The first run bootstraps ExtractBench into `.extractbench/`. This is opt-in,
makes live model calls, and is not part of default CI. See
[docs/extractbench.md](docs/extractbench.md) for commands and the latest local
smoke status.

## Live provider smoke tests (maintainers)

Default `pytest` does not call live models. To run the opt-in harness:

```bash
OPENEXTRACT_LIVE_SMOKE=1 uv run pytest -m integration tests/test_live_smoke.py -v
```

See [docs/live-smoke.md](docs/live-smoke.md).

## Documentation site

Markdown under `docs/` is built with Jekyll and deployed by
[`.github/workflows/deploy-docs.yml`](.github/workflows/deploy-docs.yml).
User/agent how-tos are `docs/guide.md` and `docs/agents.md`. `docs/llms.txt`
is copied through as a static file for coding agents. The landing page is
`docs/index.html`. Do not add `docs/.nojekyll`; the workflow uploads `_site`.

## Release checklist (maintainers)

Use this before cutting a release (a version bump on `main` publishes only after
CI is green — [`.github/workflows/release.yml`](.github/workflows/release.yml)):

1. **Version bump** — set `version` in `pyproject.toml` to the release version.
2. **Changelog** — move `[Unreleased]` notes into a dated `## [X.Y.Z]` section in
   `CHANGELOG.md`; call out breaking changes explicitly.
3. **Docs and examples review** — skim README, `docs/`, and `examples/README.md`
   for version-sensitive install/CLI notes.
4. **Local checks**
   ```bash
   uv sync --dev
   uv run ruff check .
   uv run ruff format --check .
   uv run ty check
   uv run pytest -n auto --cov=openextract --cov-report=term-missing --cov-fail-under=100
   ```
5. **Dependency embargo** — confirm `[tool.uv] exclude-newer` is within the
   rolling 24h window (merge recent `embargo-bump` PRs from
   `.github/workflows/embargo-bump.yml` rather than hand-editing past the window).
6. **Merge to `main`** — CI (`.github/workflows/ci.yml`) must be green.
7. **GitHub Actions release job** — after CI succeeds on `main`, `release.yml`
   builds, publishes to PyPI (Trusted Publishing), and creates the `vX.Y.Z`
   GitHub Release when that version is new. Docs-only commits skip the release
   toolchain once the existing version is detected.
8. **PyPI verification** — `pip index versions openextract` / the PyPI project
   page shows the new version; a clean venv install imports.
9. **GitHub Release verification** — tag `vX.Y.Z` exists with release notes.
10. **Post-release cleanup** — update docs/issues that referenced the prior
    version; close or retarget completed milestone issues.
11. **Security-sensitive notes** — if the release changes URL fetching, SSRF
    controls, or supply-chain policy, call that out in `CHANGELOG.md` and
    `SECURITY.md` as appropriate.

## Questions?

Open an issue on GitHub.
