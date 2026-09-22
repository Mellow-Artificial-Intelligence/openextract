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

## Releases

Bump `version` in `pyproject.toml` and move the `[Unreleased]` notes in
`CHANGELOG.md` into a dated section. After CI passes on `main`,
`.github/workflows/release.yml` publishes to PyPI and creates the GitHub
release, but only when that version is new.

## Dependency embargo

`[tool.uv] exclude-newer` in `pyproject.toml` refuses any package version
published within the last 24 hours, which protects against short-lived
compromised releases. `.github/workflows/embargo-bump.yml` opens a daily PR to
move that cutoff forward. Merge those PRs rather than editing the date by hand.
