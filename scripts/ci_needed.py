"""Decide which CI jobs a change set must run."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

ALWAYS_ALL = frozenset(
    {
        ".github/workflows/ci.yml",
        ".github/actions/detect-jobs/action.yml",
        ".github/actions/run-test-suite/action.yml",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "scripts/ci_needed.py",
    }
)
JOB_PREFIXES: dict[str, tuple[str, ...]] = {
    "lint": ("src/", "tests/", "examples/", "scripts/", "docs/api-reference.md"),
    "test": ("src/", "tests/", "examples/"),
    "package": ("src/", "README.md", "LICENSE"),
}
PATH_FILTERED_EVENTS = frozenset({"pull_request", "merge_group"})


def jobs_for_paths(changed: Sequence[str]) -> dict[str, bool]:
    """Return which CI jobs to run for ``changed`` repository paths."""
    if not changed:
        return dict.fromkeys(JOB_PREFIXES, False)
    if any(path in ALWAYS_ALL for path in changed):
        return dict.fromkeys(JOB_PREFIXES, True)
    return {job: _touched(changed, prefixes) for job, prefixes in JOB_PREFIXES.items()}


def _touched(changed: Iterable[str], prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix) for path in changed for prefix in prefixes)


def _write_outputs(path: Path, jobs: Mapping[str, bool]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for name, enabled in jobs.items():
            handle.write(f"{name}={'true' if enabled else 'false'}\n")


def _changed_files(base: str, head: str) -> list[str] | None:
    try:
        return subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", "--diff-filter=ACDMRT", base, head],
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError as exc:
        print(f"git diff failed: {exc}")
        return None


def _run_all_jobs(output: Path, reason: str) -> int:
    print(f"Running all jobs ({reason})")
    _write_outputs(output, dict.fromkeys(JOB_PREFIXES, True))
    return 0


def main() -> int:
    output = Path(os.environ["GITHUB_OUTPUT"])
    event = os.environ.get("EVENT_NAME", "")
    base = os.environ.get("BASE_SHA", "")
    head = os.environ.get("HEAD_SHA", "")

    if event not in PATH_FILTERED_EVENTS:
        return _run_all_jobs(output, f"event {event!r} is not path-filtered")
    if not base or set(base) == {"0"}:
        return _run_all_jobs(output, "no comparison base")

    changed = _changed_files(base, head)
    if changed is None:
        return _run_all_jobs(output, "git diff failed")

    print("Changed files:")
    for path in changed:
        print(f"  {path}")
    jobs = jobs_for_paths(changed)
    print("jobs: " + " ".join(f"{name}={value}" for name, value in jobs.items()))
    _write_outputs(output, jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
