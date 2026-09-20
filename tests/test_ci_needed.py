"""Path-filter decisions for CI jobs."""

import runpy
import subprocess
from pathlib import Path

_MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "ci_needed.py"))
jobs_for_paths = _MODULE["jobs_for_paths"]
ci_needed_main = _MODULE["main"]

ALL_JOBS_OUTPUT = ["lint=true", "test=true", "package=true"]


def _run_main(monkeypatch, tmp_path, *, event, base="", head=""):
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("EVENT_NAME", event)
    monkeypatch.setenv("BASE_SHA", base)
    monkeypatch.setenv("HEAD_SHA", head)
    assert ci_needed_main() == 0
    return output.read_text(encoding="utf-8").splitlines()


def _git(repo, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.email=ci@example.com",
            "-c",
            "user.name=CI",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        text=True,
    ).strip()


def test_docs_only_skips_python_jobs():
    assert jobs_for_paths(["docs/cli.md", "docs/guide.md"]) == {
        "lint": False,
        "test": False,
        "package": False,
    }


def test_src_change_runs_all_jobs():
    assert jobs_for_paths(["src/openextract/_extract.py"]) == {
        "lint": True,
        "test": True,
        "package": True,
    }


def test_tests_skip_package_smoke():
    assert jobs_for_paths(["tests/test_cli.py"]) == {
        "lint": True,
        "test": True,
        "package": False,
    }


def test_api_reference_runs_lint_only():
    assert jobs_for_paths(["docs/api-reference.md"]) == {
        "lint": True,
        "test": False,
        "package": False,
    }


def test_lockfile_runs_all_jobs():
    assert jobs_for_paths(["uv.lock"]) == {
        "lint": True,
        "test": True,
        "package": True,
    }


def test_examples_run_lint_and_tests():
    assert jobs_for_paths(["examples/basic/local_file.py"]) == {
        "lint": True,
        "test": True,
        "package": False,
    }


def test_scripts_run_lint_only():
    assert jobs_for_paths(["scripts/bench.py"]) == {
        "lint": True,
        "test": False,
        "package": False,
    }


def test_empty_change_set_skips_all_jobs():
    assert jobs_for_paths([]) == {
        "lint": False,
        "test": False,
        "package": False,
    }


def test_ci_workflow_change_runs_all_jobs():
    assert jobs_for_paths([".github/workflows/ci.yml"]) == {
        "lint": True,
        "test": True,
        "package": True,
    }


def test_run_test_suite_action_change_runs_all_jobs():
    assert jobs_for_paths([".github/actions/run-test-suite/action.yml"]) == {
        "lint": True,
        "test": True,
        "package": True,
    }


def test_readme_runs_package_job():
    assert jobs_for_paths(["README.md"]) == {
        "lint": False,
        "test": False,
        "package": True,
    }


def test_license_runs_package_job():
    assert jobs_for_paths(["LICENSE"]) == {
        "lint": False,
        "test": False,
        "package": True,
    }


def test_push_event_runs_all_jobs(monkeypatch, tmp_path):
    """Pushes to main gate releases, so their runs must never be path-filtered."""
    lines = _run_main(monkeypatch, tmp_path, event="push", base="a" * 40, head="b" * 40)
    assert lines == ALL_JOBS_OUTPUT


def test_workflow_dispatch_runs_all_jobs(monkeypatch, tmp_path):
    assert _run_main(monkeypatch, tmp_path, event="workflow_dispatch") == ALL_JOBS_OUTPUT


def test_pull_request_without_base_runs_all_jobs(monkeypatch, tmp_path):
    assert _run_main(monkeypatch, tmp_path, event="pull_request", base="0" * 40) == ALL_JOBS_OUTPUT


def test_pull_request_rename_out_of_src_runs_package(monkeypatch, tmp_path):
    """A file moved out of src/ must still count as a package change (no rename coalescing)."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    _git(repo, "init")
    (repo / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add module")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "mv", "src/module.py", "module.py")
    _git(repo, "commit", "-m", "move module out of src")
    head = _git(repo, "rev-parse", "HEAD")

    monkeypatch.chdir(repo)
    lines = _run_main(monkeypatch, tmp_path, event="pull_request", base=base, head=head)
    assert "package=true" in lines
