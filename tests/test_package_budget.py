"""Regression checks for CI budget failures; no tooling ships in the library."""

import copy
import json
import runpy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CHECK = runpy.run_path(str(SCRIPTS / "check_package.py"))
BUDGET = json.loads((SCRIPTS / "package_budget.json").read_text())


def test_baseline_and_exact_limits_pass():
    report = copy.deepcopy(BUDGET["baseline"])
    assert CHECK["violations"](report, BUDGET) == []
    report.update(BUDGET["limits"])
    assert CHECK["violations"](report, BUDGET) == []


@pytest.mark.parametrize("metric", BUDGET["limits"])
def test_size_growth_fails(metric):
    report = copy.deepcopy(BUDGET["baseline"])
    report[metric] = BUDGET["limits"][metric] + 1
    assert CHECK["violations"](report, BUDGET) == [
        f"{metric}: {report[metric]:,} exceeds {BUDGET['limits'][metric]:,} bytes"
    ]


@pytest.mark.parametrize("kind", ["direct_dependencies", "transitive_dependencies"])
def test_new_dependency_fails(kind):
    report = copy.deepcopy(BUDGET["baseline"])
    report[kind].append("unexpected-sdk")
    assert any(kind in error for error in CHECK["violations"](report, BUDGET))


def test_installed_size_excludes_only_generated_bookkeeping(tmp_path):
    files = {"module.py": b"source", "module.pyc": b"bytecode", "RECORD": b"paths"}
    files.update(
        {"direct_url.json": b"local path", "METADATA": b"metadata", "native.so": b"native"}
    )
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)

    paths = [Path(name) for name in files]

    class Distribution:
        files = paths

        def locate_file(self, file):
            return tmp_path / file

    assert CHECK["installed_bytes"](Distribution()) == len(b"sourcemetadatanative")
