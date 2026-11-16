"""The CI footprint check fails on size growth and on any change of dependencies."""

import runpy
from pathlib import Path

import pytest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_package.py"))
LIMITS, DEPENDENCIES, violations = CHECK["LIMITS"], CHECK["DEPENDENCIES"], CHECK["violations"]
AT_LIMIT = {**LIMITS, **DEPENDENCIES}


def test_sizes_at_the_limit_pass():
    assert violations(AT_LIMIT) == []


@pytest.mark.parametrize("metric", LIMITS)
def test_size_growth_fails(metric):
    report = AT_LIMIT | {metric: LIMITS[metric] + 1}
    message = f"{metric}: {report[metric]:,} exceeds {LIMITS[metric]:,} bytes"
    assert violations(report) == [message]


@pytest.mark.parametrize("kind", DEPENDENCIES)
def test_new_dependency_fails(kind):
    report = AT_LIMIT | {kind: [*DEPENDENCIES[kind], "unexpected-sdk"]}
    assert [error.split(":")[0] for error in violations(report)] == [kind]


def test_installed_size_excludes_only_generated_bookkeeping(tmp_path):
    counted = {"module.py": b"source", "METADATA": b"metadata", "native.so": b"native"}
    ignored = {"module.pyc": b"bytecode", "RECORD": b"paths", "direct_url.json": b"local path"}
    for name, content in (counted | ignored).items():
        (tmp_path / name).write_bytes(content)

    class Distribution:
        files = [Path(name) for name in counted | ignored]

        def locate_file(self, file):
            return tmp_path / file

    assert CHECK["installed_bytes"](Distribution()) == sum(map(len, counted.values()))
