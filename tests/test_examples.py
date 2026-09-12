"""Smoke tests for runnable examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

API_EXAMPLES: list[tuple[str, list[str]]] = [
    ("examples.basic.local_file", ["--fixture"]),
    ("examples.basic.bytes_input", []),
    ("examples.images.document_summary", []),
    ("examples.advanced.extract_with_usage", ["--fixture"]),
]

ALL_MODULES = [
    "examples.basic.local_file",
    "examples.basic.bytes_input",
    "examples.basic.url_extract",
    "examples.images.document_summary",
    "examples.images.receipt_extraction",
    "examples.documents.invoice_extraction",
    "examples.batch.batch_extract",
    "examples.batch.stream_batch_extract",
    "examples.async.async_extract",
    "examples.advanced.extract_with_usage",
    "examples.advanced.extract_with_citations",
    "examples.advanced.retry_extract",
    "examples.advanced.reusable_sessions",
    "examples.advanced.extraction_styles",
    "examples.advanced.swarm_extract",
    "examples.advanced.error_handling",
    "examples.audio.meeting_notes",
]


def _run(module: str, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *(extra or [])],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("module", ALL_MODULES)
def test_example_module_is_runnable(module: str) -> None:
    """Each example module at least starts; input-driven examples exit 1."""
    if module == "examples.audio.meeting_notes":
        result = _run(module)
        assert result.returncode == 1
        assert "Usage:" in result.stdout + result.stderr
        return
    if module == "examples.basic.local_file":
        result = _run(module)
        assert result.returncode == 1
        assert "Usage:" in result.stdout + result.stderr
        return
    if module in {
        "examples.images.receipt_extraction",
        "examples.documents.invoice_extraction",
    }:
        result = _run(module)
        assert result.returncode == 1
        assert "Usage:" in result.stdout + result.stderr
        return
    if module == "examples.advanced.extract_with_usage":
        result = _run(module)
        assert result.returncode == 1
        assert "Usage:" in result.stdout + result.stderr
        return


def test_error_handling_example() -> None:
    result = _run("examples.advanced.error_handling")
    assert result.returncode == 0, result.stderr
    assert "UrlFetchError" in result.stdout
    assert "completed successfully" in result.stdout


def test_reusable_sessions_example() -> None:
    result = _run("examples.advanced.reusable_sessions")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("ada@example.com") == 2


def test_extraction_styles_example() -> None:
    result = _run("examples.advanced.extraction_styles")
    assert result.returncode == 0, result.stderr
    assert "Q4 notes" in result.stdout


def test_extract_with_citations_example() -> None:
    result = _run("examples.advanced.extract_with_citations")
    assert result.returncode == 0, result.stderr
    assert "Acme Corp" in result.stdout
    assert "citations:" in result.stdout
    assert "ExtractBench:" in result.stdout


def test_stream_batch_extract_example() -> None:
    result = _run("examples.batch.stream_batch_extract")
    assert result.returncode == 0, result.stderr
    assert "extract_many" in result.stdout
    assert "iter_extract_many_async" in result.stdout
    assert "ada@example.com" in result.stdout
    assert "TypeError" in result.stdout
    assert "broken.bin" in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize("module,args", API_EXAMPLES)
def test_api_examples_with_fixture(module: str, args: list[str]) -> None:
    """Live model call; skipped in CI unless OPENEXTRACT_RUN_EXAMPLES=1."""
    import os

    if not os.environ.get("OPENEXTRACT_RUN_EXAMPLES"):
        pytest.skip("Set OPENEXTRACT_RUN_EXAMPLES=1 to run live example tests")
    result = _run(module, args)
    assert result.returncode == 0, result.stderr or result.stdout
