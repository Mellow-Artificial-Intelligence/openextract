"""Smoke tests for runnable examples."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

import examples.advanced.openrouter_jev as jev
from examples._shared import OPENROUTER_MODEL, openrouter_model

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
    "examples.documents.line_item_extraction",
    "examples.advanced.openrouter_jev",
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


def test_openrouter_jev_example() -> None:
    result = _run("examples.advanced.openrouter_jev", ["--fixture"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == jev.FIXTURE_OUTPUT


def test_openrouter_jev_example_default() -> None:
    result = _run("examples.advanced.openrouter_jev")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["decision"] == "approve"


def test_openrouter_jev_usage_unknown_flag() -> None:
    result = _run("examples.advanced.openrouter_jev", ["--nope"])
    assert result.returncode == 1
    assert "Usage:" in result.stdout


def test_openrouter_jev_rejects_both_flags() -> None:
    result = _run("examples.advanced.openrouter_jev", ["--fixture", "--live"])
    assert result.returncode == 1
    assert "not both" in result.stdout


def test_openrouter_jev_missing_file() -> None:
    result = _run("examples.advanced.openrouter_jev", ["missing-memo.txt"])
    assert result.returncode == 1
    assert "Missing file:" in result.stdout


def test_openrouter_model_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENEXTRACT_MODEL", raising=False)
    assert OPENROUTER_MODEL == "openrouter:~typesafe/jev-latest"
    assert openrouter_model() == OPENROUTER_MODEL


def test_openrouter_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENEXTRACT_MODEL", "openrouter:anthropic/claude-sonnet-4")
    assert openrouter_model() == "openrouter:anthropic/claude-sonnet-4"


def test_jev_decision_schema() -> None:
    parsed = jev.Decision.model_validate(jev.FIXTURE_OUTPUT)
    assert parsed.decision == "approve"
    assert parsed.confidence == 0.91
    assert parsed.risk_flags == []
    with pytest.raises(ValidationError):
        jev.Decision.model_validate({**jev.FIXTURE_OUTPUT, "decision": "defer"})


def test_jev_resolve_model_fixture() -> None:
    model = jev.resolve_model()
    assert isinstance(model, TestModel)
    assert model.custom_output_args == jev.FIXTURE_OUTPUT
    assert jev.fixture_model().custom_output_args == jev.FIXTURE_OUTPUT


def test_jev_resolve_model_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENEXTRACT_MODEL", raising=False)
    assert jev.resolve_model(live=True) == OPENROUTER_MODEL
    monkeypatch.setenv("OPENEXTRACT_MODEL", "openrouter:other/model")
    assert jev.resolve_model(live=True) == "openrouter:other/model"


def test_jev_parse_argv_and_load_memo(tmp_path: Path) -> None:
    assert jev.parse_argv([]) == (False, jev.SAMPLE_MEMO)
    assert jev.parse_argv(["--fixture"]) == (False, jev.SAMPLE_MEMO)
    assert jev.parse_argv(["--live"]) == (True, jev.SAMPLE_MEMO)
    memo = tmp_path / "memo.txt"
    memo.write_text("Escalate if the card is not on file.", encoding="utf-8")
    assert jev.parse_argv(["--fixture", str(memo)]) == (
        False,
        "Escalate if the card is not on file.",
    )
    assert jev.parse_argv(["--live", str(memo)]) == (
        True,
        "Escalate if the card is not on file.",
    )
    assert jev.load_memo(None) == jev.SAMPLE_MEMO
    assert jev.load_memo(str(memo)) == "Escalate if the card is not on file."


def test_jev_parse_argv_extra_path() -> None:
    with pytest.raises(SystemExit) as exc:
        jev.parse_argv(["a.txt", "b.txt"])
    assert exc.value.code == 1


def test_jev_extract_decision_fixture() -> None:
    result = jev.extract_decision(jev.SAMPLE_MEMO)
    assert result.model_dump() == jev.FIXTURE_OUTPUT


def test_jev_extract_call_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_extract(**kwargs: object) -> jev.Decision:
        captured.update(kwargs)
        return jev.Decision.model_validate(jev.FIXTURE_OUTPUT)

    monkeypatch.delenv("OPENEXTRACT_MODEL", raising=False)
    monkeypatch.setattr(jev, "extract", fake_extract)
    result = jev.extract_decision(jev.SAMPLE_MEMO, live=True)
    assert result.decision == "approve"
    assert captured["schema"] is jev.Decision
    assert captured["model"] == OPENROUTER_MODEL
    assert captured["input_file"] == jev.SAMPLE_MEMO.encode()
    assert captured["media_type"] == "text/plain"
    assert captured["instructions"] == jev.INSTRUCTIONS


def test_jev_main_live_mocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_extract(**kwargs: object) -> jev.Decision:
        return jev.Decision.model_validate(jev.FIXTURE_OUTPUT)

    monkeypatch.setattr(jev, "extract", fake_extract)
    jev.main(["--live"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "approve"
    assert jev.USAGE.startswith("Usage:")


def test_jev_main_reads_sys_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["openrouter_jev", "--fixture"])
    jev.main()
    assert json.loads(capsys.readouterr().out) == jev.FIXTURE_OUTPUT


def test_line_item_extraction_example() -> None:
    result = _run("examples.documents.line_item_extraction")
    assert result.returncode == 0, result.stderr
    assert "Acme Corp" in result.stdout
    assert "Widget" in result.stdout
    assert "Gadget" in result.stdout


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
    if not os.environ.get("OPENEXTRACT_RUN_EXAMPLES"):
        pytest.skip("Set OPENEXTRACT_RUN_EXAMPLES=1 to run live example tests")
    result = _run(module, args)
    assert result.returncode == 0, result.stderr or result.stdout
