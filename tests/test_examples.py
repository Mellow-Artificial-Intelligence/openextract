"""Smoke tests for runnable examples."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

import examples.advanced.openrouter_jev as jev

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


def _decisions_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openrouter_jev_example() -> None:
    result = _run("examples.advanced.openrouter_jev", ["--fixture"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == jev.FIXTURE_OUTPUT


def test_openrouter_jev_example_default() -> None:
    result = _run("examples.advanced.openrouter_jev")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["decision"]["action"] == "approve"


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


def test_openrouter_jev_live_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = _run("examples.advanced.openrouter_jev", ["--live"])
    assert result.returncode == 1
    assert "OPENROUTER_API_KEY is required" in result.stdout
    assert "sk-" not in result.stdout + result.stderr


def test_jev_decision_schema() -> None:
    parsed = jev.Decision.model_validate(jev.FIXTURE_OUTPUT["decision"])
    assert parsed.action == "approve"
    assert parsed.auto_refund is True
    with pytest.raises(ValidationError):
        jev.Decision.model_validate({**jev.FIXTURE_OUTPUT["decision"], "action": "defer"})


def test_jev_decisions_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_DECISIONS_MODEL", raising=False)
    monkeypatch.delenv("OPENEXTRACT_MODEL", raising=False)
    assert jev.JEV_MODEL == "~typesafe/jev-latest"
    assert jev.DECISIONS_URL == "https://openrouter.ai/api/alpha/decisions"
    assert jev.decisions_model() == jev.JEV_MODEL
    monkeypatch.setenv("OPENROUTER_DECISIONS_MODEL", "typesafe/jev-1.13")
    monkeypatch.setenv("OPENEXTRACT_MODEL", "openrouter:~typesafe/jev-latest")
    assert jev.decisions_model() == "typesafe/jev-1.13"


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


def test_jev_fixture_skips_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fixture path must not call the Decisions API")

    monkeypatch.setattr(jev, "submit_decisions", boom)
    result = jev.run_cookbook(jev.SAMPLE_MEMO)
    assert result.model_dump() == jev.FIXTURE_OUTPUT


def test_jev_extract_state_uses_testmodel(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_extract(**kwargs: object) -> jev.TicketState:
        captured.update(kwargs)
        return jev.TicketState.model_validate(jev.FIXTURE_STATE)

    monkeypatch.setattr(jev, "extract", fake_extract)
    state = jev.extract_state(jev.SAMPLE_MEMO)
    assert state.ticket_id == "4412"
    assert captured["schema"] is jev.TicketState
    assert isinstance(captured["model"], TestModel)
    assert captured["input_file"] == jev.SAMPLE_MEMO.encode()
    assert captured["media_type"] == "text/plain"
    assert captured["instructions"] == jev.STATE_INSTRUCTIONS
    assert captured["model"].custom_output_args == jev.FIXTURE_STATE  # type: ignore[union-attr]


def test_jev_submit_decisions_request_shape() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["authorization"] = request.headers["authorization"]
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=jev.FIXTURE_RESPONSE)

    result = jev.submit_decisions(
        jev.TicketState.model_validate(jev.FIXTURE_STATE),
        api_key="test-key-do-not-log",
        client=_decisions_client(handler),
    )
    assert result == jev.FIXTURE_RESPONSE
    assert captured["url"] == jev.DECISIONS_URL
    assert captured["method"] == "POST"
    assert captured["authorization"] == "Bearer test-key-do-not-log"
    assert captured["content_type"] == "application/json"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == jev.JEV_MODEL
    assert body["state"] == jev.FIXTURE_STATE
    assert body["questions"] == jev.QUESTIONS


def test_jev_submit_decisions_owns_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=jev.FIXTURE_RESPONSE)

    real = _decisions_client(handler)

    def fake_client(*_args: object, **kwargs: object) -> httpx.Client:
        created.update(kwargs)
        return real

    monkeypatch.setattr(jev.httpx, "Client", fake_client)
    result = jev.submit_decisions(jev.FIXTURE_STATE, api_key="k")
    assert result == jev.FIXTURE_RESPONSE
    assert created["timeout"] == 30.0
    # httpx.Client.close() is idempotent; owned-client path must have closed it.
    real.close()


def test_jev_submit_decisions_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            text="~typesafe/jev-latest is a decisions model and cannot be used "
            "with the chat/completions endpoint.",
        )

    with pytest.raises(jev.DecisionsError, match="Decisions API returned 400"):
        jev.submit_decisions(jev.FIXTURE_STATE, api_key="k", client=_decisions_client(handler))


def test_jev_submit_decisions_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(jev.DecisionsError, match="Decisions API request failed"):
        jev.submit_decisions(jev.FIXTURE_STATE, api_key="k", client=_decisions_client(handler))


def test_jev_submit_decisions_non_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>")

    with pytest.raises(jev.DecisionsError, match="non-JSON"):
        jev.submit_decisions(jev.FIXTURE_STATE, api_key="k", client=_decisions_client(handler))


def test_jev_submit_decisions_non_object_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["nope"])

    with pytest.raises(jev.DecisionsError, match="non-object"):
        jev.submit_decisions(jev.FIXTURE_STATE, api_key="k", client=_decisions_client(handler))


def test_jev_decision_from_response_errors() -> None:
    with pytest.raises(jev.DecisionsError, match="missing answers"):
        jev.decision_from_response({})
    with pytest.raises(jev.DecisionsError, match="missing answer 'action'"):
        jev.decision_from_response({"answers": {}})
    with pytest.raises(jev.DecisionsError, match="type 'noul' != 'choice'"):
        jev.decision_from_response({"answers": {"action": {"type": "noul", "noul": 0.1}}})
    with pytest.raises(jev.DecisionsError, match="unsupported answer type"):
        jev.decision_from_response({"answers": {"action": {"type": "text", "text": "x"}}})
    with pytest.raises(jev.DecisionsError, match="invalid action"):
        jev.decision_from_response(
            {
                "answers": {
                    "action": {"type": "choice"},
                    "auto_refund": {"type": "noul", "noul": 0.9},
                    "urgency": {"type": "score", "score": 1.0},
                }
            }
        )
    with pytest.raises(jev.DecisionsError, match="Could not map"):
        jev.decision_from_response(
            {
                "answers": {
                    "action": {"type": "choice", "choice": "approve", "confidence": 0.9},
                    "auto_refund": {"type": "noul"},
                    "urgency": {"type": "score", "score": 1.0},
                }
            }
        )


def test_jev_decision_from_response_noul_threshold() -> None:
    payload = {
        "model": "typesafe/jev-1.13",
        "answers": {
            "action": {"type": "choice", "choice": "reject", "confidence": 0.6},
            "auto_refund": {"type": "noul", "noul": 0.49},
            "urgency": {"type": "score", "score": 0.2},
        },
    }
    decision = jev.decision_from_response(payload)
    assert decision.action == "reject"
    assert decision.auto_refund is False
    assert decision.model == "typesafe/jev-1.13"


def test_jev_run_cookbook_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-do-not-log")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=jev.FIXTURE_RESPONSE)

    result = jev.run_cookbook(jev.SAMPLE_MEMO, live=True, client=_decisions_client(handler))
    assert result.model_dump() == jev.FIXTURE_OUTPUT
    assert seen["authorization"] == "Bearer test-key-do-not-log"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == jev.JEV_MODEL
    assert body["state"] == jev.FIXTURE_STATE


def test_jev_main_live_mocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-do-not-log")

    def fake_submit(*_args: object, **_kwargs: object) -> dict[str, object]:
        return jev.FIXTURE_RESPONSE

    monkeypatch.setattr(jev, "submit_decisions", fake_submit)
    jev.main(["--live"])
    out = capsys.readouterr().out
    assert json.loads(out) == jev.FIXTURE_OUTPUT
    assert "test-key-do-not-log" not in out
    assert jev.USAGE.startswith("Usage:")


def test_jev_main_live_http_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-do-not-log")

    def fake_run(*_args: object, **_kwargs: object) -> jev.CookbookResult:
        raise jev.DecisionsError("Decisions API returned 400: decisions model")

    monkeypatch.setattr(jev, "run_cookbook", fake_run)
    with pytest.raises(SystemExit) as exc:
        jev.main(["--live"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Decisions API returned 400" in out
    assert "test-key-do-not-log" not in out


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
