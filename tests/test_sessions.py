"""Tests for reusable extractor sessions."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

from openextract import (
    AsyncExtractor,
    ExtractionInput,
    ExtractionResult,
    Extractor,
    ExtractProgress,
    ModelError,
    RetryPolicy,
    SchemaValidationError,
    Usage,
    extract,
)
from openextract.exceptions import ProviderNotInstalledError


class Person(BaseModel):
    name: str
    age: int


class FakeResult:
    def __init__(self, output, usage=None):
        self.output = output
        self._usage = usage or SimpleNamespace(
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
        )

    def usage(self):
        return self._usage


class FakeAgent:
    def __init__(self, outcomes, *, enter_error=None, exit_result=False, exit_error=None):
        self.outcomes = iter(outcomes)
        self.enter_error = enter_error
        self.exit_result = exit_result
        self.exit_error = exit_error
        self.enter_count = 0
        self.exit_count = 0
        self.run_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self

    async def __aexit__(self, *exc_info):
        self.exit_count += 1
        if self.exit_error is not None:
            raise self.exit_error
        return self.exit_result

    def _next(self):
        self.run_count += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeResult(outcome)

    def run_sync(self, inputs):
        return self._next()

    async def run(self, inputs):
        return self._next()


def test_configured_model_works_in_session_and_function_wrapper():
    model = TestModel(custom_output_args={"name": "Ada", "age": 36})

    with Extractor(Person, model) as extractor:
        first = extractor.extract(b"first", media_type="text/plain")
        second = extractor.extract(b"second", media_type="text/plain")

    assert first == second == Person(name="Ada", age=36)
    assert extract(Person, model, b"one-shot", media_type="text/plain") == first


def test_sync_session_accepts_path_input(tmp_path):
    local = tmp_path / "input.txt"
    local.write_bytes(b"hello")
    agent = FakeAgent([{"name": "Ada", "age": 36}])

    with Extractor(Person, agent=agent) as extractor:
        result = extractor.extract(local)

    assert result == Person(name="Ada", age=36)


async def test_async_session_accepts_path_input(tmp_path):
    local = tmp_path / "input.txt"
    local.write_bytes(b"hello")
    agent = FakeAgent([{"name": "Grace", "age": 85}])

    async with AsyncExtractor(Person, agent=agent) as extractor:
        result = await extractor.extract(local)

    assert result == Person(name="Grace", age=85)


def test_sync_session_reuses_agent_and_clients_and_returns_usage(mocker):
    agent = FakeAgent([{"name": "Ada", "age": 36}, {"name": "Grace", "age": 85}])
    agent_factory = mocker.patch("openextract._agent.Agent", return_value=agent)
    client = MagicMock()
    client_factory = mocker.patch("openextract._session.httpx.Client", return_value=client)

    with Extractor(
        Person,
        "openai:gpt-test",
        model_settings={"temperature": 0},
        timeout=12,
        instrument=True,
    ) as extractor:
        first = extractor.extract(b"one", media_type="text/plain")
        second, usage = extractor.extract_with_usage(b"two", media_type="text/plain")

    assert first == Person(name="Ada", age=36)
    assert second == Person(name="Grace", age=85)
    assert usage == Usage(input_tokens=1, output_tokens=2, total_tokens=3)
    assert agent_factory.call_count == 1
    assert agent_factory.call_args.kwargs["model_settings"] == {
        "temperature": 0,
        "timeout": 12.0,
    }
    assert len(agent_factory.call_args.kwargs["capabilities"]) == 1
    client_factory.assert_called_once_with(follow_redirects=False, timeout=30.0)
    client.close.assert_called_once_with()
    assert agent.enter_count == agent.exit_count == 1
    assert agent.run_count == 2


def test_sync_session_reuses_client_for_url_input(mocker):
    agent = FakeAgent([{"name": "Ada", "age": 36}])
    client = MagicMock()
    mocker.patch("openextract._session.httpx.Client", return_value=client)
    read = mocker.patch(
        "openextract._media._read_url_with_client",
        return_value=(b"document", {"content-type": "text/plain"}),
    )

    with Extractor(Person, agent=agent) as extractor:
        result = extractor.extract("https://example.com/document")

    assert result == Person(name="Ada", age=36)
    read.assert_called_once_with(
        "https://example.com/document",
        client,
        limit=50 * 1024 * 1024,
    )


def test_instrumentation_settings_and_validation(mocker):
    settings = InstrumentationSettings(include_content=False)
    agent_factory = mocker.patch("openextract._agent.Agent", return_value=FakeAgent([]))

    Extractor(Person, "test", instrument=settings)

    capability = agent_factory.call_args.kwargs["capabilities"][0]
    assert capability.settings is settings
    with pytest.raises(TypeError, match="InstrumentationSettings"):
        Extractor(Person, "test", instrument=object())  # type: ignore[arg-type]


def test_configured_model_missing_provider_is_actionable(mocker):
    model = TestModel(custom_output_args={"name": "Ada", "age": 36})
    mocker.patch("openextract._agent.Agent", side_effect=ImportError("missing sdk"))

    with pytest.raises(ProviderNotInstalledError, match="configured model.*missing sdk"):
        Extractor(Person, model)


def test_sync_injected_agent_retries_and_validates_output(mocker):
    retryable = ModelError("temporary", retryable=True)
    agent = FakeAgent([retryable, {"name": "Ada", "age": 36}])
    sleep = mocker.patch("openextract._retry.time.sleep")

    with Extractor(
        Person,
        agent=agent,
        retry_policy=RetryPolicy(max_retries=1, backoff=0, max_backoff=0),
    ) as extractor:
        assert extractor.extract(b"x", media_type="text/plain") == Person(name="Ada", age=36)

    sleep.assert_called_once_with(0)
    assert agent.run_count == 2


def test_sync_injected_agent_invalid_output_is_mapped():
    agent = FakeAgent([{"name": "Ada", "age": "invalid"}])

    with (
        Extractor(Person, agent=agent) as extractor,
        pytest.raises(SchemaValidationError, match=r"age: expected int, got str"),
    ):
        extractor.extract(b"x", media_type="text/plain")


def test_sync_lifecycle_and_thread_guards():
    agent = FakeAgent([{"name": "Ada", "age": 36}])
    extractor = Extractor(Person, agent=agent)

    with pytest.raises(RuntimeError, match="context manager"):
        extractor.extract(b"x", media_type="text/plain")

    with extractor:
        errors = []

        def use_from_other_thread():
            try:
                extractor.extract(b"x", media_type="text/plain")
            except Exception as exc:  # noqa: BLE001 - asserting the public guard
                errors.append(exc)
            try:
                extractor.close()
            except Exception as exc:  # noqa: BLE001 - asserting the public guard
                errors.append(exc)

        thread = threading.Thread(target=use_from_other_thread)
        thread.start()
        thread.join()
        assert isinstance(errors[0], RuntimeError)
        assert "thread" in str(errors[0])
        assert isinstance(errors[1], RuntimeError)
        assert "thread" in str(errors[1])

        with pytest.raises(RuntimeError, match="already entered"):
            extractor.__enter__()

    with pytest.raises(RuntimeError, match="closed"):
        extractor.__enter__()


def test_sync_close_before_enter_and_context_suppression():
    closed = Extractor(Person, agent=FakeAgent([]))
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        closed.__enter__()

    suppressing = Extractor(Person, agent=FakeAgent([], exit_result=True))
    with suppressing:
        raise RuntimeError("suppressed")


def test_sync_enter_and_exit_failures_still_close_owned_resources(mocker):
    enter_client = MagicMock()
    exit_client = MagicMock()
    mocker.patch(
        "openextract._session.httpx.Client",
        side_effect=[enter_client, exit_client],
    )
    entering = Extractor(Person, agent=FakeAgent([], enter_error=RuntimeError("enter")))
    with pytest.raises(RuntimeError, match="enter"):
        entering.__enter__()
    enter_client.close.assert_called_once_with()

    exiting = Extractor(Person, agent=FakeAgent([], exit_error=RuntimeError("exit")))
    with pytest.raises(RuntimeError, match="exit"), exiting:
        pass
    exit_client.close.assert_called_once_with()


@pytest.mark.parametrize("value", [False, 0, -1, float("inf"), "slow"])
def test_session_timeout_validation(value):
    with pytest.raises(ValueError, match="finite positive"):
        Extractor(Person, "test", timeout=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite positive"):
        Extractor(Person, "test", url_timeout=value)  # type: ignore[arg-type]


def test_session_constructor_rejects_ambiguous_configuration():
    agent = FakeAgent([])
    with pytest.raises(TypeError, match="model is required"):
        Extractor(Person)
    with pytest.raises(ValueError, match="mutually exclusive"):
        Extractor(Person, "test", agent=agent)
    with pytest.raises(ValueError, match="configured on an injected agent"):
        Extractor(Person, agent=agent, instructions="extract")
    with pytest.raises(ValueError, match="configured on an injected agent"):
        Extractor(Person, agent=agent, model_settings={"temperature": 0})
    with pytest.raises(ValueError, match="configured on an injected agent"):
        Extractor(Person, agent=agent, timeout=1)
    with pytest.raises(ValueError, match="instrument"):
        Extractor(Person, agent=agent, instrument=True)
    with pytest.raises(TypeError, match="RetryPolicy"):
        Extractor(Person, "test", retry_policy=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cite cannot be used"):
        Extractor(Person, agent=agent, cite=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"backoff": -1},
        {"max_backoff": float("nan")},
    ],
)
def test_retry_policy_validates_options(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


async def test_sync_extractor_rejects_running_event_loop():
    extractor = Extractor(Person, agent=FakeAgent([]))
    with pytest.raises(RuntimeError, match="AsyncExtractor"):
        extractor.__enter__()


async def test_async_configured_model_and_usage():
    model = TestModel(custom_output_args={"name": "Grace", "age": 85})
    async with AsyncExtractor(Person, model) as extractor:
        output, usage = await extractor.extract_with_usage(b"x", media_type="text/plain")

    assert output == Person(name="Grace", age=85)
    assert usage.total_tokens > 0


async def test_async_session_reuses_agent_and_retries(mocker):
    retryable = ModelError("temporary", retryable=True)
    agent = FakeAgent([retryable, {"name": "Ada", "age": 36}, {"name": "Grace", "age": 85}])
    sleep = mocker.patch("openextract._retry.asyncio.sleep")
    client = MagicMock()
    client.aclose = AsyncMock()
    client_factory = mocker.patch("openextract._session.httpx.AsyncClient", return_value=client)

    async with AsyncExtractor(
        Person,
        agent=agent,
        retry_policy=RetryPolicy(max_retries=1, backoff=0, max_backoff=0),
        url_timeout=5,
    ) as extractor:
        first = await extractor.extract(b"one", media_type="text/plain")
        second = await extractor.extract(b"two", media_type="text/plain")

    assert first == Person(name="Ada", age=36)
    assert second == Person(name="Grace", age=85)
    assert agent.enter_count == agent.exit_count == 1
    assert agent.run_count == 3
    sleep.assert_awaited_once_with(0)
    client_factory.assert_called_once_with(follow_redirects=False, timeout=5.0)


async def test_async_invalid_output_and_lifecycle_guards():
    agent = FakeAgent([{"name": "Ada", "age": "invalid"}])
    extractor = AsyncExtractor(Person, agent=agent)

    with pytest.raises(RuntimeError, match="context manager"):
        await extractor.extract(b"x", media_type="text/plain")

    async with extractor:
        with pytest.raises(SchemaValidationError, match=r"age: expected int, got str"):
            await extractor.extract(b"x", media_type="text/plain")
        with pytest.raises(RuntimeError, match="already entered"):
            await extractor.__aenter__()

    with pytest.raises(RuntimeError, match="closed"):
        await extractor.__aenter__()


async def test_async_close_before_enter_and_context_suppression():
    closed = AsyncExtractor(Person, agent=FakeAgent([]))
    await closed.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await closed.__aenter__()

    suppressing = AsyncExtractor(Person, agent=FakeAgent([], exit_result=True))
    async with suppressing:
        raise RuntimeError("suppressed")


async def test_async_enter_and_exit_failures_close_client(mocker):
    enter_client = MagicMock()
    enter_client.aclose = AsyncMock()
    exit_client = MagicMock()
    exit_client.aclose = AsyncMock()
    mocker.patch(
        "openextract._session.httpx.AsyncClient",
        side_effect=[enter_client, exit_client],
    )

    entering = AsyncExtractor(Person, agent=FakeAgent([], enter_error=RuntimeError("enter")))
    with pytest.raises(RuntimeError, match="enter"):
        await entering.__aenter__()
    enter_client.aclose.assert_called_once_with()

    exiting = AsyncExtractor(Person, agent=FakeAgent([], exit_error=RuntimeError("exit")))
    with pytest.raises(RuntimeError, match="exit"):
        async with exiting:
            pass
    exit_client.aclose.assert_called_once_with()


def test_async_session_rejects_a_different_event_loop():
    extractor = AsyncExtractor(Person, agent=FakeAgent([{"name": "Ada", "age": 36}]))
    first_loop = asyncio.new_event_loop()
    second_loop = asyncio.new_event_loop()
    try:
        first_loop.run_until_complete(extractor.__aenter__())
        with pytest.raises(RuntimeError, match="event loop"):
            second_loop.run_until_complete(extractor.extract(b"x", media_type="text/plain"))
        with pytest.raises(RuntimeError, match="event loop"):
            second_loop.run_until_complete(extractor.aclose())
        first_loop.run_until_complete(extractor.aclose())
    finally:
        first_loop.close()
        second_loop.close()


def test_sync_extract_with_result_shape_and_usage_match():
    model = TestModel(custom_output_args={"name": "Ada", "age": 36})
    named = ExtractionInput(b"Ada Lovelace", media_type="text/plain", name="bio.txt")
    with Extractor(Person, model) as extractor:
        result = extractor.extract_with_result(named)
        output, usage = extractor.extract_with_usage(named)

    assert isinstance(result, ExtractionResult)
    assert result.output == output == Person(name="Ada", age=36)
    assert result.usage == usage
    assert result.attempts == 1
    assert result.duration >= 0
    assert result.media_type == "text/plain"
    assert result.source == "bio.txt"
    assert result.warnings == ()
    assert result.citations == ()


def test_sync_extract_with_result_cite_and_pages(monkeypatch):
    from tests.pdf_fixture import synthetic_pdf

    monkeypatch.setattr("openextract._parse.DEFAULT_PARSE_WINDOW_CHARS", 40)
    pdf = synthetic_pdf(pages=["AAAA " * 30, "Ada Lovelace " + "BBBB " * 30, "CCCC " * 30])
    model = TestModel(
        custom_output_args={
            "output": {"name": "Ada", "age": 36},
            "citations": [{"field": "name", "quote": "Ada Lovelace", "page": 1}],
        }
    )
    events: list[ExtractProgress] = []
    with Extractor(Person, model, cite=True, pages=(1,)) as extractor:
        cited = extractor.extract_with_result(pdf, media_type="application/pdf")
        windowed = extractor.extract_with_result(pdf, media_type="application/pdf", pages=(1, 2, 3))
        override = extractor.extract_with_result(
            pdf, media_type="application/pdf", pages=(2,), on_progress=events.append
        )
        _, usage = extractor.extract_with_usage(pdf, media_type="application/pdf", pages=(2,))

    assert cited.output == Person(name="Ada", age=36)
    assert cited.citations[0].field == "name"
    assert cited.citations[0].quote == "Ada Lovelace"
    assert windowed.attempts >= 2
    assert windowed.output == Person(name="Ada", age=36)
    assert [event.page for event in events] == [2]
    assert override.usage == usage
    assert override.citations[0].field == "name"


async def test_async_extract_with_result_shape_cite_and_usage():
    model = TestModel(
        custom_output_args={
            "output": {"name": "Grace", "age": 85},
            "citations": [{"field": "name", "quote": "Grace", "page": 1}],
        }
    )
    named = ExtractionInput(b"Grace Hopper", media_type="text/plain", name="note.txt")
    async with AsyncExtractor(Person, model, cite=True) as extractor:
        result = await extractor.extract_with_result(named)
        output, usage = await extractor.extract_with_usage(named)

    assert isinstance(result, ExtractionResult)
    assert result.output == output == Person(name="Grace", age=85)
    assert result.usage == usage
    assert result.attempts == 1
    assert result.duration >= 0
    assert result.media_type == "text/plain"
    assert result.source == "note.txt"
    assert result.citations[0].field == "name"
    assert result.citations[0].quote == "Grace"


async def test_async_extract_with_result_pages_override(monkeypatch):
    from tests.pdf_fixture import synthetic_pdf

    monkeypatch.setattr("openextract._parse.DEFAULT_PARSE_WINDOW_CHARS", 40)
    pdf = synthetic_pdf(pages=["AAAA " * 30, "Ada Lovelace " + "BBBB " * 30, "CCCC " * 30])
    model = TestModel(custom_output_args={"name": "Ada", "age": 36})
    events: list[ExtractProgress] = []
    async with AsyncExtractor(Person, model, pages=(1,)) as extractor:
        result = await extractor.extract_with_result(
            pdf, media_type="application/pdf", pages=(2,), on_progress=events.append
        )
        windowed = await extractor.extract_with_result(
            pdf, media_type="application/pdf", pages=(1, 2, 3)
        )
        output, usage = await extractor.extract_with_usage(
            pdf, media_type="application/pdf", pages=(2,)
        )
    assert result.output == output == Person(name="Ada", age=36)
    assert result.usage == usage
    assert [event.page for event in events] == [2]
    assert windowed.attempts >= 2
    assert windowed.output == Person(name="Ada", age=36)


def test_sync_extract_with_result_counts_retries(mocker):
    retryable = ModelError("temporary", retryable=True)
    agent = FakeAgent([retryable, {"name": "Ada", "age": 36}])
    mocker.patch("openextract._retry.time.sleep")

    with Extractor(
        Person,
        agent=agent,
        retry_policy=RetryPolicy(max_retries=1, backoff=0, max_backoff=0),
    ) as extractor:
        result = extractor.extract_with_result(b"x", media_type="text/plain")

    assert result.output == Person(name="Ada", age=36)
    assert result.attempts == 2
    assert result.usage == Usage(input_tokens=1, output_tokens=2, total_tokens=3)
