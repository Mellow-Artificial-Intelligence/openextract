"""Oneshot, batch, and swarm parity for model_settings and timeout."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from openextract import (
    define_agent,
    extract,
    extract_async,
    extract_many,
    extract_many_async,
    extract_many_with_results,
    extract_swarm,
    extract_swarm_async,
    extract_swarm_with_results,
    extract_with_result,
    extract_with_usage,
    extract_with_usage_async,
    iter_extract_many_async,
)
from openextract._agent import _build_agent
from tests.test_extract import _make_agent_mock, _make_async_agent_mock


class Person(BaseModel):
    name: str
    age: int


_SETTINGS = {"temperature": 0}
_MERGED = {"temperature": 0, "timeout": 12.0}
_INVALID_TIMEOUTS = (False, 0, -1, float("inf"), "slow")


def _test_model() -> TestModel:
    return TestModel(custom_output_args={"name": "Ada", "age": 36})


def test_extract_passes_model_settings_and_timeout(mocker):
    expected = Person(name="Ada", age=36)
    agent_cls, _ = _make_agent_mock(mocker, output=expected)

    result = extract(
        Person,
        "openai:gpt-5",
        b"Ada is 36",
        media_type="text/plain",
        model_settings=_SETTINGS,
        timeout=12,
    )

    assert result == expected
    assert agent_cls.call_args.kwargs["model_settings"] == _MERGED


def test_extract_timeout_overrides_model_settings_timeout(mocker):
    agent_cls, _ = _make_agent_mock(mocker, output=Person(name="Ada", age=36))

    extract(
        Person,
        "openai:gpt-5",
        b"x",
        media_type="text/plain",
        model_settings={"temperature": 0, "timeout": 99},
        timeout=12,
    )

    assert agent_cls.call_args.kwargs["model_settings"] == _MERGED


def test_extract_default_settings_are_none(mocker):
    agent_cls, _ = _make_agent_mock(mocker, output=Person(name="Ada", age=36))

    extract(Person, "openai:gpt-5", b"x", media_type="text/plain")

    assert agent_cls.call_args.kwargs["model_settings"] is None


@pytest.mark.parametrize("value", _INVALID_TIMEOUTS)
def test_extract_invalid_timeout_raises_before_model_call(mocker, value):
    agent = mocker.patch("openextract._agent.Agent")

    with pytest.raises(ValueError, match="finite positive"):
        extract(
            Person,
            "openai:gpt-5",
            b"x",
            media_type="text/plain",
            timeout=value,  # type: ignore[arg-type]
        )

    agent.assert_not_called()


def test_extract_with_usage_passes_settings(mocker):
    expected = Person(name="Ada", age=36)
    agent_cls, _ = _make_agent_mock(mocker, output=expected)
    output, _usage = extract_with_usage(
        Person,
        "openai:gpt-5",
        b"x",
        media_type="text/plain",
        model_settings=_SETTINGS,
        timeout=12,
    )
    assert output == expected
    assert agent_cls.call_args.kwargs["model_settings"] == _MERGED


def test_extract_with_result_passes_settings(mocker):
    expected = Person(name="Ada", age=36)
    agent_cls, _ = _make_agent_mock(mocker, output=expected)
    result = extract_with_result(
        Person,
        "openai:gpt-5",
        b"x",
        media_type="text/plain",
        model_settings=_SETTINGS,
        timeout=12,
    )
    assert result.output == expected
    assert agent_cls.call_args.kwargs["model_settings"] == _MERGED


async def test_extract_async_passes_model_settings_and_timeout(mocker):
    expected = Person(name="Ada", age=36)
    agent_cls, _ = _make_async_agent_mock(mocker, output=expected)

    result = await extract_async(
        Person,
        "openai:gpt-5",
        b"x",
        media_type="text/plain",
        model_settings=_SETTINGS,
        timeout=12,
    )

    assert result == expected
    assert agent_cls.call_args.kwargs["model_settings"] == _MERGED


async def test_extract_with_usage_async_accepts_settings():
    output, usage = await extract_with_usage_async(
        Person,
        _test_model(),
        b"Ada is 36",
        media_type="text/plain",
        model_settings=_SETTINGS,
        timeout=12,
    )
    assert output == Person(name="Ada", age=36)
    assert usage.total_tokens >= 0


def test_extract_testmodel_accepts_settings():
    assert extract(
        Person,
        _test_model(),
        b"Ada is 36",
        media_type="text/plain",
        model_settings=_SETTINGS,
        timeout=12,
    ) == Person(name="Ada", age=36)


def test_solo_agent_oneshot_still_applies_timeout(mocker):
    agent_cls, _ = _make_agent_mock(mocker, output=Person(name="Ada", age=36))
    specialist = define_agent(
        "Solo",
        model="openai:gpt-5",
        instructions="focus",
        output_schema=Person,
    )

    extract(specialist, b"x", media_type="text/plain", timeout=12)

    assert agent_cls.call_args.kwargs["model_settings"] == {"timeout": 12.0}


def test_batch_passes_settings_to_shared_agent(mocker):
    captured: list[object] = []
    model = _test_model()

    def build(schema, built_model, instructions, **kwargs):
        captured.append(kwargs.get("model_settings"))
        return _build_agent(schema, built_model, instructions, **kwargs)

    mocker.patch("openextract._batch._build_agent", side_effect=build)

    results = extract_many(
        Person,
        model,
        [b"Ada is 36"],
        media_type="text/plain",
        model_settings=_SETTINGS,
        timeout=12,
    )

    assert results == [Person(name="Ada", age=36)]
    assert captured == [_MERGED]


async def test_batch_async_and_iter_accept_settings():
    model = _test_model()
    results = await extract_many_async(
        Person,
        model,
        [b"Ada is 36"],
        media_type="text/plain",
        timeout=12,
    )
    assert results == [Person(name="Ada", age=36)]

    streamed = [
        item
        async for item in iter_extract_many_async(
            Person,
            model,
            [b"Ada is 36"],
            media_type="text/plain",
            model_settings=_SETTINGS,
            timeout=12,
        )
    ]
    assert streamed == [(0, Person(name="Ada", age=36))]

    rich = extract_many_with_results(
        Person,
        model,
        [b"Ada is 36"],
        media_type="text/plain",
        timeout=12,
    )
    assert rich[0].output == Person(name="Ada", age=36)


@pytest.mark.parametrize("value", _INVALID_TIMEOUTS)
def test_batch_invalid_timeout_raises_before_model_call(mocker, value):
    build = mocker.patch("openextract._batch._build_agent")

    with pytest.raises(ValueError, match="finite positive"):
        extract_many(
            Person,
            "openai:gpt-5",
            [b"x"],
            media_type="text/plain",
            timeout=value,  # type: ignore[arg-type]
        )

    build.assert_not_called()


def test_swarm_passes_settings_to_each_agent(mocker):
    captured: list[object] = []
    model = _test_model()

    def build(schema, built_model, instructions, **kwargs):
        captured.append(kwargs.get("model_settings"))
        return _build_agent(schema, built_model, instructions, **kwargs)

    mocker.patch("openextract._swarm._build_agent", side_effect=build)

    result = extract_swarm(
        Person,
        model,
        b"Ada is 36",
        media_type="text/plain",
        size=2,
        model_settings=_SETTINGS,
        timeout=12,
    )

    assert result == Person(name="Ada", age=36)
    assert captured == [_MERGED, _MERGED]


async def test_swarm_async_and_results_accept_settings():
    model = _test_model()
    assert await extract_swarm_async(
        Person,
        model,
        b"Ada is 36",
        media_type="text/plain",
        timeout=12,
    ) == Person(name="Ada", age=36)

    swarm = extract_swarm_with_results(
        Person,
        model,
        b"Ada is 36",
        media_type="text/plain",
        model_settings=_SETTINGS,
        timeout=12,
    )
    assert swarm.output == Person(name="Ada", age=36)


@pytest.mark.parametrize("value", _INVALID_TIMEOUTS)
def test_swarm_invalid_timeout_raises_before_model_call(mocker, value):
    build = mocker.patch("openextract._swarm._build_agent")

    with pytest.raises(ValueError, match="finite positive"):
        extract_swarm(
            Person,
            "openai:gpt-5",
            b"x",
            media_type="text/plain",
            timeout=value,  # type: ignore[arg-type]
        )

    build.assert_not_called()


def test_group_agent_oneshot_forwards_timeout_to_swarm(mocker):
    captured: list[object] = []

    def build(schema, built_model, instructions, **kwargs):
        captured.append(kwargs.get("model_settings"))
        return _build_agent(schema, built_model, instructions, **kwargs)

    mocker.patch("openextract._swarm._build_agent", side_effect=build)
    group = define_agent(
        "Group",
        output_schema=Person,
        subagents=[
            define_agent("A", model=_test_model()),
            define_agent("B", model=_test_model()),
        ],
    )

    assert extract(group, b"Ada is 36", media_type="text/plain", timeout=12) == Person(
        name="Ada", age=36
    )
    assert captured == [{"timeout": 12.0}, {"timeout": 12.0}]
