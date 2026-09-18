"""Tests for agents passed to the one-shot extract APIs."""

import pytest
from pydantic import BaseModel

from openextract import (
    define_agent,
    define_remote_agent,
    extract,
    extract_async,
    extract_with_usage,
    extract_with_usage_async,
)
from openextract._extract import _plan_agent, _resolve_agent_call, _resolve_oneshot


class Person(BaseModel):
    name: str | None = None
    age: int | None = None


class FakeUsage:
    input_tokens = 2
    output_tokens = 3
    total_tokens = 5


class FakeResult:
    def __init__(self, output):
        self.output = output

    @property
    def usage(self):
        return FakeUsage()


class FakeAgent:
    def __init__(self, schema, model, instructions):
        self.schema = schema
        self.model = model
        self.instructions = instructions


def install_agents(mocker, outcomes: dict[str, dict]) -> list[FakeAgent]:
    """Serve every local model call from ``outcomes``, keyed by model id."""
    built: list[FakeAgent] = []

    def build(schema, model, instructions, **kwargs):
        agent = FakeAgent(schema, model, instructions)
        built.append(agent)
        return agent

    def run_sync(agent, inputs):
        return FakeResult(agent.schema.model_validate(outcomes[agent.model]))

    async def run_async(agent, inputs):
        return run_sync(agent, inputs)

    for module in ("openextract._agent", "openextract._swarm"):
        mocker.patch(f"{module}._build_agent", side_effect=build)
    mocker.patch("openextract._extract._build_agent", side_effect=build)
    mocker.patch("openextract._extract._run_extraction", side_effect=run_sync)
    mocker.patch("openextract._extract._run_extraction_async", side_effect=run_async)
    mocker.patch("openextract._swarm._run_extraction_async", side_effect=run_async)
    return built


SOLO = define_agent("Solo", model="test:a", instructions="focus", output_schema=Person)
GROUP = define_agent(
    "Group",
    output_schema=Person,
    subagents=[
        define_agent("Names", model="test:a"),
        define_agent("Ages", model="test:b"),
    ],
)


class TestResolveAgentCall:
    def test_a_schema_call_is_left_alone(self):
        assert _resolve_agent_call(Person, "test:a", b"doc") == (Person, "test:a", b"doc")

    def test_an_agent_call_supplies_the_schema_and_shifts_the_input(self):
        assert _resolve_agent_call(SOLO, b"doc", None) == (Person, SOLO, b"doc")

    def test_an_agent_plus_a_separate_model_is_rejected(self):
        with pytest.raises(ValueError, match="takes no separate model"):
            _resolve_agent_call(SOLO, "test:a", b"doc")

    def test_a_missing_input_is_reported(self):
        with pytest.raises(ValueError, match="input_file is required"):
            _resolve_agent_call(Person, "test:a", None)

    def test_an_agent_without_an_output_schema_is_reported(self):
        agent = define_agent("Solo", model="test:a")
        with pytest.raises(ValueError, match="missing output_schema"):
            _resolve_agent_call(agent, b"doc", None)


class TestPlanAgent:
    def test_a_plain_model_is_unchanged(self):
        assert _plan_agent("test:a", "focus", "direct") == ("test:a", "focus", "direct", False)

    def test_a_single_local_agent_becomes_a_one_shot_call(self):
        assert _plan_agent(SOLO, None, "direct") == ("test:a", "focus", "direct", False)

    def test_call_site_values_fill_gaps_the_agent_left(self):
        agent = define_agent("Solo", model="test:a")
        assert _plan_agent(agent, "from caller", "search") == (
            "test:a",
            "from caller",
            "search",
            False,
        )

    def test_a_group_agent_defers_to_the_swarm(self):
        assert _plan_agent(GROUP, None, "direct") == (GROUP, None, "direct", True)

    def test_a_remote_agent_defers_to_the_swarm(self):
        remote = define_remote_agent("https://agents.example.com", "Remote")
        assert _plan_agent(remote, None, "direct") == (remote, None, "direct", True)


class TestResolveOneshot:
    def test_a_solo_agent_becomes_a_local_oneshot(self):
        assert _resolve_oneshot(SOLO, b"doc", None, None, "direct") == (
            Person,
            "test:a",
            b"doc",
            "focus",
            "direct",
            False,
        )

    def test_a_group_agent_is_deferred_to_the_swarm(self):
        assert _resolve_oneshot(GROUP, b"doc", None, None, "direct") == (
            Person,
            GROUP,
            b"doc",
            None,
            "direct",
            True,
        )


class TestExtractWithAgents:
    def test_a_single_agent_runs_as_one_shot(self, mocker):
        built = install_agents(mocker, {"test:a": {"name": "Ada"}})
        assert extract(SOLO, b"doc", media_type="text/plain") == Person(name="Ada")
        assert [agent.instructions for agent in built] == ["focus"]

    def test_the_schema_may_still_be_named_explicitly(self, mocker):
        install_agents(mocker, {"test:a": {"name": "Ada"}})
        assert extract(Person, SOLO, b"doc", media_type="text/plain") == Person(name="Ada")

    def test_a_group_agent_merges_its_subagents(self, mocker):
        install_agents(mocker, {"test:a": {"name": "Ada"}, "test:b": {"age": 36}})
        assert extract(GROUP, b"doc", media_type="text/plain") == Person(name="Ada", age=36)

    def test_usage_comes_back_for_a_single_agent(self, mocker):
        install_agents(mocker, {"test:a": {"name": "Ada"}})
        output, usage = extract_with_usage(SOLO, b"doc", media_type="text/plain")
        assert output == Person(name="Ada")
        assert usage.total_tokens == 5

    def test_usage_is_summed_across_a_group(self, mocker):
        install_agents(mocker, {"test:a": {"name": "Ada"}, "test:b": {"age": 36}})
        output, usage = extract_with_usage(GROUP, b"doc", media_type="text/plain")
        assert output == Person(name="Ada", age=36)
        assert usage.total_tokens == 10

    async def test_async_single_agent(self, mocker):
        install_agents(mocker, {"test:a": {"name": "Ada"}})
        assert await extract_async(SOLO, b"doc", media_type="text/plain") == Person(name="Ada")

    async def test_async_group_agent(self, mocker):
        install_agents(mocker, {"test:a": {"name": "Ada"}, "test:b": {"age": 36}})
        result = await extract_async(GROUP, b"doc", media_type="text/plain")
        assert result == Person(name="Ada", age=36)

    async def test_async_usage_for_a_single_agent(self, mocker):
        install_agents(mocker, {"test:a": {"name": "Ada"}})
        output, usage = await extract_with_usage_async(SOLO, b"doc", media_type="text/plain")
        assert (output, usage.total_tokens) == (Person(name="Ada"), 5)

    async def test_async_usage_for_a_group(self, mocker):
        install_agents(mocker, {"test:a": {"name": "Ada"}, "test:b": {"age": 36}})
        output, usage = await extract_with_usage_async(GROUP, b"doc", media_type="text/plain")
        assert (output, usage.total_tokens) == (Person(name="Ada", age=36), 10)

    def test_a_missing_input_is_reported_by_every_entry_point(self):
        with pytest.raises(ValueError, match="input_file is required"):
            extract(Person, "test:a")
