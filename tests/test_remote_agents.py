"""Tests for openextract._remote and remote agents inside a swarm."""

import base64

import httpx
import pytest
from pydantic import BaseModel

from openextract import (
    ExtractionResult,
    RemoteAgentError,
    SchemaValidationError,
    UrlFetchError,
    define_remote_agent,
    extract_swarm,
    extract_swarm_with_results,
)
from openextract._remote import _usage_from_payload, run_remote_extraction
from openextract._styles import ExtractionStyle
from openextract.auth import bearer


class FakeUsage:
    input_tokens = 1
    output_tokens = 1
    total_tokens = 2


class Person(BaseModel):
    name: str | None = None
    age: int | None = None


def mock_transport(mocker, handler):
    """Serve every remote agent request from ``handler`` instead of the network."""
    requests: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    original = httpx.AsyncClient

    def build(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(capture)
        return original(*args, **kwargs)

    mocker.patch("openextract._remote.httpx.AsyncClient", side_effect=build)
    return requests


def json_handler(payload, status_code: int = 200):
    return lambda request: httpx.Response(status_code, json=payload)


async def run(agent, mocker, handler, **kwargs):
    mock_transport(mocker, handler)
    return await run_remote_extraction(
        Person,
        agent,
        b"doc",
        "text/plain",
        instructions=kwargs.pop("instructions", None),
        style=kwargs.pop("style", ExtractionStyle.DIRECT),
        max_retries=kwargs.pop("max_retries", 0),
        retry_backoff=0,
        retry_max_backoff=0,
    )


AGENT = define_remote_agent("https://agents.example.com", "Remote invoices")


class TestRequest:
    async def test_posts_the_schema_media_and_style(self, mocker):
        requests = mock_transport(mocker, json_handler({"output": {"name": "Ada"}}))
        await run_remote_extraction(
            Person,
            AGENT,
            b"doc",
            "text/plain",
            instructions="totals",
            style=ExtractionStyle.SEARCH,
            max_retries=0,
            retry_backoff=0,
            retry_max_backoff=0,
        )
        request = requests[0]
        assert str(request.url) == "https://agents.example.com/extract"
        body = __import__("json").loads(request.content)
        assert body["schema"] == Person.model_json_schema()
        assert body["input"]["data"] == base64.b64encode(b"doc").decode()
        assert body["input"]["mediaType"] == "text/plain"
        assert body["instructions"] == "totals"
        assert body["style"] == "search"
        assert request.headers["content-type"] == "application/json"

    async def test_a_trailing_slash_does_not_double_up(self, mocker):
        agent = define_remote_agent("https://agents.example.com/", "Remote", path="run")
        requests = await_requests = mock_transport(mocker, json_handler({"output": {}}))
        await run(agent, mocker, json_handler({"output": {}}))
        assert str(await_requests[0].url) == "https://agents.example.com/run"
        assert requests is await_requests

    async def test_lazy_urls_and_auth_are_resolved_per_request(self, mocker):
        agent = define_remote_agent(
            lambda: "https://agents.example.com",
            "Remote",
            auth=bearer("abc"),
            headers={"x-run": "1"},
        )
        requests = mock_transport(mocker, json_handler({"output": {"name": "Ada"}}))
        await run(agent, mocker, json_handler({"output": {"name": "Ada"}}))
        assert requests[0].headers["authorization"] == "Bearer abc"
        assert requests[0].headers["x-run"] == "1"

    async def test_callable_headers_are_resolved(self, mocker):
        agent = define_remote_agent(
            "https://agents.example.com", "Remote", headers=lambda: {"x-run": "2"}
        )
        requests = mock_transport(mocker, json_handler({"output": {}}))
        await run(agent, mocker, json_handler({"output": {}}))
        assert requests[0].headers["x-run"] == "2"

    async def test_a_url_provider_must_return_a_string(self, mocker):
        agent = define_remote_agent(lambda: "", "Remote")
        with pytest.raises(ValueError, match="must resolve to a non-empty string"):
            await run(agent, mocker, json_handler({"output": {}}))

    async def test_non_http_urls_are_rejected(self, mocker):
        agent = define_remote_agent("file:///etc/passwd", "Remote")
        with pytest.raises(ValueError, match="must be http or https"):
            await run(agent, mocker, json_handler({"output": {}}))

    async def test_private_hosts_are_blocked_by_default(self, mocker, monkeypatch):
        import socket

        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))],
        )
        agent = define_remote_agent("http://localhost:8000", "Remote")
        with pytest.raises(UrlFetchError, match="non-public host"):
            await run(agent, mocker, json_handler({"output": {}}))

    async def test_private_hosts_are_reachable_when_explicitly_allowed(self, mocker, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_ALLOW_PRIVATE_URLS", "1")
        agent = define_remote_agent("http://localhost:8000", "Remote")
        output, _, _ = await run(agent, mocker, json_handler({"output": {"name": "Ada"}}))
        assert output == Person(name="Ada")


class TestResponse:
    async def test_the_output_key_is_validated(self, mocker):
        output, usage, attempts = await run(
            AGENT, mocker, json_handler({"output": {"name": "Ada"}, "usage": {"totalTokens": 7}})
        )
        assert output == Person(name="Ada")
        assert usage.total_tokens == 7
        assert attempts == 1

    async def test_a_bare_body_is_treated_as_the_output(self, mocker):
        output, _, _ = await run(AGENT, mocker, json_handler({"name": "Ada"}))
        assert output == Person(name="Ada")

    async def test_output_that_does_not_match_the_schema_fails(self, mocker):
        with pytest.raises(SchemaValidationError, match=r"age: expected int, got str"):
            await run(AGENT, mocker, json_handler({"output": {"age": "old"}}))

    async def test_non_json_bodies_are_reported(self, mocker):
        handler = lambda request: httpx.Response(200, content=b"<html>")  # noqa: E731
        with pytest.raises(RemoteAgentError, match="non-JSON") as info:
            await run(AGENT, mocker, handler)
        assert info.value.url == "https://agents.example.com/extract"

    async def test_an_empty_body_is_reported(self, mocker):
        handler = lambda request: httpx.Response(200, content=b"")  # noqa: E731
        with pytest.raises(RemoteAgentError, match="empty response"):
            await run(AGENT, mocker, handler)

    async def test_a_non_object_body_is_reported(self, mocker):
        with pytest.raises(RemoteAgentError, match="empty response"):
            await run(AGENT, mocker, json_handler([1, 2]))

    async def test_an_error_field_on_a_success_is_reported(self, mocker):
        with pytest.raises(RemoteAgentError, match="agent said no") as info:
            await run(AGENT, mocker, json_handler({"error": "agent said no"}))
        assert info.value.retryable is False

    async def test_an_error_status_uses_the_reported_message(self, mocker):
        with pytest.raises(RemoteAgentError, match="schema unsupported") as info:
            await run(AGENT, mocker, json_handler({"error": "schema unsupported"}, 400))
        assert info.value.status_code == 400
        assert info.value.retryable is False

    async def test_an_error_status_without_a_message_falls_back(self, mocker):
        with pytest.raises(RemoteAgentError, match="failed with status 503") as info:
            await run(AGENT, mocker, json_handler({}, 503))
        assert info.value.retryable is True

    async def test_transport_failures_are_retryable(self, mocker):
        def handler(request):
            raise httpx.ConnectError("no route", request=request)

        with pytest.raises(RemoteAgentError, match="request failed") as info:
            await run(AGENT, mocker, handler)
        assert info.value.retryable is True

    async def test_retryable_failures_are_retried(self, mocker):
        attempts = {"count": 0}

        def handler(request):
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(503, json={})
            return httpx.Response(200, json={"output": {"name": "Ada"}})

        output, _, count = await run(AGENT, mocker, handler, max_retries=1)
        assert output == Person(name="Ada")
        assert count == 2


class TestUsagePayload:
    def test_camel_case_is_read(self):
        usage = _usage_from_payload({"inputTokens": 1, "outputTokens": 2, "totalTokens": 3})
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (1, 2, 3)

    def test_snake_case_is_read(self):
        usage = _usage_from_payload({"input_tokens": 4, "output_tokens": 5, "total_tokens": 9})
        assert usage.total_tokens == 9

    def test_missing_usage_is_zero(self):
        assert _usage_from_payload(None).total_tokens == 0

    def test_non_integer_values_are_ignored(self):
        assert _usage_from_payload({"totalTokens": "many", "inputTokens": True}).total_tokens == 0


class TestRemoteAgentsInASwarm:
    def test_a_remote_agent_runs_alongside_a_local_one(self, mocker):
        mock_transport(mocker, json_handler({"output": {"age": 36}}))
        mocker.patch(
            "openextract._swarm._build_agent",
            side_effect=lambda schema, model, instructions, **kw: model,
        )

        async def local_run(agent, inputs):
            class Result:
                output = Person(name="Ada")

                @property
                def usage(self):
                    return FakeUsage()

            return Result()

        mocker.patch("openextract._swarm._run_extraction_async", side_effect=local_run)
        result = extract_swarm(Person, ["test:a", AGENT], b"doc", media_type="text/plain")
        assert result == Person(name="Ada", age=36)

    def test_a_remote_agent_is_labelled_by_url(self, mocker):
        mock_transport(mocker, json_handler({"output": {"name": "Ada"}}))
        swarm = extract_swarm_with_results(Person, AGENT, b"doc", media_type="text/plain")
        assert isinstance(swarm.agents[0], ExtractionResult)
        assert swarm.agents[0].model == "https://agents.example.com"

    def test_a_lazy_url_agent_is_labelled_by_description(self, mocker):
        agent = define_remote_agent(lambda: "https://agents.example.com", "Remote invoices")
        mock_transport(mocker, json_handler({"output": {"name": "Ada"}}))
        swarm = extract_swarm_with_results(Person, agent, b"doc", media_type="text/plain")
        assert swarm.agents[0].model == "Remote invoices"
