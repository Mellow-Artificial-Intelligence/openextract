"""Offline by default: a mock provider, isolated credentials, and no retry delays."""

import json
from functools import partial

import httpx
import pytest

from openextract import _transport, extract, extract_async, models
from tests.helpers import output


def pytest_addoption(parser):
    parser.addoption(
        "--live", action="store_true", help="Run tests marked live against real provider APIs."
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    disabled = pytest.mark.skip(reason="Live API calls are disabled; pass --live to enable them.")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(disabled)


@pytest.fixture(autouse=True)
def offline(request, monkeypatch):
    """Outside live tests, block real HTTP, hide provider settings, and skip retry delays."""
    if request.node.get_closest_marker("live"):
        return

    def blocked(self, request):
        pytest.fail("Real HTTP requests require a live-marked test and --live.")

    async def blocked_async(self, request):
        blocked(self, request)

    async def no_delay(seconds):
        pass

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)
    monkeypatch.setattr(_transport.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(_transport.asyncio, "sleep", no_delay)
    variables = {"GEMINI_API_KEY", "AWS_REGION", "AWS_DEFAULT_REGION"}
    for _, _, key_variable, url_variable in models._PROVIDERS.values():
        variables |= {key_variable, url_variable} - {None}
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)


class Provider:
    """A mock model provider that records requests and replays queued replies."""

    def __init__(self):
        self.requests = []
        self.reply()

    def reply(self, *replies, status=200):
        """Queue JSON payloads or request handlers; the last one answers every later request."""
        self.replies = replies or (output(),)
        self.status = status

    def __call__(self, request):
        self.requests.append(request)
        reply = self.replies[min(len(self.requests), len(self.replies)) - 1]
        return reply(request) if callable(reply) else httpx.Response(self.status, json=reply)

    @property
    def bodies(self):
        """The decoded JSON body of every model request."""
        return [json.loads(request.content) for request in self.requests if request.content]


@pytest.fixture
def provider(monkeypatch):
    """Route every HTTPX request, including source URL downloads, to a mock provider."""
    provider = Provider()
    transport = httpx.MockTransport(provider)

    def get(url, **options):
        with httpx.Client(**options) as client:
            return client.get(url)

    for client in ("Client", "AsyncClient"):
        monkeypatch.setattr(httpx, client, partial(getattr(httpx, client), transport=transport))
    monkeypatch.setattr(httpx, "get", get)
    return provider


@pytest.fixture(params=["sync", "async"])
def run(request, provider):
    """``await run(...)`` calls ``extract`` or ``extract_async``, so one test covers both."""

    async def call(*args, **kwargs):
        if request.param == "sync":
            return extract(*args, **kwargs)
        return await extract_async(*args, **kwargs)

    return call
