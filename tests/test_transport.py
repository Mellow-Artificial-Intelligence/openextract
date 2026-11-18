"""Transport: bounded retries, Retry-After, timeouts, and failures that never echo secrets."""

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from openextract import Model, ModelError, _transport, extract, extract_async
from tests.helpers import Invoice, configured, output


def failing(failure):
    """A request handler that raises ``failure`` with a credential in its message."""

    def handler(request):
        raise failure("Illegal header value: Bearer private-key", request=request)

    return handler


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
async def test_transient_status_retries_the_same_request(run, provider, monkeypatch, status):
    delays = []
    monkeypatch.setattr(_transport.time, "sleep", delays.append)
    monkeypatch.setattr(_transport.asyncio, "sleep", AsyncMock(side_effect=delays.append))
    provider.reply(lambda request: httpx.Response(status, headers={"Retry-After": "2"}), output())
    assert (await run(Invoice, configured(max_retries=1), b"text")).total == 42
    first, second = provider.requests
    assert first.content == second.content
    assert delays == [2.0]


async def test_transport_error_is_retried(run, provider):
    provider.reply(failing(httpx.ConnectError), output())
    assert (await run(Invoice, configured(max_retries=1), b"text")).total == 42
    assert len(provider.requests) == 2


@pytest.mark.parametrize(
    "failure,max_retries,requests,next_step",
    [
        (httpx.ConnectError, 1, 2, "Check Model(base_url=...), network connectivity"),
        (httpx.ReadTimeout, 1, 2, "Check provider availability or increase Model(timeout=...)"),
        (httpx.LocalProtocolError, 0, 1, "Check Model(headers=...)"),
        # A response that cannot be decoded is not transient, so it is never retried.
        (httpx.DecodingError, 2, 1, "Check the provider or proxy's response encoding"),
    ],
)
async def test_request_failure_reports_attempts_and_a_next_step_without_secrets(
    run, provider, failure, max_retries, requests, next_step
):
    provider.reply(failing(failure))
    with pytest.raises(ModelError, match=failure.__name__) as error:
        await run(Invoice, configured(api_key="private-key", max_retries=max_retries), b"text")
    attempts = "1 HTTP attempt" if requests == 1 else f"{requests} HTTP attempts"
    assert f"after {attempts}.\nNext step: {next_step}" in str(error.value)
    assert "private" not in str(error.value)
    assert isinstance(error.value.__cause__, failure)
    assert len(provider.requests) == requests


@pytest.mark.parametrize("status,attempts", [(400, 1), (401, 1), (403, 1), (429, 3), (503, 3)])
async def test_default_budget_retries_only_transient_statuses(run, provider, status, attempts):
    provider.reply(lambda request: httpx.Response(status))
    with pytest.raises(ModelError, match=f"HTTP {status}"):
        await run(Invoice, Model("openai:test", api_key="key"), b"text")
    assert len(provider.requests) == attempts


async def test_redirect_is_not_followed_even_by_a_client_that_follows_redirects(provider):
    def redirect(request):
        return httpx.Response(307, headers={"Location": "https://untrusted.example"})

    provider.reply(redirect)
    model = Model("openai:test", api_key="key")
    with httpx.Client(follow_redirects=True) as client, pytest.raises(ModelError, match="HTTP 307"):
        extract(Invoice, model, b"text", client=client)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        with pytest.raises(ModelError, match="HTTP 307"):
            await extract_async(Invoice, model, b"text", client=client)
    assert [request.url.host for request in provider.requests] == ["api.openai.com"] * 2


@pytest.mark.parametrize(
    "retry_after,delay",
    {"0": 0, "2.5": 2.5, "999": 30, "-1": 0, "nan": 0.5, "inf": 0.5, "not-a-date": 0.5}.items(),
)
def test_retry_after_is_bounded_and_ignored_when_invalid(retry_after, delay):
    assert _transport._delay(0, httpx.Response(429, headers={"Retry-After": retry_after})) == delay


def test_retry_after_http_date_and_backoff_cap(monkeypatch):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(_transport, "datetime", SimpleNamespace(now=lambda timezone: now))
    retry_after = format_datetime(now + timedelta(seconds=10))
    assert _transport._delay(0, httpx.Response(429, headers={"Retry-After": retry_after})) == 10
    assert _transport._delay(10, None) == 8


def test_model_timeout_is_applied_to_a_caller_owned_client(provider):
    with httpx.Client(timeout=100) as client:
        extract(Invoice, configured(timeout=12), b"text", client=client)
    assert set(provider.requests[0].extensions["timeout"].values()) == {12}
