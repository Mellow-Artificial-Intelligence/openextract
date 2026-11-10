from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from openextract import Model, ModelError, _transport, extract, extract_async
from tests.helpers import output
from tests.test_extract import Invoice


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
async def test_transient_http_error_retries_with_same_request(monkeypatch, asynchronous, status):
    seen, delays = [], []

    async def sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(_transport.time, "sleep", delays.append)
    monkeypatch.setattr(_transport.asyncio, "sleep", sleep)

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(status, headers={"Retry-After": "2"})
        return httpx.Response(200, json=output())

    model = Model("openai:test", api_key="key", max_retries=1)
    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(Invoice, model, b"text", client=client)
    else:
        with httpx.Client(transport=transport) as client:
            result = extract(Invoice, model, b"text", client=client)
    assert result.total == 42
    assert len(seen) == 2
    assert seen[0].content == seen[1].content
    assert delays == [2.0]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("recover", [False, True])
async def test_transport_error_retry_and_exhaustion(asynchronous, recover):
    seen = []

    def handler(request):
        seen.append(request)
        if not recover or len(seen) == 1:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json=output())

    model = Model("openai:test", api_key="key", max_retries=1)
    transport = httpx.MockTransport(handler)

    async def run():
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                return await extract_async(Invoice, model, b"text", client=client)
        with httpx.Client(transport=transport) as client:
            return extract(Invoice, model, b"text", client=client)

    if recover:
        assert (await run()).total == 42
    else:
        with pytest.raises(ModelError, match="ConnectError") as error:
            await run()
        assert isinstance(error.value.__cause__, httpx.ConnectError)
        assert "after 2 HTTP attempts" in str(error.value)
        assert "Next step: Check Model(base_url=...), network connectivity" in str(error.value)
    assert len(seen) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "status,attempts", [(400, 1), (401, 1), (403, 1), (429, 3), (503, 3), (307, 1)]
)
async def test_retry_budget_and_permanent_errors(asynchronous, status, attempts):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, headers={"Location": "https://untrusted.example"})

    transport = httpx.MockTransport(handler)
    model = Model("openai:test", api_key="key")
    with pytest.raises(ModelError, match=f"HTTP {status}"):
        if asynchronous:
            async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
                await extract_async(Invoice, model, b"text", client=client)
        else:
            with httpx.Client(transport=transport, follow_redirects=True) as client:
                extract(Invoice, model, b"text", client=client)
    assert len(seen) == attempts
    assert {r.url.host for r in seen} == {"api.openai.com"}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0", 0),
        ("2.5", 2.5),
        ("999", 30),
        ("-1", 0),
        ("nan", 0.5),
        ("inf", 0.5),
        ("not-a-date", 0.5),
    ],
)
def test_retry_after_limits(value, expected):
    assert _transport._delay(0, httpx.Response(429, headers={"Retry-After": value})) == expected


def test_retry_after_http_date(monkeypatch):
    now = datetime(2026, 1, 1, tzinfo=UTC)

    class Clock:
        @staticmethod
        def now(tz):
            return now

    monkeypatch.setattr(_transport, "datetime", Clock)
    response = httpx.Response(
        429, headers={"Retry-After": format_datetime(now + timedelta(seconds=10))}
    )
    assert _transport._delay(0, response) == 10
    assert _transport._delay(10, None) == 8


def test_timeout_is_applied_to_injected_client():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=output())

    with httpx.Client(timeout=100, transport=httpx.MockTransport(handler)) as client:
        extract(Invoice, Model("openai:test", api_key="key", timeout=12), b"text", client=client)
    assert set(seen[0].extensions["timeout"].values()) == {12}


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_response_decoding_failure_does_not_retry(asynchronous):
    seen = []

    def handler(request):
        seen.append(request)
        raise httpx.DecodingError("invalid compressed response", request=request)

    model = Model("openai:test", api_key="key")
    transport = httpx.MockTransport(handler)
    with pytest.raises(ModelError, match="DecodingError"):
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                await extract_async(Invoice, model, b"text", client=client)
        else:
            with httpx.Client(transport=transport) as client:
                extract(Invoice, model, b"text", client=client)
    assert len(seen) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_transport_error_message_does_not_echo_secrets(asynchronous):
    def handler(request):
        raise httpx.LocalProtocolError("Illegal header value: Bearer private-key", request=request)

    transport = httpx.MockTransport(handler)
    model = Model("openai:test", api_key="private-key", max_retries=0)
    with pytest.raises(ModelError, match="LocalProtocolError") as error:
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                await extract_async(Invoice, model, b"text", client=client)
        else:
            with httpx.Client(transport=transport) as client:
                extract(Invoice, model, b"text", client=client)
    assert "private-key" not in str(error.value)
    assert "after 1 HTTP attempt" in str(error.value)
    assert "Next step: Check Model(headers=...)" in str(error.value)
    assert isinstance(error.value.__cause__, httpx.LocalProtocolError)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_timeout_error_explains_how_to_adjust_timeout(asynchronous):
    def handler(request):
        raise httpx.ReadTimeout("private network details", request=request)

    transport = httpx.MockTransport(handler)
    model = Model("openai:test", api_key="private-key", max_retries=1)
    with pytest.raises(ModelError) as error:
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                await extract_async(Invoice, model, b"Invoice", client=client)
        else:
            with httpx.Client(transport=transport) as client:
                extract(Invoice, model, b"Invoice", client=client)
    message = str(error.value)
    assert "ReadTimeout" in message and "after 2 HTTP attempts" in message
    assert "Next step:" in message and "Model(timeout=...)" in message
    assert "private" not in message
    assert isinstance(error.value.__cause__, httpx.ReadTimeout)
