"""Configured defaults reach HTTP requests; canned responses do not measure model quality."""

import asyncio
import json
import re
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from openextract import extract, extract_async
from tests.helpers import configured, output, serve

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_configured_model_defaults(use_async, monkeypatch):
    seen = []

    class Invoice(BaseModel):
        total: Decimal

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=output(value={"total": "120.00"}))

    serve(monkeypatch, respond)
    model = configured(settings={"max_output_tokens": 1024})
    source = b"Total due: 120.00 USD"
    if use_async:
        invoice = asyncio.run(extract_async(Invoice, model, source))
    else:
        invoice = extract(Invoice, model, source)
    assert type(invoice) is Invoice
    assert invoice.total == Decimal("120.00")
    assert len(seen) == 1
    assert seen[0]["max_output_tokens"] == 1024
    assert model.settings == {"max_output_tokens": 1024}


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_published_anthropic_settings(use_async, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-key")
    requests = []

    def respond(request):
        assert request.url == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "offline-test-key"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=output(
                "anthropic", {"vendor": "Acme Paper Co.", "total": "120.00", "currency": "USD"}
            ),
        )

    serve(monkeypatch, respond)
    blocks = re.findall(
        r"```python\n(.*?)```", (ROOT / "examples/configured_models.md").read_text(), re.S
    )
    assert len(blocks) == 2
    code = blocks[0]
    if use_async:
        code = code.split("invoice = extract(")[0] + blocks[1]
    namespace = {}
    exec(compile(code, "configured_models.md", "exec"), namespace)
    invoice = namespace["invoice"]
    assert type(invoice) is namespace["Invoice"]
    assert invoice.total == Decimal("120.00")
    assert invoice.vendor == "Acme Paper Co."
    assert invoice.currency == "USD"
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "claude-sonnet-4-5"
    assert request["max_tokens"] == 1024
    assert namespace["model"].settings == {"max_tokens": 1024}
    assert "Seller: Acme Paper Co." in str(request["messages"])
    assert capsys.readouterr().out == "120.00\n"
