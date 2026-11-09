"""Configured defaults reach requests; canned responses do not measure model quality."""

import asyncio
import re
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.settings import ModelSettings

from openextract import extract, extract_async

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_configured_model_defaults(use_async, monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    seen = []

    class Invoice(BaseModel):
        total: Decimal

    def respond(messages, info):
        seen.append(info.model_settings)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"total": "120.00"})])

    model = FunctionModel(respond, settings=ModelSettings(max_tokens=1024))
    source = b"Total due: 120.00 USD"
    if use_async:
        invoice = asyncio.run(extract_async(Invoice, model, source))
    else:
        invoice = extract(Invoice, model, source)
    assert type(invoice) is Invoice
    assert invoice.total == Decimal("120.00")
    assert seen == [{"max_tokens": 1024}]
    assert model.settings == {"max_tokens": 1024}


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_published_anthropic_settings(use_async, monkeypatch, capsys):
    from anthropic.resources.beta.messages.messages import AsyncMessages
    from anthropic.types.beta import BetaMessage

    # Exercise the real provider/model constructors; intercept the SDK request
    # boundary rather than replacing OpenExtract or its Agent.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-key")
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    requests = []

    async def respond(**kwargs):
        requests.append(kwargs)
        return BetaMessage.model_validate(
            {
                "id": "msg_offline",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_offline",
                        "name": kwargs["tools"][0]["name"],
                        "input": {
                            "vendor": "Acme Paper Co.",
                            "total": "120.00",
                            "currency": "USD",
                        },
                    }
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 20},
            }
        )

    monkeypatch.setattr(AsyncMessages, "create", AsyncMock(side_effect=respond))
    # Fail closed if the adapter changes to another SDK request path.
    import httpx

    async def no_network(*args, **kwargs):
        raise AssertionError("Unexpected network request")

    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    try:
        import httpx2
    except ImportError:
        pass
    else:
        monkeypatch.setattr(httpx2.AsyncClient, "send", no_network)

    blocks = re.findall(
        r"```python\n(.*?)```", (ROOT / "examples/configured_models.md").read_text(), re.S
    )
    assert len(blocks) == 2
    code = blocks[0]
    if use_async:
        code = code.split("invoice = extract(")[0] + blocks[1]
    namespace = {}
    try:
        exec(compile(code, "configured_models.md", "exec"), namespace)
    finally:
        if "model" in namespace:
            asyncio.run(namespace["model"].client.close())
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
