import json

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

from openextract import Model, SchemaValidationError, extract, extract_async
from tests.helpers import configured


class Item(BaseModel):
    description: str
    quantity: int


class Invoice(BaseModel):
    total: float
    items: list[Item] = Field(min_length=1)


INVOICE = {"total": 42.5, "items": [{"description": "Notebook", "quantity": 2}]}


def response(value=INVOICE):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "name", ["openai-chat:test", "openrouter:google/gemini-2.5-flash", "custom:test"]
)
async def test_json_schema_request_and_response_contract(asynchronous, name):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=response())

    original = Invoice.model_json_schema()
    model = configured(
        name, api="chat", base_url="https://example.com/v1", output_mode="json_schema"
    )
    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(
                Invoice, model, b"Invoice", "Include every item.", client=client
            )
    else:
        with httpx.Client(transport=transport) as client:
            result = extract(Invoice, model, b"Invoice", "Include every item.", client=client)
    assert result == Invoice.model_validate(INVOICE)
    request = json.loads(seen[0].content)
    assert seen[0].url == "https://example.com/v1/chat/completions"
    assert "tools" not in request and "tool_choice" not in request
    assert request["messages"][0] == {"role": "system", "content": "Include every item."}
    assert request["response_format"]["type"] == "json_schema"
    format = request["response_format"]["json_schema"]
    assert format["name"] == "extract"
    assert format["strict"] is True
    assert format["schema"]["additionalProperties"] is False
    assert format["schema"]["$defs"]["Item"]["additionalProperties"] is False
    assert format["schema"]["properties"]["items"]["minItems"] == 1
    assert Invoice.model_json_schema() == original


@pytest.mark.parametrize(
    "finish_reason,message",
    [
        ("length", {"content": json.dumps(INVOICE)}),
        ("content_filter", {"content": json.dumps(INVOICE)}),
        ("tool_calls", {"content": json.dumps(INVOICE)}),
        ("stop", {"content": json.dumps(INVOICE), "refusal": "refused"}),
        (
            "stop",
            {"content": json.dumps(INVOICE), "tool_calls": [{"function": {"name": "extract"}}]},
        ),
        ("stop", {"content": None}),
        ("stop", {"content": INVOICE}),
        ("stop", {"content": ""}),
        ("stop", {"content": "not JSON"}),
    ],
)
def test_json_schema_rejects_invalid_output(finish_reason, message):
    payload = {"choices": [{"finish_reason": finish_reason, "message": message}]}
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ) as client,
        pytest.raises(SchemaValidationError),
    ):
        extract(
            Invoice,
            configured("openai-chat:test", output_mode="json_schema"),
            b"Invoice",
            client=client,
        )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_json_schema_validates_output_and_retries(asynchronous):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        value = {"total": 42.5, "items": []} if len(seen) == 1 else INVOICE
        return httpx.Response(200, json=response(value))

    model = Model(
        "openai-chat:test", api_key="test", output_mode="json_schema", validation_retries=1
    )
    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(Invoice, model, b"Invoice", client=client)
    else:
        with httpx.Client(transport=transport) as client:
            result = extract(Invoice, model, b"Invoice", client=client)
    assert result.total == 42.5
    assert len(seen) == 2
    assert seen[0]["response_format"] == seen[1]["response_format"]
    assert "items" in seen[1]["messages"][0]["content"]
    assert all("tools" not in body for body in seen)


def test_json_schema_validation_failure_keeps_pydantic_error():
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=response({"total": 42, "items": []}))
            )
        ) as client,
        pytest.raises(SchemaValidationError) as error,
    ):
        extract(
            Invoice,
            configured("openai-chat:test", output_mode="json_schema"),
            b"Invoice",
            client=client,
        )
    assert isinstance(error.value.__cause__, ValidationError)
