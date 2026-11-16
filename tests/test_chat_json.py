"""Chat Completions JSON Schema output: a strict schema, a finished JSON message, validation."""

import json

import pytest
from pydantic import BaseModel, Field, ValidationError

from openextract import SchemaValidationError, extract
from tests.helpers import configured, output


class Item(BaseModel):
    description: str
    quantity: int


class Invoice(BaseModel):
    total: float
    items: list[Item] = Field(min_length=1)


INVOICE = {"total": 42.5, "items": [{"description": "Notebook", "quantity": 2}]}
INVOICE_JSON = json.dumps(INVOICE)


def json_model(name="openai-chat:test", **options):
    """A Chat Completions model that asks for JSON Schema output instead of a tool call."""
    return configured(name, output_mode="json_schema", **options)


@pytest.mark.parametrize(
    "name", ["openai-chat:test", "openrouter:google/gemini-2.5-flash", "custom:test"]
)
async def test_requests_json_schema_output_with_a_strict_schema(run, provider, name):
    provider.reply(output("chat-json", INVOICE))
    original = Invoice.model_json_schema()
    model = json_model(name, api="chat", base_url="https://example.com/v1")
    result = await run(Invoice, model, b"Invoice", "Include every item.")
    assert result == Invoice.model_validate(INVOICE)
    request, body = provider.requests[0], provider.bodies[0]
    assert request.url == "https://example.com/v1/chat/completions"
    assert "tools" not in body and "tool_choice" not in body
    assert body["messages"][0] == {"role": "system", "content": "Include every item."}
    assert body["response_format"]["type"] == "json_schema"
    json_schema = body["response_format"]["json_schema"]
    assert json_schema["name"] == "extract"
    assert json_schema["strict"] is True
    assert json_schema["schema"]["additionalProperties"] is False
    assert json_schema["schema"]["$defs"]["Item"]["additionalProperties"] is False
    assert json_schema["schema"]["properties"]["items"]["minItems"] == 1
    assert Invoice.model_json_schema() == original


@pytest.mark.parametrize(
    "finish_reason,message",
    [
        ("length", {"content": INVOICE_JSON}),
        ("content_filter", {"content": INVOICE_JSON}),
        ("tool_calls", {"content": INVOICE_JSON}),
        ("stop", {"content": INVOICE_JSON, "refusal": "refused"}),
        ("stop", {"content": INVOICE_JSON, "tool_calls": [{"function": {"name": "extract"}}]}),
        ("stop", {"content": None}),
        ("stop", {"content": INVOICE}),
        ("stop", {"content": ""}),
        ("stop", {"content": "not JSON"}),
    ],
)
def test_flawed_response_raises_schema_validation_error(provider, finish_reason, message):
    provider.reply({"choices": [{"finish_reason": finish_reason, "message": message}]})
    with pytest.raises(SchemaValidationError):
        extract(Invoice, json_model(), b"Invoice")


async def test_validation_retry_repeats_the_schema_with_feedback(run, provider):
    provider.reply(output("chat-json", INVOICE | {"items": []}), output("chat-json", INVOICE))
    result = await run(Invoice, json_model(validation_retries=1), b"Invoice")
    assert result.total == 42.5
    first, retry = provider.bodies
    assert first["response_format"] == retry["response_format"]
    assert "items" in retry["messages"][0]["content"]
    assert "tools" not in first and "tools" not in retry


def test_schema_invalid_output_keeps_the_pydantic_error(provider):
    provider.reply(output("chat-json", {"total": 42, "items": []}))
    with pytest.raises(SchemaValidationError) as error:
        extract(Invoice, json_model(), b"Invoice")
    assert isinstance(error.value.__cause__, ValidationError)
