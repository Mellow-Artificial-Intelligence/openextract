"""Anthropic native JSON output: an adapted schema, one finished JSON block, local validation."""

from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from openextract import SchemaValidationError, extract
from openextract._schemas import anthropic_json
from tests.helpers import configured, output


class Item(BaseModel):
    name: str = Field(min_length=2, max_length=20, pattern=r"^\w+$")
    quantity: int = Field(ge=1, le=10, multiple_of=1)


class Invoice(BaseModel):
    items: list[Item] = Field(min_length=2, max_length=5)
    note: str | None = None


class Recursive(BaseModel):
    children: list["Recursive"]


class Map(BaseModel):
    values: dict[str, int]


VALUE = {"items": [{"name": "Pen", "quantity": 2}, {"name": "Pad", "quantity": 1}]}


def json_model(**options):
    """An Anthropic model that asks for native JSON output instead of a tool call."""
    return configured("anthropic:test", output_mode="json_schema", **options)


@pytest.mark.parametrize("source", [b"Invoice", b"%PDF-mock", b"\x89PNGmock"])
async def test_requests_native_json_with_an_adapted_schema(run, provider, source):
    provider.reply(output("anthropic-json", VALUE))
    original = Invoice.model_json_schema()
    model = json_model(settings={"max_tokens": 1234})
    assert await run(Invoice, model, source, "Read all items") == Invoice.model_validate(VALUE)
    request, body = provider.requests[0], provider.bodies[0]
    assert request.url.path == "/v1/messages"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert "anthropic-beta" not in request.headers
    assert "tools" not in body and "tool_choice" not in body
    assert body["max_tokens"] == 1234 and body["system"] == "Read all items"
    assert body["output_config"]["format"]["type"] == "json_schema"
    schema = body["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["items"]
    assert schema["properties"]["note"]["default"] is None
    assert "minItems" not in schema["properties"]["items"]
    item = schema["$defs"]["Item"]
    quantity, name = item["properties"]["quantity"], item["properties"]["name"]
    assert item["additionalProperties"] is False
    assert "minimum" not in quantity and "minimum" in quantity["description"]
    assert "maxLength" not in name and "maxLength" in name["description"]
    assert Invoice.model_json_schema() == original


@pytest.mark.parametrize(
    "flaw",
    [
        {"stop_reason": "refusal"},
        {"stop_reason": "max_tokens"},
        {"stop_reason": "stop_sequence"},
        {"stop_reason": "tool_use"},
        {"content": []},
        {"content": [{"type": "text", "text": None}]},
        {"content": [{"type": "text", "text": ""}]},
        {"content": [{"type": "text", "text": "{"}]},
        {"content": [{"type": "tool_use"}]},
        {"content": [{"type": "text", "text": "{}"}, {"type": "text", "text": "{}"}]},
        {"content": [{"type": "thinking", "thinking": "internal"}]},
    ],
)
async def test_flawed_response_raises_schema_validation_error(run, provider, flaw):
    provider.reply(output("anthropic-json", VALUE) | flaw)
    with pytest.raises(SchemaValidationError):
        await run(Invoice, json_model(), b"Invoice")


async def test_validation_retry_repeats_the_schema_with_feedback(run, provider):
    provider.reply(output("anthropic-json", {"items": []}), output("anthropic-json", VALUE))
    result = await run(Invoice, json_model(validation_retries=1), b"Invoice")
    assert result == Invoice.model_validate(VALUE)
    first, retry = provider.bodies
    assert first["output_config"] == retry["output_config"]
    assert "items" in retry["system"]


def test_constraints_anthropic_cannot_enforce_are_validated_locally(provider):
    provider.reply(output("anthropic-json", {"items": [{"name": "!", "quantity": 0}]}))
    with pytest.raises(SchemaValidationError) as error:
        extract(Invoice, json_model(), b"Invoice")
    cause = error.value.__cause__
    assert isinstance(cause, ValidationError)
    violations = {detail["type"] for detail in cause.errors()}
    assert violations >= {"string_too_short", "greater_than_equal"}


@pytest.mark.parametrize("schema", [Recursive, Map])
def test_unsupported_schema_fails_before_any_request(provider, schema):
    with pytest.raises(ValueError, match="Anthropic JSON output"):
        extract(schema, json_model(), b"Invoice")
    assert not provider.requests


@pytest.mark.parametrize("keyword", ["output_config", "output_format"])
@pytest.mark.parametrize("mode", [None, "tool", "json_schema"])
def test_settings_cannot_override_output_fields(keyword, mode):
    model = configured("anthropic:test", output_mode=mode, settings={keyword: {}})
    with pytest.raises(ValueError, match="cannot override"):
        extract(Invoice, model, b"Invoice")


@pytest.mark.parametrize("mode", [None, "tool"])
def test_forced_tool_call_remains_the_default(provider, mode):
    provider.reply(output("anthropic", VALUE))
    model = configured("anthropic:test", output_mode=mode)
    assert extract(Invoice, model, b"Invoice") == Invoice.model_validate(VALUE)
    body = provider.bodies[0]
    assert "output_config" not in body
    assert body["tools"][0]["input_schema"] == Invoice.model_json_schema()
    assert body["tool_choice"] == {"type": "tool", "name": "extract"}


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "https://example.com/schema"},
        {"allOf": []},
        {"prefixItems": []},
        {"patternProperties": {}},
        {"enum": [{}]},
    ],
)
def test_adapter_rejects_schemas_anthropic_cannot_express(schema):
    with pytest.raises(ValueError, match="Anthropic JSON output"):
        anthropic_json(schema)


def test_adapter_keeps_only_the_keywords_anthropic_accepts():
    assert anthropic_json({"type": "string", "format": "date"})["format"] == "date"
    assert "format" not in anthropic_json({"type": "string", "format": "password"})
    assert anthropic_json({"type": "array", "minItems": 1})["minItems"] == 1


def test_adapter_expresses_a_tagged_union_without_a_discriminator():
    class Cat(BaseModel):
        kind: Literal["cat"]

    class Dog(BaseModel):
        kind: Literal["dog"]

    class Pet(BaseModel):
        pet: Annotated[Cat | Dog, Field(discriminator="kind")]

    pet = anthropic_json(Pet.model_json_schema())["properties"]["pet"]
    assert "anyOf" in pet and "discriminator" not in pet
