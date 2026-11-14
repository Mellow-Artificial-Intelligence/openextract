import json
from typing import Annotated, Literal

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

from openextract import Model, SchemaValidationError, extract, extract_async
from tests.helpers import configured


class Item(BaseModel):
    name: str = Field(min_length=2, max_length=20, pattern=r"^\w+$")
    quantity: int = Field(ge=1, le=10, multiple_of=1)


class Invoice(BaseModel):
    items: list[Item] = Field(min_length=2, max_length=5)
    note: str | None = None


VALUE = {"items": [{"name": "Pen", "quantity": 2}, {"name": "Pad", "quantity": 1}]}


def response(value=VALUE):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(value)}]}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("source", [b"Invoice", b"%PDF-mock", b"\x89PNGmock"])
async def test_native_json_contract(asynchronous, source):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=response())

    original = Invoice.model_json_schema()
    model = configured("anthropic:test", output_mode="json_schema", settings={"max_tokens": 1234})
    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(Invoice, model, source, "Read all items", client=client)
    else:
        with httpx.Client(transport=transport) as client:
            result = extract(Invoice, model, source, "Read all items", client=client)
    assert result == Invoice.model_validate(VALUE)
    request = seen[0]
    body = json.loads(request.content)
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
    item = schema["$defs"]["Item"]
    assert item["additionalProperties"] is False
    assert "minimum" not in item["properties"]["quantity"]
    assert "minimum" in item["properties"]["quantity"]["description"]
    assert "maxLength" not in item["properties"]["name"]
    assert "maxLength" in item["properties"]["name"]["description"]
    assert "minItems" not in schema["properties"]["items"]
    assert Invoice.model_json_schema() == original


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "stop_sequence", "tool_use"])
async def test_http_200_incomplete_or_refused(asynchronous, stop_reason):
    payload = response()
    payload["stop_reason"] = stop_reason
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    model = configured("anthropic:test", output_mode="json_schema")
    with pytest.raises(SchemaValidationError):
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                await extract_async(Invoice, model, b"Invoice", client=client)
        else:
            with httpx.Client(transport=transport) as client:
                extract(Invoice, model, b"Invoice", client=client)


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "text", "text": None}],
        [{"type": "text", "text": ""}],
        [{"type": "text", "text": "{"}],
        [{"type": "tool_use"}],
        [{"type": "text", "text": "{}"}, {"type": "text", "text": "{}"}],
        [{"type": "thinking", "thinking": "internal"}],
    ],
)
def test_invalid_json_content(content):
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"stop_reason": "end_turn", "content": content})
            )
        ) as client,
        pytest.raises(SchemaValidationError),
    ):
        extract(
            Invoice, configured("anthropic:test", output_mode="json_schema"), b"x", client=client
        )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_local_validation_and_retry(asynchronous):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=response({"items": []} if len(seen) == 1 else VALUE))

    model = Model("anthropic:test", api_key="test", output_mode="json_schema", validation_retries=1)
    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(Invoice, model, b"x", client=client)
    else:
        with httpx.Client(transport=transport) as client:
            result = extract(Invoice, model, b"x", client=client)
    assert result == Invoice.model_validate(VALUE)
    assert len(seen) == 2
    assert seen[0]["output_config"] == seen[1]["output_config"]
    assert "items" in seen[1]["system"]


def test_local_constraints_remain_enforced():
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, json=response({"items": [{"name": "!", "quantity": 0}]})
                )
            )
        ) as client,
        pytest.raises(SchemaValidationError) as error,
    ):
        extract(
            Invoice, configured("anthropic:test", output_mode="json_schema"), b"x", client=client
        )
    assert isinstance(error.value.__cause__, ValidationError)
    assert {e["type"] for e in error.value.__cause__.errors()} >= {
        "string_too_short",
        "greater_than_equal",
    }


class Recursive(BaseModel):
    children: list["Recursive"]


class Map(BaseModel):
    values: dict[str, int]


@pytest.mark.parametrize("schema", [Recursive, Map])
def test_unsupported_models_fail_before_request(schema):
    with pytest.raises(ValueError, match="Anthropic JSON output"):
        extract(schema, configured("anthropic:test", output_mode="json_schema"), b"x")


@pytest.mark.parametrize("keyword", ["output_config", "output_format"])
@pytest.mark.parametrize("mode", [None, "tool", "json_schema"])
def test_output_settings_reserved(keyword, mode):
    with pytest.raises(ValueError, match="cannot override"):
        extract(
            Invoice, configured("anthropic:test", output_mode=mode, settings={keyword: {}}), b"x"
        )


@pytest.mark.parametrize("mode", [None, "tool"])
def test_forced_tool_default_unchanged(mode):
    def handler(request):
        body = json.loads(request.content)
        assert "output_config" not in body
        assert body["tools"][0]["input_schema"] == Invoice.model_json_schema()
        assert body["tool_choice"] == {"type": "tool", "name": "extract"}
        return httpx.Response(
            200,
            json={
                "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "name": "extract", "input": VALUE}],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert extract(
            Invoice, configured("anthropic:test", output_mode=mode), b"x", client=client
        ) == Invoice.model_validate(VALUE)


def test_schema_edge_cases():
    from openextract._anthropic_schema import json_schema

    for schema in [
        {"$ref": "https://example.com/schema"},
        {"allOf": []},
        {"prefixItems": []},
        {"patternProperties": {}},
        {"enum": [{}]},
    ]:
        with pytest.raises(ValueError, match="Anthropic JSON output"):
            json_schema(schema)
    assert json_schema({"type": "string", "format": "date"})["format"] == "date"
    assert "format" not in json_schema({"type": "string", "format": "password"})
    assert json_schema({"type": "array", "minItems": 1})["minItems"] == 1

    class Cat(BaseModel):
        kind: Literal["cat"]

    class Dog(BaseModel):
        kind: Literal["dog"]

    class Pet(BaseModel):
        pet: Annotated[Cat | Dog, Field(discriminator="kind")]

    adapted = json_schema(Pet.model_json_schema())
    assert "anyOf" in adapted["properties"]["pet"]
    assert "discriminator" not in adapted["properties"]["pet"]
