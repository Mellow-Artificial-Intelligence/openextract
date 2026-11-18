"""General adapters: each API's request shape, media encoding, settings, and response checks."""

import copy
from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openextract import SchemaValidationError, extract
from openextract._schemas import openai_strict
from tests.helpers import Invoice, configured, custom, output

SCHEMA = Invoice.model_json_schema()
PNG = "iVBORw=="
PNG_URI = "data:image/png;base64," + PNG
PDF_URI = "data:application/pdf;base64," + PNG
# API: model prefix, request path, key header, and the request fields that carry the schema.
APIS = {
    "responses": (
        "openai",
        "/v1/responses",
        "authorization",
        {"text.format.strict": True, "text.format.schema.required": ["total"]},
    ),
    "chat": (
        "openai-chat",
        "/v1/chat/completions",
        "authorization",
        {
            "tools.0.function.parameters": SCHEMA,
            "tool_choice": {"type": "function", "function": {"name": "extract"}},
        },
    ),
    "anthropic": (
        "anthropic",
        "/v1/messages",
        "x-api-key",
        {
            "tools.0.input_schema": SCHEMA,
            "tool_choice": {"type": "tool", "name": "extract"},
            "max_tokens": 4096,
        },
    ),
    "gemini": (
        "google-gla",
        "/v1beta/models/test:generateContent",
        "x-goog-api-key",
        {
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": SCHEMA,
            },
            "contents.0.role": "user",
        },
    ),
    "ollama": ("ollama", "/api/chat", "authorization", {"stream": False, "format": SCHEMA}),
    "cohere": (
        "cohere",
        "/v2/chat",
        "authorization",
        {"response_format": {"type": "json_object", "schema": SCHEMA}},
    ),
    "bedrock": (
        "bedrock",
        "/model/test/converse",
        "authorization",
        {
            "toolConfig.tools.0.toolSpec.inputSchema": {"json": SCHEMA},
            "toolConfig.toolChoice": {"tool": {"name": "extract"}},
        },
    ),
}


def dig(value, path):
    """Follow a dotted path such as ``tools.0.function`` through dicts and lists."""
    for key in path.split("."):
        value = value[int(key) if key.isdigit() else key]
    return value


def document(body, api):
    """The part of a request body that carries the source."""
    if api == "responses":
        return body["input"][-1]["content"][-1]
    if api == "gemini":
        return body["contents"][-1]["parts"][-1]
    if api == "ollama":
        return body["messages"][-1]
    return body["messages"][-1]["content"][-1]


def chat(finish_reason, **message):
    return {"choices": [{"finish_reason": finish_reason, "message": message}]}


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("instructions", [None, "Use EUR."])
async def test_request_contract(run, provider, api, instructions):
    prefix, path, key_header, expected = APIS[api]
    base_url = "https://example.com" if api == "bedrock" else None
    provider.reply(output(api))
    model = configured(prefix + ":test", base_url=base_url)
    assert await run(Invoice, model, "Total: 42", instructions) == Invoice(total=42)
    request, body = provider.requests[0], provider.bodies[0]
    assert request.url.path == path
    assert request.headers[key_header] == (
        "test-key" if key_header != "authorization" else "Bearer test-key"
    )
    assert "Total: 42" in str(document(body, api))
    assert instructions is None or instructions in str(body)
    assert {field: dig(body, field) for field in expected} == expected
    if api == "anthropic":
        assert request.headers["anthropic-version"] == "2023-06-01"
    if api == "cohere":
        assert "JSON" in body["messages"][-1]["content"][0]["text"]


@pytest.mark.parametrize(
    "api,media_type,expected",
    [
        ("responses", "image/png", {"type": "input_image", "image_url": PNG_URI}),
        (
            "responses",
            "application/pdf",
            {"type": "input_file", "filename": "document.pdf", "file_data": PDF_URI},
        ),
        ("chat", "image/png", {"type": "image_url", "image_url": {"url": PNG_URI}}),
        (
            "chat",
            "application/pdf",
            {"type": "file", "file": {"filename": "document.pdf", "file_data": PDF_URI}},
        ),
        (
            "chat",
            "audio/mpeg",
            {"type": "input_audio", "input_audio": {"format": "mp3", "data": PNG}},
        ),
        (
            "chat",
            "audio/wav",
            {"type": "input_audio", "input_audio": {"format": "wav", "data": PNG}},
        ),
        ("cohere", "image/png", {"type": "image_url", "image_url": {"url": PNG_URI}}),
        (
            "anthropic",
            "image/png",
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
        ),
        (
            "anthropic",
            "application/pdf",
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": PNG},
            },
        ),
        ("gemini", "video/mp4", {"inlineData": {"mimeType": "video/mp4", "data": PNG}}),
        ("gemini", "audio/flac", {"inlineData": {"mimeType": "audio/flac", "data": PNG}}),
        ("gemini", "application/pdf", {"inlineData": {"mimeType": "application/pdf", "data": PNG}}),
        ("bedrock", "image/png", {"image": {"format": "png", "source": {"bytes": PNG}}}),
        (
            "bedrock",
            "application/pdf",
            {"document": {"format": "pdf", "name": "document", "source": {"bytes": PNG}}},
        ),
    ],
)
def test_media_encoding(provider, api, media_type, expected):
    provider.reply(output(api))
    extract(Invoice, custom(api), b"\x89PNG", media_type=media_type)
    assert document(provider.bodies[0], api) == expected


def test_ollama_sends_images_and_keeps_model_tags(provider):
    provider.reply(output("ollama"))
    extract(Invoice, "ollama:llama3:latest", b"\x89PNG")
    assert provider.bodies[0]["model"] == "llama3:latest"
    assert provider.bodies[0]["messages"][-1]["images"] == [PNG]


@pytest.mark.parametrize("api", APIS)
def test_unsupported_media_is_rejected_before_any_request(provider, api):
    with pytest.raises(ValueError, match="does not support"):
        extract(Invoice, custom(api), b"file", media_type="application/x-unknown")
    assert provider.requests == []


@pytest.mark.parametrize(
    "api,settings,key",
    [
        ("responses", {"temperature": 0}, "temperature"),
        ("anthropic", {"max_tokens": 1024}, "max_tokens"),
        ("gemini", {"maxOutputTokens": 1024}, "generationConfig.maxOutputTokens"),
        ("ollama", {"options": {"temperature": 0}}, "options"),
        ("bedrock", {"inferenceConfig": {"maxTokens": 1024}}, "inferenceConfig"),
    ],
)
def test_generation_settings_use_native_names(provider, api, settings, key):
    provider.reply(output(api))
    extract(Invoice, custom(api, settings=settings), b"text")
    assert dig(provider.bodies[0], key) == next(iter(settings.values()))


@pytest.mark.parametrize(
    "api,settings",
    [
        ("responses", {"model": "wrong"}),
        ("responses", {"instructions": "replace"}),
        ("chat", {"messages": []}),
        ("gemini", {"responseJsonSchema": {}}),
        ("gemini", {"responseFormat": []}),
        ("chat", {"stream": True}),
    ],
)
def test_settings_cannot_replace_request_structure(provider, api, settings):
    with pytest.raises(ValueError, match="settings"):
        extract(Invoice, custom(api, settings=settings), b"text")
    assert provider.requests == []


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("quantity", [2, "2"], ids=["valid", "invalid"])
async def test_strict_types_keep_json_validation_semantics(run, provider, api, quantity):
    class StrictInvoice(BaseModel):
        model_config = ConfigDict(strict=True)

        issued: date
        total: Decimal
        quantity: int

    provider.reply(output(api, {"issued": "2026-11-10", "total": "12.50", "quantity": quantity}))
    if quantity == 2:
        expected = StrictInvoice(issued=date(2026, 11, 10), total=Decimal("12.50"), quantity=2)
        assert await run(StrictInvoice, custom(api), b"Invoice") == expected
        return
    with pytest.raises(SchemaValidationError) as error:
        await run(StrictInvoice, custom(api), b"Invoice")
    assert isinstance(error.value.__cause__, ValidationError)
    assert [detail["loc"] for detail in error.value.__cause__.errors()] == [("quantity",)]


@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("kind", ["card", "unknown"], ids=["valid", "invalid"])
async def test_tagged_unions_in_strict_output(run, provider, api, kind):
    class Card(BaseModel):
        kind: Literal["card"]
        last_four: str

    class Cash(BaseModel):
        kind: Literal["cash"]
        currency: str

    class Receipt(BaseModel):
        payments: list[Annotated[Card | Cash, Field(discriminator="kind")]]

    original = Receipt.model_json_schema()
    value = {"payments": [{"kind": kind, "last_four": "1234"}]}
    if api == "chat":
        model, reply, field = (
            custom(api, output_mode="json_schema"),
            output("chat-json", value),
            "response_format.json_schema.schema",
        )
    else:
        model, reply, field = custom(api), output(api, value), "text.format.schema"
    provider.reply(reply)
    if kind == "card":
        expected = Receipt(payments=[Card(kind="card", last_four="1234")])
        assert await run(Receipt, model, b"Receipt") == expected
    else:
        with pytest.raises(SchemaValidationError) as error:
            await run(Receipt, model, b"Receipt")
        assert error.value.__cause__.errors()[0]["type"] == "union_tag_invalid"
    schema = dig(provider.bodies[0], field)
    union = schema["properties"]["payments"]["items"]
    assert "discriminator" not in union and "oneOf" not in union
    assert union["anyOf"] == original["properties"]["payments"]["items"]["oneOf"]
    assert all(child["additionalProperties"] is False for child in schema["$defs"].values())
    assert Receipt.model_json_schema() == original


def test_strict_schema_requires_every_field_and_keeps_the_original():
    class Child(BaseModel):
        value: int = 3

    class Data(BaseModel):
        children: list[Child]
        optional: Child | None = None
        category: Literal["a", "b"] = "a"

    original = Data.model_json_schema()
    before = copy.deepcopy(original)
    adapted = openai_strict(original)
    assert original == before
    assert adapted["required"] == ["children", "optional", "category"]
    assert "default" not in adapted["properties"]["category"]
    assert adapted["$defs"]["Child"]["additionalProperties"] is False
    assert adapted["$defs"]["Child"]["required"] == ["value"]


def test_open_dictionaries_need_tool_output(provider):
    class Data(BaseModel):
        values: dict[str, int]

    with pytest.raises(ValueError, match="open-ended dictionaries"):
        extract(Data, configured(), b"text")
    provider.reply(output("chat", {"values": {"arbitrary": 2}}))
    assert extract(Data, configured("openai-chat:test"), b"text").values == {"arbitrary": 2}


@pytest.mark.parametrize(
    "api,payload",
    [
        ("responses", output() | {"status": "incomplete"}),
        ("responses", {"status": "completed", "output": [{"type": "reasoning"}]}),
        (
            "responses",
            {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "refusal"}]}],
            },
        ),
        (
            "responses",
            {
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "not JSON"}]}
                ],
            },
        ),
        ("chat", chat("stop", content="plain text")),
        ("chat", chat("tool_calls", refusal="refused")),
        ("chat", chat("tool_calls", tool_calls=[])),
        ("chat", chat("tool_calls", tool_calls=[{"function": {"name": "wrong"}}])),
        (
            "chat",
            chat("tool_calls", tool_calls=[{"function": {"name": "extract", "arguments": 42}}]),
        ),
        ("chat", chat("length", **output("chat")["choices"][0]["message"])),
        ("chat", chat("content_filter", **output("chat")["choices"][0]["message"])),
        ("anthropic", output("anthropic") | {"stop_reason": "max_tokens"}),
        ("anthropic", {"stop_reason": "tool_use", "content": []}),
        (
            "anthropic",
            {"stop_reason": "tool_use", "content": [{"type": "tool_use", "name": "wrong"}]},
        ),
        ("bedrock", output("bedrock") | {"stopReason": "guardrail_intervened"}),
        ("bedrock", {"stopReason": "tool_use", "output": {"message": {"content": []}}}),
        (
            "bedrock",
            {
                "stopReason": "tool_use",
                "output": {"message": {"content": [{"toolUse": {"name": "wrong"}}]}},
            },
        ),
        ("gemini", {"promptFeedback": {"blockReason": "SAFETY"}}),
        ("gemini", {"candidates": [{"finishReason": "MAX_TOKENS"}]}),
        ("ollama", output("ollama") | {"done": False}),
        ("ollama", output("ollama") | {"done_reason": "length"}),
        ("cohere", output("cohere") | {"finish_reason": "MAX_TOKENS"}),
    ],
)
def test_refusals_and_incomplete_output_raise_schema_validation_error(provider, api, payload):
    provider.reply(payload)
    with pytest.raises(SchemaValidationError):
        extract(Invoice, custom(api), b"text")
    assert len(provider.requests) == 1


def test_empty_output_fails_without_a_validation_retry(provider):
    provider.reply({"status": "completed", "output": [{"type": "reasoning"}]})
    with pytest.raises(SchemaValidationError, match="no structured output"):
        extract(Invoice, configured(validation_retries=1), b"text")
    assert len(provider.requests) == 1


async def test_chat_accepts_a_forced_tool_call_that_reports_a_natural_stop(run, provider):
    provider.reply(chat("stop", **output("chat")["choices"][0]["message"]))
    assert await run(Invoice, configured("openai-chat:test"), b"text") == Invoice(total=42)


def test_gemini_thoughts_are_not_read_as_output(provider):
    payload = output("gemini")
    payload["candidates"][0]["content"]["parts"].insert(0, {"thought": True, "text": "private"})
    provider.reply(payload)
    assert extract(Invoice, configured("google:test"), b"text").total == 42


def test_empty_object_output_is_valid(provider):
    class Empty(BaseModel):
        pass

    provider.reply(output("anthropic", {}))
    assert extract(Empty, configured("anthropic:test"), b"text") == Empty()
