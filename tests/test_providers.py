import copy
import json
from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openextract import SchemaValidationError, extract, extract_async
from openextract._providers import _strict_schema
from tests.helpers import configured, output


class Invoice(BaseModel):
    total: float


CASES = [
    ("openai", "responses", "/v1/responses", "authorization"),
    ("openai-chat", "chat", "/v1/chat/completions", "authorization"),
    ("anthropic", "anthropic", "/v1/messages", "x-api-key"),
    ("google-gla", "gemini", "/v1beta/models/test:generateContent", "x-goog-api-key"),
    ("ollama", "ollama", "/api/chat", "authorization"),
    ("cohere", "cohere", "/v2/chat", "authorization"),
    ("bedrock", "bedrock", "/model/test/converse", "authorization"),
]


def document(body, api):
    if api == "responses":
        return body["input"][-1]["content"][-1]
    if api == "gemini":
        return body["contents"][-1]["parts"][-1]
    if api == "ollama":
        return body["messages"][-1]
    return body["messages"][-1]["content"][-1]


@pytest.mark.parametrize("prefix,api,path,auth_header", CASES)
@pytest.mark.parametrize("instructions", [None, "Use EUR."])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_adapter_contract(prefix, api, path, auth_header, instructions, asynchronous):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=output(api))

    model = configured(
        prefix + ":test", base_url="https://example.com" if api == "bedrock" else None
    )
    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(Invoice, model, "Total: 42", instructions, client=client)
    else:
        with httpx.Client(transport=transport) as client:
            result = extract(Invoice, model, "Total: 42", instructions, client=client)
    assert result == Invoice(total=42)
    request = seen[0]
    assert request.url.path == path
    assert request.headers[auth_header] == (
        "test-key" if api in ("gemini", "anthropic") else "Bearer test-key"
    )
    body = json.loads(request.content)
    assert "Total: 42" in json.dumps(document(body, api))
    if instructions:
        assert instructions in json.dumps(body)
    if api == "responses":
        assert body["text"]["format"]["strict"] is True
        assert body["text"]["format"]["schema"]["required"] == ["total"]
    elif api == "chat":
        assert body["tools"][0]["function"]["parameters"] == Invoice.model_json_schema()
        assert body["tool_choice"] == {"type": "function", "function": {"name": "extract"}}
    elif api == "anthropic":
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert body["tools"][0]["input_schema"] == Invoice.model_json_schema()
        assert body["tool_choice"] == {"type": "tool", "name": "extract"}
        assert body["max_tokens"] == 4096
    elif api == "gemini":
        assert body["generationConfig"] == {
            "responseMimeType": "application/json",
            "responseJsonSchema": Invoice.model_json_schema(),
        }
        assert body["contents"][0]["role"] == "user"
    elif api == "ollama":
        assert body["stream"] is False
        assert body["format"] == Invoice.model_json_schema()
    elif api == "cohere":
        assert body["response_format"] == {
            "type": "json_object",
            "schema": Invoice.model_json_schema(),
        }
        assert "JSON" in body["messages"][-1]["content"][0]["text"]
    else:
        assert body["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"] == {
            "json": Invoice.model_json_schema()
        }
        assert body["toolConfig"]["toolChoice"] == {"tool": {"name": "extract"}}


@pytest.mark.parametrize("api", [case[1] for case in CASES])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("valid", [False, True])
async def test_strict_schema_uses_json_validation_across_adapters(api, asynchronous, valid):
    class StrictInvoice(BaseModel):
        model_config = ConfigDict(strict=True)

        issued: date
        total: Decimal
        quantity: int

    value = {"issued": "2026-11-10", "total": "12.50", "quantity": 2 if valid else "2"}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=output(api, value)))
    model = configured("custom:test", api=api, base_url="https://example.com")

    async def run():
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                return await extract_async(StrictInvoice, model, b"Invoice", client=client)
        with httpx.Client(transport=transport) as client:
            return extract(StrictInvoice, model, b"Invoice", client=client)

    if valid:
        assert await run() == StrictInvoice(
            issued=date(2026, 11, 10), total=Decimal("12.50"), quantity=2
        )
    else:
        with pytest.raises(SchemaValidationError) as error:
            await run()
        assert isinstance(error.value.__cause__, ValidationError)
        assert [detail["loc"] for detail in error.value.__cause__.errors()] == [("quantity",)]


@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("valid", [False, True])
async def test_discriminated_union_in_strict_output(api, asynchronous, valid):
    class Card(BaseModel):
        kind: Literal["card"]
        last_four: str

    class Cash(BaseModel):
        kind: Literal["cash"]
        currency: str

    class Receipt(BaseModel):
        payments: list[Annotated[Card | Cash, Field(discriminator="kind")]]

    original = Receipt.model_json_schema()
    value = {"payments": [{"kind": "card" if valid else "unknown", "last_four": "1234"}]}

    def handler(request):
        body = json.loads(request.content)
        schema = (
            body["text"]["format"]["schema"]
            if api == "responses"
            else body["response_format"]["json_schema"]["schema"]
        )
        union = schema["properties"]["payments"]["items"]
        assert "discriminator" not in union
        assert "oneOf" not in union
        assert union["anyOf"] == original["properties"]["payments"]["items"]["oneOf"]
        assert all(child["additionalProperties"] is False for child in schema["$defs"].values())
        payload = (
            output(api, value)
            if api == "responses"
            else {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
        )
        return httpx.Response(200, json=payload)

    model = configured(
        "custom:test",
        api=api,
        base_url="https://example.com",
        output_mode="json_schema" if api == "chat" else None,
    )
    transport = httpx.MockTransport(handler)

    async def run():
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                return await extract_async(Receipt, model, b"Receipt", client=client)
        with httpx.Client(transport=transport) as client:
            return extract(Receipt, model, b"Receipt", client=client)

    if valid:
        assert await run() == Receipt(payments=[Card(kind="card", last_four="1234")])
    else:
        with pytest.raises(SchemaValidationError) as error:
            await run()
        assert error.value.__cause__.errors()[0]["type"] == "union_tag_invalid"
    assert Receipt.model_json_schema() == original


@pytest.mark.parametrize(
    "api,media_type,expected",
    [
        (
            "responses",
            "image/png",
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw=="},
        ),
        (
            "responses",
            "application/pdf",
            {
                "type": "input_file",
                "filename": "document.pdf",
                "file_data": "data:application/pdf;base64,iVBORw==",
            },
        ),
        (
            "chat",
            "image/png",
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}},
        ),
        (
            "chat",
            "application/pdf",
            {
                "type": "file",
                "file": {
                    "filename": "document.pdf",
                    "file_data": "data:application/pdf;base64,iVBORw==",
                },
            },
        ),
        (
            "chat",
            "audio/mpeg",
            {"type": "input_audio", "input_audio": {"format": "mp3", "data": "iVBORw=="}},
        ),
        (
            "chat",
            "audio/wav",
            {"type": "input_audio", "input_audio": {"format": "wav", "data": "iVBORw=="}},
        ),
        (
            "cohere",
            "image/png",
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}},
        ),
        (
            "anthropic",
            "image/png",
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw=="},
            },
        ),
        (
            "anthropic",
            "application/pdf",
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "iVBORw=="},
            },
        ),
        ("gemini", "video/mp4", {"inlineData": {"mimeType": "video/mp4", "data": "iVBORw=="}}),
        ("gemini", "audio/flac", {"inlineData": {"mimeType": "audio/flac", "data": "iVBORw=="}}),
        (
            "gemini",
            "application/pdf",
            {"inlineData": {"mimeType": "application/pdf", "data": "iVBORw=="}},
        ),
        ("bedrock", "image/png", {"image": {"format": "png", "source": {"bytes": "iVBORw=="}}}),
        (
            "bedrock",
            "application/pdf",
            {"document": {"format": "pdf", "name": "document", "source": {"bytes": "iVBORw=="}}},
        ),
    ],
)
def test_media_encoding(api, media_type, expected):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=output(api))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = configured("custom:test", api=api, base_url="https://example.com")
        extract(Invoice, model, b"\x89PNG", media_type=media_type, client=client)
    assert document(seen[0], api) == expected


def test_ollama_image_encoding_and_model_tag():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=output("ollama"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        extract(Invoice, "ollama:llama3:latest", b"\x89PNG", client=client)
    assert seen[0]["model"] == "llama3:latest"
    assert seen[0]["messages"][-1]["images"] == ["iVBORw=="]


@pytest.mark.parametrize(
    "api", ["responses", "chat", "anthropic", "cohere", "gemini", "ollama", "bedrock"]
)
def test_unsupported_media_rejected_before_request(api):
    def unexpected(request):
        pytest.fail("Unsupported inputs must not be sent")

    with (
        httpx.Client(transport=httpx.MockTransport(unexpected)) as client,
        pytest.raises(ValueError, match="does not support"),
    ):
        extract(
            Invoice,
            configured("custom:test", api=api, base_url="https://example.com"),
            b"file",
            media_type="application/x-unknown",
            client=client,
        )


@pytest.mark.parametrize(
    "api,field,value",
    [
        ("responses", "status", "incomplete"),
        ("anthropic", "stop_reason", "max_tokens"),
        ("ollama", "done", False),
        ("ollama", "done_reason", "length"),
        ("cohere", "finish_reason", "MAX_TOKENS"),
        ("bedrock", "stopReason", "guardrail_intervened"),
    ],
)
def test_incomplete_output_rejected(api, field, value):
    payload = output(api)
    payload[field] = value
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ) as client,
        pytest.raises(SchemaValidationError),
    ):
        extract(
            Invoice,
            configured("custom:test", api=api, base_url="https://example.com"),
            b"text",
            client=client,
        )


@pytest.mark.parametrize(
    "api,payload",
    [
        (
            "responses",
            {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "refusal"}]}],
            },
        ),
        ("responses", {"status": "completed", "output": [{"type": "reasoning"}]}),
        (
            "responses",
            {
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "not JSON"}]}
                ],
            },
        ),
        ("chat", {"choices": [{"finish_reason": "stop", "message": {"content": "plain text"}}]}),
        ("chat", {"choices": [{"finish_reason": "tool_calls", "message": {"refusal": "refused"}}]}),
        ("chat", {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": []}}]}),
        (
            "chat",
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {"tool_calls": [{"function": {"name": "wrong"}}]},
                    }
                ]
            },
        ),
        (
            "chat",
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [{"function": {"name": "extract", "arguments": 42}}]
                        },
                    }
                ]
            },
        ),
        ("anthropic", {"stop_reason": "tool_use", "content": []}),
        (
            "anthropic",
            {"stop_reason": "tool_use", "content": [{"type": "tool_use", "name": "wrong"}]},
        ),
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
    ],
)
def test_refusals_and_invalid_output(api, payload):
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ) as client,
        pytest.raises(SchemaValidationError),
    ):
        extract(
            Invoice,
            configured("custom:test", api=api, base_url="https://example.com"),
            b"text",
            client=client,
        )


def test_gemini_thoughts_are_not_parsed_as_output():
    payload = output("gemini")
    payload["candidates"][0]["content"]["parts"].insert(
        0, {"thought": True, "text": "private reasoning"}
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        assert extract(Invoice, configured("google:test"), b"text", client=client).total == 42


def test_empty_object_output_is_valid():
    class Empty(BaseModel):
        pass

    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=output("anthropic", {})))
    ) as client:
        assert extract(Empty, configured("anthropic:test"), b"text", client=client) == Empty()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_chat_accepts_complete_forced_tool_call_with_natural_stop(asynchronous):
    payload = output("chat")
    payload["choices"][0]["finish_reason"] = "stop"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(
                Invoice, configured("openai-chat:test"), b"text", client=client
            )
    else:
        with httpx.Client(transport=transport) as client:
            result = extract(Invoice, configured("openai-chat:test"), b"text", client=client)
    assert result == Invoice(total=42)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_chat_rejects_incomplete_response_even_with_valid_tool_arguments(finish_reason):
    payload = output("chat")
    payload["choices"][0]["finish_reason"] = finish_reason
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ) as client,
        pytest.raises(SchemaValidationError),
    ):
        extract(Invoice, configured("openai-chat:test"), b"text", client=client)


def test_strict_schema_preserves_nested_definitions_and_original():
    class Child(BaseModel):
        value: int = 3

    class Data(BaseModel):
        children: list[Child]
        optional: Child | None = None
        category: Literal["a", "b"] = "a"

    original = Data.model_json_schema()
    before = copy.deepcopy(original)
    adapted = _strict_schema(original)
    assert original == before
    assert adapted["required"] == ["children", "optional", "category"]
    assert "default" not in adapted["properties"]["category"]
    assert adapted["$defs"]["Child"]["additionalProperties"] is False
    assert adapted["$defs"]["Child"]["required"] == ["value"]


def test_open_dictionary_uses_tool_api():
    class Data(BaseModel):
        values: dict[str, int]

    with pytest.raises(ValueError, match="open-ended dictionaries"):
        extract(Data, configured(), b"text")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=output("chat", {"values": {"arbitrary": 2}}))
        )
    ) as client:
        result = extract(Data, configured("openai-chat:test"), b"text", client=client)
    assert result.values == {"arbitrary": 2}


@pytest.mark.parametrize(
    "api,settings,key",
    [
        ("responses", {"temperature": 0}, "temperature"),
        ("anthropic", {"max_tokens": 1024}, "max_tokens"),
        ("gemini", {"maxOutputTokens": 1024}, "generationConfig"),
        ("ollama", {"options": {"temperature": 0}}, "options"),
        ("bedrock", {"inferenceConfig": {"maxTokens": 1024}}, "inferenceConfig"),
    ],
)
def test_generation_settings(api, settings, key):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=output(api))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        extract(
            Invoice,
            configured("custom:test", api=api, base_url="https://example.com", settings=settings),
            b"text",
            client=client,
        )
    if api == "gemini":
        assert seen[0][key]["maxOutputTokens"] == 1024
    else:
        assert seen[0][key] == settings[key]


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
def test_settings_cannot_replace_request_structure(api, settings):
    with pytest.raises(ValueError):
        extract(
            Invoice,
            configured("custom:test", api=api, base_url="https://example.com", settings=settings),
            b"text",
        )
