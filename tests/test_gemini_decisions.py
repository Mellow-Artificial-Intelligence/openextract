import base64
import json
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, create_model

from openextract import (
    ExtractionError,
    Model,
    ModelError,
    SchemaValidationError,
    extract,
    extract_async,
)
from tests.test_decisions import TRIAGE_ANSWERS, Currency, Triage


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "source,media_type,kind",
    [
        ("Paid receipt", "text/plain", "text"),
        (b"\x89PNG....", "image/png", "image"),
        (b"%PDF-1.7 fake", "application/pdf", "document"),
    ],
)
@pytest.mark.parametrize("instructions", [None, "Check carefully."])
async def test_gemini_rest_contract(
    monkeypatch, asynchronous, source, media_type, kind, instructions
):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")

    def handler(request):
        assert request.url == "https://generativelanguage.googleapis.com/v1beta/decisions"
        assert request.headers["x-goog-api-key"] == "gemini-test"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert set(body) == {"model", "input", "questions"}
        assert body["model"] == "gemini-decision-1"
        assert body["input"][0]["text"].endswith("\n")
        if instructions:
            assert body["input"][1] == {"type": "text", "text": instructions + "\n"}
        document = body["input"][-1]
        assert document["type"] == kind
        if kind == "text":
            assert document == {"type": "text", "text": source}
        else:
            assert document == {
                "type": kind,
                "mime_type": media_type,
                "data": base64.b64encode(source).decode(),
            }
        assert body["questions"] == {
            "document_type": {
                "type": "choice",
                "instructions": "What kind of document is this?",
                "criteria": {"invoice": "invoice", "receipt": "receipt"},
            },
            "currency": {
                "type": "choice",
                "instructions": "currency",
                "criteria": {"usd": "usd", "eur": "eur"},
            },
            "is_paid": {"type": "noul", "instructions": "is paid"},
        }
        return httpx.Response(200, json={"answers": TRIAGE_ANSWERS})

    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(
                Triage, "gemini-decisions:gemini-decision-1", source, instructions, client=client
            )
    else:
        with httpx.Client(transport=transport) as client:
            result = extract(
                Triage, "gemini-decisions:gemini-decision-1", source, instructions, client=client
            )
    assert result == Triage(document_type="receipt", currency=Currency.EUR, is_paid=True)


@pytest.mark.parametrize("count", [0, 1001])
def test_field_count(count):
    schema = create_model("Questions", **{f"q{i}": (bool, ...) for i in range(count)})
    with pytest.raises(ValueError, match="1 and 1000"):
        extract(schema, Model("gemini-decisions:test", api_key="test"), "text")


@pytest.mark.parametrize("name", ["é", "a" * 65])
def test_field_name(name):
    schema = create_model("Questions", **{name: (bool, ...)})
    with pytest.raises(ValueError, match="field names"):
        extract(schema, Model("gemini-decisions:test", api_key="test"), "text")


@pytest.mark.parametrize("options", [("only",), tuple(str(i) for i in range(256))])
def test_choice_count(options):
    schema = create_model("Choices", value=(Literal[options], ...))
    with pytest.raises(ValueError, match="2 and 255"):
        extract(schema, Model("gemini-decisions:test", api_key="test"), "text")


@pytest.mark.parametrize(
    "key", ["", "x" * 129, " space", "space ", "new\nline", "no\x7f", "a\u00a0b"]
)
def test_option_keys(key):
    schema = create_model("Choices", value=(Literal[key, "ok"], ...))
    with pytest.raises(ValueError, match="option key"):
        extract(schema, Model("gemini-decisions:test", api_key="test"), "text")


@pytest.mark.parametrize("media_type", ["audio/wav", "video/mp4", "image/svg+xml"])
def test_unsupported_media(media_type):
    with pytest.raises(ValueError, match="supports text, images and PDF"):
        extract(
            Triage, Model("gemini-decisions:test", api_key="test"), b"data", media_type=media_type
        )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"answers": []},
        {"answers": {}},
        {"answers": {**TRIAGE_ANSWERS, "is_paid": {"type": "noul", "noul": True}}},
        {"answers": {**TRIAGE_ANSWERS, "currency": {"type": "choice", "choice": "invalid"}}},
    ],
)
async def test_invalid_answers(payload, asynchronous):
    model = Model("gemini-decisions:gemini-decision-1", api_key="test")
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(SchemaValidationError):
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                await extract_async(Triage, model, "text", client=client)
        else:
            with httpx.Client(transport=transport) as client:
                extract(Triage, model, "text", client=client)


def test_auth_settings_and_free_form_rejection():
    with pytest.raises(ExtractionError, match="GOOGLE_API_KEY"):
        extract(Triage, "gemini-decisions:gemini-decision-1", "text")
    with pytest.raises(ValueError, match="settings"):
        extract(Triage, Model("gemini-decisions:test", api_key="test", settings={"x": 1}), "text")

    class FreeForm(BaseModel):
        value: str

    with pytest.raises(TypeError, match="only fill"):
        extract(FreeForm, Model("gemini-decisions:test", api_key="test"), "text")


def test_explicit_api_and_safety_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    monkeypatch.setenv("GOOGLE_BASE_URL", "https://proxy.example/v1beta")

    def handler(request):
        assert request.url == "https://proxy.example/v1beta/decisions"
        assert request.headers["x-goog-api-key"] == "google-test"
        return httpx.Response(400, json={"error": {"code": "safety", "message": "Blocked"}})

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ModelError, match="HTTP 400"),
    ):
        extract(
            Triage,
            Model("google:gemini-decision-1", api="gemini-decisions"),
            "text",
            client=client,
        )
