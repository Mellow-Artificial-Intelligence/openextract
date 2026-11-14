"""Offline document annotation contracts; no OCR or paid provider calls."""

import base64
import json
from datetime import date
from decimal import Decimal

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from openextract import Model, ModelError, SchemaValidationError, extract, extract_async
from tests.helpers import configured, output, serve


class Invoice(BaseModel):
    model_config = ConfigDict(strict=True)

    issued: date
    total: Decimal


PDF = b"%PDF-1.7\nscanned invoice fixture"
ANNOTATION = '{"issued":"2026-11-14","total":"42.50"}'


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("source_kind", ["bytes", "path", "url"])
@pytest.mark.parametrize("instructions", [None, "Use the invoice date."])
async def test_document_annotation(asynchronous, source_kind, instructions, tmp_path, monkeypatch):
    seen = []

    def handler(request):
        if request.method == "GET":
            assert request.url == "https://example.com/scan.pdf"
            return httpx.Response(200, content=PDF)
        seen.append(request)
        return httpx.Response(200, json={"document_annotation": ANNOTATION, "pages": []})

    serve(monkeypatch, handler)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    model = Model("mistral:mistral-ocr-latest", api="mistral-ocr", settings={"pages": [0]})
    source = PDF
    if source_kind == "path":
        source = tmp_path / "scan.pdf"
        source.write_bytes(PDF)
    elif source_kind == "url":
        source = "https://example.com/scan.pdf"
    if asynchronous:
        result = await extract_async(Invoice, model, source, instructions)
    else:
        result = extract(Invoice, model, source, instructions)
    assert result == Invoice(issued=date(2026, 11, 14), total=Decimal("42.50"))
    (request,) = seen
    assert str(request.url) == "https://api.mistral.ai/v1/ocr"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "mistral-ocr-latest"
    assert body["document"] == {
        "type": "document_url",
        "document_url": "data:application/pdf;base64," + base64.b64encode(PDF).decode(),
    }
    assert body["document_annotation_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "extract", "schema": Invoice.model_json_schema(), "strict": True},
    }
    assert body["document_annotation_prompt"]
    if instructions:
        assert instructions in body["document_annotation_prompt"]
    assert body["pages"] == [0]
    assert body["include_image_base64"] is False
    assert body["bbox_annotation_format"] is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"document_annotation": None},
        {"document_annotation": ""},
        {"document_annotation": "I cannot extract that."},
        {"document_annotation": '{"issued":'},
        {"document_annotation": '{"issued":"invalid","total":"42.50"}'},
    ],
)
async def test_invalid_annotation(asynchronous, payload):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    model = configured("mistral:mistral-ocr-latest", api="mistral-ocr")
    with pytest.raises(SchemaValidationError):
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                await extract_async(Invoice, model, PDF, client=client)
        else:
            with httpx.Client(transport=transport) as client:
                extract(Invoice, model, PDF, client=client)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_annotation_validation_retry(asynchronous):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        value = '{"issued":"bad","total":"42.50"}' if len(seen) == 1 else ANNOTATION
        return httpx.Response(200, json={"document_annotation": value})

    transport = httpx.MockTransport(handler)
    model = Model("mistral:mistral-ocr-latest", api="mistral-ocr", api_key="test-key")
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(Invoice, model, PDF, "Read carefully.", client=client)
    else:
        with httpx.Client(transport=transport) as client:
            result = extract(Invoice, model, PDF, "Read carefully.", client=client)
    assert result.total == Decimal("42.50")
    assert len(seen) == 2
    assert "Correct these output validation errors" in seen[1]["document_annotation_prompt"]
    assert "Read carefully." in seen[1]["document_annotation_prompt"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [401, 413, 422, 429, 500])
async def test_provider_errors(asynchronous, status):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, json={"detail": "error"})
    )
    model = configured("mistral:mistral-ocr-latest", api="mistral-ocr")
    with pytest.raises(ModelError):
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                await extract_async(Invoice, model, PDF, client=client)
        else:
            with httpx.Client(transport=transport) as client:
                extract(Invoice, model, PDF, client=client)


@pytest.mark.parametrize("media_type", ["text/plain", "image/png", "audio/wav"])
def test_pdf_only(media_type):
    with pytest.raises(ValueError, match="only PDF"):
        extract(
            Invoice, configured("mistral:test", api="mistral-ocr"), b"data", media_type=media_type
        )


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "document",
        "document_annotation_format",
        "document_annotation_prompt",
        "bbox_annotation_format",
        "include_image_base64",
    ],
)
def test_reserved_fields(field):
    model = configured("mistral:test", api="mistral-ocr", settings={field: "override"})
    with pytest.raises(ValueError, match="cannot override request fields"):
        extract(Invoice, model, PDF)


def test_mistral_chat_is_not_rerouted():
    class Total(BaseModel):
        total: float

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        assert "document_annotation_format" not in json.loads(request.content)
        return httpx.Response(200, json=output("chat"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert (
            extract(
                Total, configured("mistral:mistral-ocr-latest"), "total 42", client=client
            ).total
            == 42
        )
