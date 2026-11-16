"""Mistral OCR document annotation: a PDF in, an annotation validated against the schema out."""

import base64
from datetime import date
from decimal import Decimal

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from openextract import Model, ModelError, SchemaValidationError, extract
from tests.helpers import configured, output

PDF = b"%PDF-1.7\nscanned invoice fixture"
PDF_URL = "https://example.com/scan.pdf"
PDF_DATA_URI = "data:application/pdf;base64," + base64.b64encode(PDF).decode()
ANNOTATION = output("mistral-ocr", {"issued": "2026-11-14", "total": "42.50"})


class Invoice(BaseModel):
    model_config = ConfigDict(strict=True)

    issued: date
    total: Decimal


def ocr(**options):
    """A Mistral model that speaks the OCR API instead of Chat Completions."""
    return configured("mistral:mistral-ocr-latest", api="mistral-ocr", **options)


@pytest.mark.parametrize("source_kind", ["bytes", "path", "url"])
@pytest.mark.parametrize("instructions", [None, "Use the invoice date."])
async def test_annotates_a_pdf_from_bytes_a_path_or_a_url(
    run, provider, monkeypatch, tmp_path, source_kind, instructions
):
    def download_then_annotate(request):
        if request.method == "GET":
            assert request.url == PDF_URL
            return httpx.Response(200, content=PDF)
        return httpx.Response(200, json={**ANNOTATION, "pages": []})

    provider.reply(download_then_annotate)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    path = tmp_path / "scan.pdf"
    path.write_bytes(PDF)
    source = {"bytes": PDF, "path": path, "url": PDF_URL}[source_kind]
    model = Model("mistral:mistral-ocr-latest", api="mistral-ocr", settings={"pages": [0]})
    result = await run(Invoice, model, source, instructions)
    assert result == Invoice(issued=date(2026, 11, 14), total=Decimal("42.50"))
    request, (body,) = provider.requests[-1], provider.bodies
    assert request.url == "https://api.mistral.ai/v1/ocr"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["model"] == "mistral-ocr-latest"
    assert body["document"] == {"type": "document_url", "document_url": PDF_DATA_URI}
    assert body["document_annotation_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "extract", "schema": Invoice.model_json_schema(), "strict": True},
    }
    assert body["document_annotation_prompt"]
    assert (instructions or "") in body["document_annotation_prompt"]
    assert body["pages"] == [0]
    assert body["include_image_base64"] is False
    assert body["bbox_annotation_format"] is None


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
async def test_invalid_annotation_raises_schema_validation_error(run, provider, payload):
    provider.reply(payload)
    with pytest.raises(SchemaValidationError):
        await run(Invoice, ocr(), PDF)


async def test_validation_retry_keeps_the_instructions(run, provider):
    provider.reply(output("mistral-ocr", {"issued": "bad", "total": "42.50"}), ANNOTATION)
    result = await run(Invoice, ocr(validation_retries=1), PDF, "Read carefully.")
    assert result.total == Decimal("42.50")
    _, retry = provider.bodies
    assert "Correct these output validation errors" in retry["document_annotation_prompt"]
    assert "Read carefully." in retry["document_annotation_prompt"]


@pytest.mark.parametrize("status", [401, 413, 422, 429, 500])
async def test_provider_error_raises_model_error(run, provider, status):
    provider.reply({"detail": "error"}, status=status)
    with pytest.raises(ModelError):
        await run(Invoice, ocr(), PDF)


@pytest.mark.parametrize("media_type", ["text/plain", "image/png", "audio/wav"])
def test_only_pdf_is_supported(media_type):
    with pytest.raises(ValueError, match="only PDF"):
        extract(Invoice, ocr(), b"data", media_type=media_type)


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
def test_settings_cannot_override_request_fields(field):
    with pytest.raises(ValueError, match="cannot override request fields"):
        extract(Invoice, ocr(settings={field: "override"}), PDF)


def test_mistral_chat_is_not_rerouted(provider):
    class Total(BaseModel):
        total: float

    provider.reply(output("chat"))
    assert extract(Total, configured("mistral:mistral-ocr-latest"), "total 42").total == 42
    assert provider.requests[0].url.path == "/v1/chat/completions"
    assert "document_annotation_format" not in provider.bodies[0]
