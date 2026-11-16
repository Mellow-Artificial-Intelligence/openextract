"""Offline Office-file contracts; the provider parses the actual document bytes."""

import base64
import json

import httpx
import pytest
from pydantic import BaseModel

from openextract import extract, extract_async
from tests.helpers import configured, output, serve

DOCUMENTS = [
    ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("doc", "application/msword"),
    ("dotx", "application/vnd.openxmlformats-officedocument.wordprocessingml.template"),
    ("odt", "application/vnd.oasis.opendocument.text"),
    ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ("ppt", "application/vnd.ms-powerpoint"),
    ("odp", "application/vnd.oasis.opendocument.presentation"),
    ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ("xls", "application/vnd.ms-excel"),
    ("ods", "application/vnd.oasis.opendocument.spreadsheet"),
]


class Invoice(BaseModel):
    total: float


@pytest.mark.parametrize("extension,media_type", DOCUMENTS)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_office_document_encoding(extension, media_type, asynchronous, tmp_path):
    data = b"\x00\xffmock document bytes"
    # Paths infer the MIME type; raw bytes need the caller's explicit MIME type.
    source = tmp_path / f"invoice.{extension}"
    source.write_bytes(data)
    seen = []

    def handler(request):
        seen.append(request)
        assert request.url.path == "/v1beta/models/gemini-4-hydrogen:generateContent"
        assert request.headers["x-goog-api-key"] == "test-key"
        body = json.loads(request.content)
        assert body["contents"][0]["parts"][-1] == {
            "inlineData": {"mimeType": media_type, "data": base64.b64encode(data).decode()}
        }
        assert body["generationConfig"]["responseJsonSchema"] == Invoice.model_json_schema()
        assert body["systemInstruction"] == {"parts": [{"text": "Use the final total."}]}
        return httpx.Response(200, json=output("gemini"))

    model = configured("google:gemini-4-hydrogen")
    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(
                Invoice, model, data, "Use the final total.", media_type=media_type, client=client
            )
    else:
        with httpx.Client(transport=transport) as client:
            result = extract(Invoice, model, source, "Use the final total.", client=client)
    assert result == Invoice(total=42)
    assert len(seen) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_document_url_downloaded_before_gemini_request(asynchronous, monkeypatch):
    media_type = "application/vnd.oasis.opendocument.spreadsheet"
    data = b"\x00\xffmock spreadsheet"
    seen = []

    def handler(request):
        seen.append(request.method)
        if request.method == "GET":
            assert request.url == "https://example.com/download"
            return httpx.Response(200, content=data, headers={"content-type": media_type})
        body = json.loads(request.content)
        assert body["contents"][0]["parts"][-1] == {
            "inlineData": {"mimeType": media_type, "data": base64.b64encode(data).decode()}
        }
        return httpx.Response(200, json=output("gemini"))

    serve(monkeypatch, handler)
    model = configured("google:gemini-4-hydrogen")
    if asynchronous:
        result = await extract_async(Invoice, model, "https://example.com/download")
    else:
        result = extract(Invoice, model, "https://example.com/download")
    assert result == Invoice(total=42)
    assert seen == ["GET", "POST"]


@pytest.mark.parametrize("media_type", ["application/rtf", "application/zip"])
def test_other_binary_documents_remain_unsupported(media_type):
    with pytest.raises(ValueError, match="gemini adapter does not support"):
        extract(Invoice, configured("google:test"), b"document", media_type=media_type)
