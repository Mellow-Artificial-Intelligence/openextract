"""Gemini parses Office and OpenDocument files itself, so their bytes are sent inline."""

import base64

import httpx
import pytest
from pydantic import BaseModel

from openextract import extract
from tests.helpers import configured, output

GEMINI = configured("google:gemini-4-hydrogen")
DATA = b"\x00\xffmock document bytes"
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


def inline(media_type):
    """``DATA`` as the Gemini API should receive it."""
    return {"inlineData": {"mimeType": media_type, "data": base64.b64encode(DATA).decode()}}


@pytest.mark.parametrize("extension,media_type", DOCUMENTS)
@pytest.mark.parametrize("source_kind", ["path", "bytes"])
async def test_office_document_is_sent_inline(
    run, provider, tmp_path, extension, media_type, source_kind
):
    path = tmp_path / f"invoice.{extension}"
    path.write_bytes(DATA)
    # A path's extension names the media type; raw bytes need the caller to pass it.
    source, explicit_type = (path, None) if source_kind == "path" else (DATA, media_type)
    provider.reply(output("gemini"))
    result = await run(Invoice, GEMINI, source, "Use the final total.", media_type=explicit_type)
    assert result == Invoice(total=42)
    (request,), (body,) = provider.requests, provider.bodies
    assert request.url.path == "/v1beta/models/gemini-4-hydrogen:generateContent"
    assert request.headers["x-goog-api-key"] == "test-key"
    assert body["contents"][0]["parts"][-1] == inline(media_type)
    assert body["generationConfig"]["responseJsonSchema"] == Invoice.model_json_schema()
    assert body["systemInstruction"] == {"parts": [{"text": "Use the final total."}]}


async def test_document_url_is_downloaded_before_the_gemini_request(run, provider):
    media_type = "application/vnd.oasis.opendocument.spreadsheet"

    def download(request):
        return httpx.Response(200, content=DATA, headers={"content-type": media_type})

    provider.reply(download, output("gemini"))
    assert await run(Invoice, GEMINI, "https://example.com/download") == Invoice(total=42)
    assert [request.method for request in provider.requests] == ["GET", "POST"]
    assert provider.requests[0].url == "https://example.com/download"
    assert provider.bodies[0]["contents"][0]["parts"][-1] == inline(media_type)


@pytest.mark.parametrize("media_type", ["application/rtf", "application/zip"])
def test_other_binary_documents_remain_unsupported(media_type):
    with pytest.raises(ValueError, match="gemini adapter does not support"):
        extract(Invoice, GEMINI, b"document", media_type=media_type)
