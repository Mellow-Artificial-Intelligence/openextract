import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

import openextract._extract as extract_module
from openextract import (
    ExtractionError,
    Model,
    ModelError,
    SchemaValidationError,
    extract,
    extract_async,
)
from tests.helpers import configured, output, serve

PDF_BYTES = b"%PDF-1.7 fake"


class Invoice(BaseModel):
    total: float


def record(monkeypatch, *, payload=None):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=output() if payload is None else payload)

    serve(monkeypatch, handler)
    return seen


def test_extracts_from_path(monkeypatch, tmp_path: Path):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(PDF_BYTES)
    seen = record(monkeypatch)
    result = extract(Invoice, configured(), path, "Find the total.")
    assert result == Invoice(total=42.0)
    body = json.loads(seen[0].content)
    assert seen[0].url == "https://api.openai.com/v1/responses"
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert body["instructions"] == "Find the total."
    assert body["input"][0]["content"][-1] == {
        "type": "input_file",
        "filename": "document.pdf",
        "file_data": "data:application/pdf;base64,JVBERi0xLjcgZmFrZQ==",
    }
    assert body["text"]["format"]["schema"]["additionalProperties"] is False


async def test_extract_async_from_bytes(monkeypatch):
    seen = record(monkeypatch)
    result = await extract_async(Invoice, configured(), b"\x89PNG....")
    assert result.total == 42.0
    assert json.loads(seen[0].content)["input"][0]["content"][-1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,iVBORy4uLi4=",
    }


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("customized", [False, True], ids=["simple", "customized"])
async def test_simple_and_customized_calls_preserve_contract(monkeypatch, asynchronous, customized):
    seen = record(monkeypatch)
    source = b"Total: 42"
    kwargs = (
        {"instructions": "Use the final total.", "media_type": "text/markdown"}
        if customized
        else {}
    )
    if asynchronous:
        result = await extract_async(Invoice, configured(), source, **kwargs)
    else:
        result = extract(Invoice, configured(), source, **kwargs)
    assert type(result) is Invoice
    assert result.total == 42.0
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body.get("instructions") == kwargs.get("instructions")
    assert body["input"][0]["content"][-1] == {"type": "input_text", "text": source.decode()}


@pytest.mark.parametrize(
    "data,media_type",
    [
        (b"%PDF-1.4", "application/pdf"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"GIF89a", "image/gif"),
        (b"ID3\x04", "audio/mpeg"),
        (b"fLaC", "audio/flac"),
        (b"OggS", "audio/ogg"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav"),
        (b"RIFF\x00\x00\x00\x00AVI LIST", None),
        (b"\x00\x00\x00\x18ftypmp42", "video/mp4"),
        (b"plain text, caf\xc3\xa9", "text/plain"),
        (b"\xff\xfe\x00\x01", None),
    ],
)
def test_sniffs_media_type(data, media_type):
    assert extract_module._sniff(data) == media_type


def test_explicit_media_type_wins(monkeypatch):
    seen = record(monkeypatch)
    extract(Invoice, configured(), PDF_BYTES, media_type="text/plain")
    assert json.loads(seen[0].content)["input"][0]["content"][-1] == {
        "type": "input_text",
        "text": PDF_BYTES.decode(),
    }


def test_uninferable_media_type_raises():
    with pytest.raises(ValueError, match="media_type"):
        extract(Invoice, configured(), b"\xff\xfe\x00\x01")


def test_url_uses_content_type_and_does_not_send_credentials(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.host == "example.com":
            assert "authorization" not in request.headers
            return httpx.Response(
                200, content=b"<p>hi</p>", headers={"content-type": "text/html; charset=utf-8"}
            )
        return httpx.Response(200, json=output())

    serve(monkeypatch, handler)
    extract(Invoice, configured(), "https://example.com/page")
    assert json.loads(seen[-1].content)["input"][0]["content"][-1]["text"] == "<p>hi</p>"


async def test_url_prefers_extension_over_generic_header(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(
                200, content=PDF_BYTES, headers={"content-type": "application/octet-stream"}
            )
        return httpx.Response(200, json=output())

    serve(monkeypatch, handler)
    await extract_async(Invoice, configured(), "https://example.com/invoice.pdf")
    assert json.loads(seen[-1].content)["input"][0]["content"][-1]["type"] == "input_file"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_url_failure_raises_extraction_error(monkeypatch, asynchronous):
    serve(monkeypatch, lambda request: httpx.Response(404))
    with pytest.raises(ExtractionError, match="Could not fetch") as error:
        if asynchronous:
            await extract_async(Invoice, configured(), "https://example.com/missing.pdf")
        else:
            extract(Invoice, configured(), "https://example.com/missing.pdf")
    assert isinstance(error.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_invalid_output_raises_schema_validation_error(monkeypatch, asynchronous):
    record(monkeypatch, payload=output(value={"total": "lots"}))
    with pytest.raises(SchemaValidationError) as error:
        if asynchronous:
            await extract_async(Invoice, configured(), PDF_BYTES)
        else:
            extract(Invoice, configured(), PDF_BYTES)
    assert isinstance(error.value.__cause__, ValidationError)
    assert "lots" not in str(error.value)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_validation_retry_can_recover(monkeypatch, asynchronous):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json=output(
                value={"total": "private-invalid-value"} if len(seen) == 1 else {"total": 42}
            ),
        )

    serve(monkeypatch, handler)
    model = Model("openai:test", api_key="test-key", validation_retries=1)
    result = (
        await extract_async(Invoice, model, PDF_BYTES, "Use EUR.")
        if asynchronous
        else extract(Invoice, model, PDF_BYTES, "Use EUR.")
    )
    assert result.total == 42
    assert len(seen) == 2
    instructions = json.loads(seen[1].content)["instructions"]
    assert instructions.startswith("Use EUR.")
    assert "total" in instructions
    assert "private-invalid-value" not in instructions


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_http_failure_raises_model_error(monkeypatch, asynchronous):
    serve(monkeypatch, lambda request: httpx.Response(503, text="private server details"))
    with pytest.raises(ModelError, match="HTTP 503") as error:
        if asynchronous:
            await extract_async(Invoice, configured(), PDF_BYTES)
        else:
            extract(Invoice, configured(), PDF_BYTES)
    assert isinstance(error.value.__cause__, httpx.HTTPStatusError)
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    "payload", [None, [], "text", {"output": []}, {"status": "completed", "output": [None]}]
)
async def test_malformed_envelope_raises_schema_error(payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(SchemaValidationError):
            await extract_async(Invoice, configured(), b"text", client=client)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_custom_client_remains_open(asynchronous):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=output()))
    if asynchronous:
        async with httpx.AsyncClient(transport=transport) as client:
            assert (await extract_async(Invoice, configured(), b"text", client=client)).total == 42
            assert not client.is_closed
    else:
        with httpx.Client(transport=transport) as client:
            assert extract(Invoice, configured(), b"text", client=client).total == 42
            assert not client.is_closed


def test_aliases_defaults_and_nested_validation(monkeypatch):
    class Item(BaseModel):
        amount: float = Field(gt=0)

    class Receipt(BaseModel):
        items: list[Item]
        currency: str = Field(default="USD", alias="currency_code")

    record(monkeypatch, payload=output(value={"items": [{"amount": 4.2}], "currency_code": "EUR"}))
    result = extract(Receipt, configured(), b"text")
    assert result.items[0].amount == 4.2
    assert result.currency == "EUR"


def test_missing_local_file():
    with pytest.raises(FileNotFoundError):
        extract(Invoice, configured(), "/missing/invoice.pdf")
