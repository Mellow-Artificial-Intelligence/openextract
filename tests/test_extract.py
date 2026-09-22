from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel
from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

import openextract._extract as extract_module
from openextract import (
    ExtractionError,
    ModelError,
    ProviderNotInstalledError,
    SchemaValidationError,
    extract,
    extract_async,
)

PDF_BYTES = b"%PDF-1.7 fake"


class Invoice(BaseModel):
    total: float


def output_call(info: AgentInfo, args: dict) -> ToolCallPart:
    return ToolCallPart(info.output_tools[0].name, args)


def capturing_model(seen: list[BinaryContent]) -> FunctionModel:
    """A model that records the file it was sent and returns a valid invoice."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = messages[0].parts[-1].content
        seen.extend(item for item in prompt if isinstance(item, BinaryContent))
        return ModelResponse(parts=[output_call(info, {"total": 42.0})])

    return FunctionModel(respond)


def serve(monkeypatch, handler) -> None:
    """Route every httpx request made by openextract through ``handler``."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    real_async_client = httpx.AsyncClient

    def fake_get(url, **kwargs):
        with real_client(transport=transport, **kwargs) as client:
            return client.get(url)

    monkeypatch.setattr(extract_module.httpx, "get", fake_get)
    monkeypatch.setattr(
        extract_module.httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(transport=transport, **kwargs),
    )


def test_extracts_from_path(tmp_path: Path):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(PDF_BYTES)
    seen: list[BinaryContent] = []

    result = extract(Invoice, capturing_model(seen), path, "Find the total.")

    assert result == Invoice(total=42.0)
    assert seen[0].media_type == "application/pdf"
    assert seen[0].data == PDF_BYTES


async def test_extract_async_from_bytes():
    seen: list[BinaryContent] = []

    result = await extract_async(Invoice, capturing_model(seen), b"\x89PNG....")

    assert result == Invoice(total=42.0)
    assert seen[0].media_type == "image/png"


@pytest.mark.parametrize(
    ("data", "media_type"),
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
def test_sniffs_media_type(data: bytes, media_type: str | None):
    assert extract_module._sniff(data) == media_type


def test_explicit_media_type_wins():
    seen: list[BinaryContent] = []

    extract(Invoice, capturing_model(seen), PDF_BYTES, media_type="application/x-custom")

    assert seen[0].media_type == "application/x-custom"


def test_uninferable_media_type_raises():
    with pytest.raises(ValueError, match="media_type"):
        extract(Invoice, TestModel(), b"\xff\xfe\x00\x01")


def test_url_uses_content_type_header(monkeypatch):
    serve(
        monkeypatch,
        lambda request: httpx.Response(
            200, content=b"<p>hi</p>", headers={"content-type": "text/html; charset=utf-8"}
        ),
    )
    seen: list[BinaryContent] = []

    extract(Invoice, capturing_model(seen), "https://example.com/page")

    assert seen[0].media_type == "text/html"
    assert seen[0].data == b"<p>hi</p>"


async def test_url_prefers_extension_over_generic_header(monkeypatch):
    serve(
        monkeypatch,
        lambda request: httpx.Response(
            200, content=PDF_BYTES, headers={"content-type": "application/octet-stream"}
        ),
    )
    seen: list[BinaryContent] = []

    await extract_async(Invoice, capturing_model(seen), "https://example.com/invoice.pdf")

    assert seen[0].media_type == "application/pdf"


def test_url_failure_raises_extraction_error(monkeypatch):
    serve(monkeypatch, lambda request: httpx.Response(404))

    with pytest.raises(ExtractionError, match="Could not fetch"):
        extract(Invoice, TestModel(), "https://example.com/missing.pdf")


def test_invalid_output_raises_schema_validation_error():
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[output_call(info, {"total": "lots"})])

    with pytest.raises(SchemaValidationError):
        extract(Invoice, FunctionModel(respond), PDF_BYTES)


async def test_provider_error_raises_model_error():
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=503, model_name="test")

    with pytest.raises(ModelError):
        await extract_async(Invoice, FunctionModel(respond), PDF_BYTES)


def test_missing_provider_sdk_raises(monkeypatch):
    def missing_sdk(*args, **kwargs):
        raise ImportError("No module named 'openai'")

    monkeypatch.setattr(extract_module, "Agent", missing_sdk)

    with pytest.raises(ProviderNotInstalledError, match="openextract\\[openai\\]"):
        extract(Invoice, "openai:gpt-5", PDF_BYTES)


def test_model_string_routing(monkeypatch):
    built: list[tuple[object, object]] = []
    monkeypatch.setattr(
        extract_module,
        "Agent",
        lambda model, output_type, instructions: built.append((model, output_type)),
    )

    extract_module._build_agent(Invoice, "openai:gpt-5", None)
    extract_module._build_agent(Invoice, "ollama:llama3", None)
    extract_module._build_agent(Invoice, "anthropic:claude-sonnet-5", None)

    assert built[0] == ("openai-responses:gpt-5", Invoice)
    assert type(built[1][1]).__name__ == "NativeOutput"
    assert built[2] == ("anthropic:claude-sonnet-5", Invoice)


def test_text_model_output_is_not_used():
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("no tool call")])

    with pytest.raises(SchemaValidationError):
        extract(Invoice, FunctionModel(respond), PDF_BYTES)
