"""Turn any file into a validated Pydantic model with one LLM call."""

from __future__ import annotations

import mimetypes
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.output import NativeOutput

from .exceptions import (
    ExtractionError,
    ModelError,
    ProviderNotInstalledError,
    SchemaValidationError,
)

type Source = str | os.PathLike[str] | bytes

_PROMPT = "Extract the requested information from this file."
_URL_TIMEOUT_SECONDS = 30.0
_GENERIC_MEDIA_TYPES = {"", "application/octet-stream", "binary/octet-stream"}

# Leading-byte signatures, used when neither a file name nor a header names the type.
_SIGNATURES = {
    b"%PDF": "application/pdf",
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
    b"ID3": "audio/mpeg",
    b"fLaC": "audio/flac",
    b"OggS": "audio/ogg",
}
_RIFF_FORMATS = {b"WEBP": "image/webp", b"WAVE": "audio/wav"}


def extract[T: BaseModel](
    schema: type[T],
    model: str | Model,
    source: Source,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
) -> T:
    """Extract ``schema`` from a file path, ``http(s)://`` URL, or raw bytes.

    A string ``source`` is always a path or URL, never the document's text;
    pass text you already have as ``text.encode()``. The media type is
    inferred from the file name, the URL's ``Content-Type``, or the content
    itself; pass ``media_type`` to override it.
    """
    content = _load(source, media_type)
    with _model_errors():
        agent = _build_agent(schema, model, instructions)
        return cast(T, agent.run_sync([_PROMPT, content]).output)


async def extract_async[T: BaseModel](
    schema: type[T],
    model: str | Model,
    source: Source,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
) -> T:
    """Async version of :func:`extract`."""
    content = await _load_async(source, media_type)
    with _model_errors():
        agent = _build_agent(schema, model, instructions)
        result = await agent.run([_PROMPT, content])
        return cast(T, result.output)


def _is_url(source: Source) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _load(source: Source, media_type: str | None) -> BinaryContent:
    if _is_url(source):
        with _fetch_errors():
            response = httpx.get(str(source), follow_redirects=True, timeout=_URL_TIMEOUT_SECONDS)
            return _from_response(response, media_type)
    if isinstance(source, bytes):
        return _to_content(source, media_type)
    path = os.fspath(source)
    return _to_content(Path(path).read_bytes(), media_type, name=path)


async def _load_async(source: Source, media_type: str | None) -> BinaryContent:
    if not _is_url(source):
        return _load(source, media_type)
    with _fetch_errors():
        async with httpx.AsyncClient(follow_redirects=True, timeout=_URL_TIMEOUT_SECONDS) as client:
            response = await client.get(str(source))
        return _from_response(response, media_type)


def _from_response(response: httpx.Response, media_type: str | None) -> BinaryContent:
    response.raise_for_status()
    header = response.headers.get("content-type", "").split(";")[0].strip().lower()
    return _to_content(
        response.content,
        media_type,
        name=response.url.path,
        header=None if header in _GENERIC_MEDIA_TYPES else header,
    )


def _to_content(
    data: bytes,
    media_type: str | None,
    *,
    name: str | None = None,
    header: str | None = None,
) -> BinaryContent:
    guessed = mimetypes.guess_type(name)[0] if name else None
    resolved = media_type or guessed or header or _sniff(data)
    if resolved is None:
        raise ValueError("Could not infer the media type; pass media_type explicitly.")
    return BinaryContent(data=data, media_type=resolved)


def _sniff(data: bytes) -> str | None:
    """Guess a media type from the content, falling back to plain text for UTF-8."""
    for signature, media_type in _SIGNATURES.items():
        if data.startswith(signature):
            return media_type
    if data.startswith(b"RIFF"):
        return _RIFF_FORMATS.get(data[8:12])
    if data[4:8] == b"ftyp":
        return "video/mp4"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return "text/plain"


def _build_agent(schema: type[BaseModel], model: str | Model, instructions: str | None) -> Agent:
    output_type: object = schema
    if isinstance(model, str):
        if model.startswith("ollama:"):
            output_type = NativeOutput(schema)
        # Documents, audio, and video are best supported on the OpenAI Responses API.
        if model.startswith("openai:"):
            model = "openai-responses:" + model.removeprefix("openai:")
    try:
        return Agent(model, output_type=output_type, instructions=instructions)
    except ImportError as exc:
        raise ProviderNotInstalledError(
            f"The provider SDK for {model!r} is not installed ({exc}). "
            "Install its extra, e.g. pip install 'openextract[openai]', or 'openextract[all]'."
        ) from exc


@contextmanager
def _fetch_errors() -> Iterator[None]:
    try:
        yield
    except httpx.HTTPError as exc:
        raise ExtractionError(f"Could not fetch the URL: {exc}") from exc


@contextmanager
def _model_errors() -> Iterator[None]:
    try:
        yield
    except UnexpectedModelBehavior as exc:
        raise SchemaValidationError(str(exc)) from exc
    except ModelAPIError as exc:
        raise ModelError(str(exc)) from exc
