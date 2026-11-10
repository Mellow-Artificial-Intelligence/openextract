"""Turn files into validated Pydantic models over HTTP."""

from __future__ import annotations

import json
import mimetypes
import os
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from . import _decisions, _openai_decisions, _providers, _transport
from ._content import Content
from ._validation_errors import invalid_response_message, validation_details, validation_message
from .exceptions import (
    ExtractionError,
    SchemaValidationError,
)
from .models import Model, ResolvedModel, resolve

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
    client: httpx.Client | None = None,
) -> T:
    """Extract ``schema`` from a file path, ``http(s)://`` URL, or raw bytes.

    A string ``source`` is always a path or URL, never the document's text;
    pass text you already have as ``text.encode()``. The media type is
    inferred from the file name, the URL's ``Content-Type``, or the content
    itself; pass ``media_type`` to override it.
    """
    resolved = resolve(model)
    content = _load(source, media_type)
    path, body = _prepare(schema, resolved, content, instructions)
    with nullcontext(client) if client is not None else httpx.Client() as session:
        for attempt in range(resolved.config.validation_retries + 1):
            try:
                payload = _transport.post(resolved, path, body, session)
                return _validate(schema, resolved, payload)
            except ValidationError as exc:
                if attempt == resolved.config.validation_retries:
                    raise SchemaValidationError(validation_message(exc, attempt + 1)) from exc
                path, body = _prepare(schema, resolved, content, _feedback(instructions, exc))
            except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
                raise SchemaValidationError(invalid_response_message(exc)) from exc
    raise AssertionError("Unreachable")  # pragma: no cover


async def extract_async[T: BaseModel](
    schema: type[T],
    model: str | Model,
    source: Source,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> T:
    """Async version of :func:`extract`."""
    resolved = resolve(model)
    content = await _load_async(source, media_type)
    path, body = _prepare(schema, resolved, content, instructions)
    async with nullcontext(client) if client is not None else httpx.AsyncClient() as session:
        for attempt in range(resolved.config.validation_retries + 1):
            try:
                payload = await _transport.post_async(resolved, path, body, session)
                return _validate(schema, resolved, payload)
            except ValidationError as exc:
                if attempt == resolved.config.validation_retries:
                    raise SchemaValidationError(validation_message(exc, attempt + 1)) from exc
                path, body = _prepare(schema, resolved, content, _feedback(instructions, exc))
            except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
                raise SchemaValidationError(invalid_response_message(exc)) from exc
    raise AssertionError("Unreachable")  # pragma: no cover


def _prepare(
    schema: type[BaseModel], model: ResolvedModel, content: Content, instructions: str | None
) -> tuple[str, dict[str, Any]]:
    if model.api == "openai-decisions":
        body = _openai_decisions._request_body(
            schema, model.model_name, content, _PROMPT, instructions
        )
    elif model.api == "openrouter-decisions":
        body = _decisions._request_body(schema, model.model_name, content, _PROMPT, instructions)
    else:
        return _providers.build(model, schema.model_json_schema(), content, _PROMPT, instructions)
    if model.config.settings:
        raise ValueError("Model settings are not supported by decisions adapters.")
    return "/decisions", body


def _validate[T: BaseModel](schema: type[T], model: ResolvedModel, payload: dict[str, Any]) -> T:
    if model.api == "openai-decisions":
        return _openai_decisions._parse(schema, payload)
    if model.api == "openrouter-decisions":
        return _decisions._parse(schema, payload)
    output = _providers.parse(model, payload)
    if isinstance(output, dict):
        # Tool inputs are decoded JSON; retain JSON validation semantics for strict types.
        output = json.dumps(output)
    return schema.model_validate_json(output)


def _feedback(instructions: str | None, error: ValidationError) -> str:
    # Do not echo the provider's raw output back in error messages or retry instructions.
    return (
        f"{instructions or _PROMPT}\nCorrect these output validation errors:\n"
        f"{validation_details(error)}"
    )


def _is_url(source: Source) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _load(source: Source, media_type: str | None) -> Content:
    if _is_url(source):
        with _fetch_errors():
            response = httpx.get(str(source), follow_redirects=True, timeout=_URL_TIMEOUT_SECONDS)
            return _from_response(response, media_type)
    if isinstance(source, bytes):
        return _to_content(source, media_type)
    path = os.fspath(source)
    return _to_content(Path(path).read_bytes(), media_type, name=path)


async def _load_async(source: Source, media_type: str | None) -> Content:
    if not _is_url(source):
        return _load(source, media_type)
    with _fetch_errors():
        async with httpx.AsyncClient(follow_redirects=True, timeout=_URL_TIMEOUT_SECONDS) as client:
            response = await client.get(str(source))
        return _from_response(response, media_type)


def _from_response(response: httpx.Response, media_type: str | None) -> Content:
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
) -> Content:
    guessed = mimetypes.guess_type(name)[0] if name else None
    resolved = media_type or guessed or header or _sniff(data)
    if resolved is None:
        raise ValueError("Could not infer the media type; pass media_type explicitly.")
    return Content(data=data, media_type=resolved)


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


@contextmanager
def _fetch_errors() -> Generator[None]:
    try:
        yield
    except httpx.HTTPError as exc:
        detail = (
            f"HTTP {exc.response.status_code}"
            if isinstance(exc, httpx.HTTPStatusError)
            else type(exc).__name__
        )
        raise ExtractionError(
            f"Could not fetch the source URL ({detail}).\n"
            "Next step: Check the URL and network access. If the source needs authentication "
            "or a custom timeout, download it with your own HTTP client "
            "and pass its bytes to extract()."
        ) from exc
