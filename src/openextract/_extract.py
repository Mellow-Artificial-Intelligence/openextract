"""Schema + model + source → a validated instance of that schema."""

from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal, overload

import httpx
from pydantic import BaseModel, ValidationError

from . import _decisions, _openrouter_extractions, _providers, _transport
from ._content import PROMPT, Source, load, load_async
from ._diagnostics import invalid_response_message, validation_details, validation_message
from ._openrouter_extractions import ExtractionResult
from .exceptions import SchemaValidationError
from .models import Model, ResolvedModel, resolve

type _Adapter = tuple[Callable[..., tuple[str, dict[str, Any]]], Callable[..., Any]]

# Malformed or schema-invalid output. Pydantic's ValidationError is a ValueError.
_OUTPUT_ERRORS = (LookupError, TypeError, ValueError, AttributeError)


@overload
def extract[T: BaseModel](
    schema: type[T],
    model: str | Model,
    source: Source,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    client: httpx.Client | None = None,
    include_metadata: Literal[False] = False,
) -> T: ...


@overload
def extract[T: BaseModel](
    schema: type[T],
    model: str | Model,
    source: Source,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    client: httpx.Client | None = None,
    include_metadata: Literal[True],
) -> ExtractionResult[T]: ...


def extract[T: BaseModel](
    schema: type[T],
    model: str | Model,
    source: Source,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    client: httpx.Client | None = None,
    include_metadata: bool = False,
) -> T | ExtractionResult[T]:
    """Extract ``schema`` from plain text, a path object, a URL, or raw bytes.

    Strings are UTF-8 text unless they start with ``http://`` or ``https://``.
    Use ``pathlib.Path`` or another ``os.PathLike`` object for local files.
    Text defaults to ``text/plain``. For files, URLs, and bytes, the media type
    is inferred from the name, URL headers, or content; ``media_type`` overrides it.

    With OpenRouter Extractions, ``include_metadata=True`` returns an
    ``ExtractionResult`` with the validated data and provider metadata.
    """
    resolved = resolve(model)
    build_request, parse_response = _adapter(resolved, include_metadata)
    content = load(source, media_type)
    guidance = instructions
    attempt = 0
    context = nullcontext(client) if client is not None else httpx.Client()
    with context as session:
        while True:
            attempt += 1
            path, body = build_request(schema, resolved, content, guidance)
            try:
                payload = _transport.post(resolved, path, body, session)
                return parse_response(schema, resolved, payload)
            except _OUTPUT_ERRORS as exc:
                guidance = _feedback(exc, attempt, resolved, instructions)


@overload
async def extract_async[T: BaseModel](
    schema: type[T],
    model: str | Model,
    source: Source,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    client: httpx.AsyncClient | None = None,
    include_metadata: Literal[False] = False,
) -> T: ...


@overload
async def extract_async[T: BaseModel](
    schema: type[T],
    model: str | Model,
    source: Source,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    client: httpx.AsyncClient | None = None,
    include_metadata: Literal[True],
) -> ExtractionResult[T]: ...


async def extract_async[T: BaseModel](
    schema: type[T],
    model: str | Model,
    source: Source,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    client: httpx.AsyncClient | None = None,
    include_metadata: bool = False,
) -> T | ExtractionResult[T]:
    """Async version of :func:`extract`."""
    resolved = resolve(model)
    build_request, parse_response = _adapter(resolved, include_metadata)
    content = await load_async(source, media_type)
    guidance = instructions
    attempt = 0
    context = nullcontext(client) if client is not None else httpx.AsyncClient()
    async with context as session:
        while True:
            attempt += 1
            path, body = build_request(schema, resolved, content, guidance)
            try:
                payload = await _transport.post_async(resolved, path, body, session)
                return parse_response(schema, resolved, payload)
            except _OUTPUT_ERRORS as exc:
                guidance = _feedback(exc, attempt, resolved, instructions)


def _adapter(model: ResolvedModel, include_metadata: bool) -> _Adapter:
    """Return the request builder and response parser that speak the model's API."""
    if model.api == "openrouter-extractions":
        parse = partial(_openrouter_extractions.parse, include_metadata=include_metadata)
        return _openrouter_extractions.request, parse
    if include_metadata:
        raise ValueError("include_metadata=True requires the OpenRouter Extractions API.")
    family = _decisions if model.api.endswith("-decisions") else _providers
    return family.request, family.parse


def _feedback(
    error: Exception, attempt: int, model: ResolvedModel, instructions: str | None
) -> str:
    """Return instructions for the next validation attempt, or raise the final failure."""
    is_decisions = model.api.endswith("-decisions")
    if not isinstance(error, ValidationError):
        context = "decision answers" if is_decisions else "model response"
        raise SchemaValidationError(invalid_response_message(error, context=context)) from error
    retries = 0 if is_decisions else model.config.validation_retries
    if attempt > retries:
        raise SchemaValidationError(validation_message(error, attempt)) from error
    # The details name the failing fields without echoing the provider's raw output.
    return (
        f"{instructions or PROMPT}\nCorrect these output validation errors:\n"
        f"{validation_details(error)}"
    )
