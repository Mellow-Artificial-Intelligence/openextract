"""Fill choice and yes/no schemas with an OpenRouter decisions model, such as TypeSafe's Jev."""

from __future__ import annotations

import os
import typing
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any, Literal, TypeGuard

import httpx
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_ai import BinaryContent

from .exceptions import ExtractionError, ModelError, SchemaValidationError

_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
_MODEL_PREFIXES = ("openrouter:typesafe/", "openrouter:~typesafe/")
_TIMEOUT_SECONDS = 60.0
_YES_THRESHOLD = 0.5


def is_decisions_model(model: object) -> TypeGuard[str]:
    """Decisions models answer typed questions and cannot serve chat completions."""
    return isinstance(model, str) and model.startswith(_MODEL_PREFIXES)


def decide[T: BaseModel](
    schema: type[T], model: str, content: BinaryContent, prompt: str, instructions: str | None
) -> T:
    body = _request_body(schema, model, content, prompt, instructions)
    with _request_errors():
        response = httpx.post(
            _DECISIONS_URL, headers=_headers(), json=body, timeout=_TIMEOUT_SECONDS
        )
    return _parse(schema, response)


async def decide_async[T: BaseModel](
    schema: type[T], model: str, content: BinaryContent, prompt: str, instructions: str | None
) -> T:
    body = _request_body(schema, model, content, prompt, instructions)
    with _request_errors():
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(_DECISIONS_URL, headers=_headers(), json=body)
    return _parse(schema, response)


def _request_body(
    schema: type[BaseModel],
    model: str,
    content: BinaryContent,
    prompt: str,
    instructions: str | None,
) -> dict[str, Any]:
    return {
        "model": model.removeprefix("openrouter:"),
        "state": [prompt, *([instructions] if instructions else []), _document(content)],
        "questions": {name: _question(name, field) for name, field in schema.model_fields.items()},
    }


def _document(content: BinaryContent) -> str | dict[str, Any]:
    if content.media_type.startswith("text/"):
        return content.data.decode(errors="replace")
    if content.media_type.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": content.data_uri}}
    raise ValueError(f"Decisions models read text and images, not {content.media_type}.")


def _question(name: str, field: FieldInfo) -> dict[str, Any]:
    instructions = field.description or name.replace("_", " ")
    if field.annotation is bool:
        return {"type": "noul", "instructions": instructions}
    criteria = {key: key for key in _choices(name, field.annotation)}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _choices(name: str, annotation: object) -> dict[str, object]:
    """Map each option's wire key to the value the schema expects."""
    if typing.get_origin(annotation) is Literal:
        return {str(value): value for value in typing.get_args(annotation)}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {str(member.value): member for member in annotation}
    raise TypeError(
        f"Decisions models only fill bool, Literal, and Enum fields; {name!r} is {annotation!r}."
    )


def _headers() -> dict[str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ExtractionError("Set OPENROUTER_API_KEY to use a decisions model.")
    return {"Authorization": f"Bearer {api_key}"}


def _parse[T: BaseModel](schema: type[T], response: httpx.Response) -> T:
    if response.is_error:
        raise ModelError(f"OpenRouter returned {response.status_code}: {response.text}")
    answers = response.json().get("answers", {})
    try:
        values = {
            name: _answer(name, field, answers[name]) for name, field in schema.model_fields.items()
        }
        return schema.model_validate(values)
    except (KeyError, ValidationError) as exc:
        raise SchemaValidationError(f"The answers did not match the schema: {exc}") from exc


def _answer(name: str, field: FieldInfo, answer: dict[str, Any]) -> object:
    if field.annotation is bool:
        return answer["noul"] >= _YES_THRESHOLD
    return _choices(name, field.annotation)[answer["choice"]]


@contextmanager
def _request_errors() -> Iterator[None]:
    try:
        yield
    except httpx.HTTPError as exc:
        raise ModelError(f"The decisions request failed: {exc}") from exc
