"""Fill choice and yes/no schemas using OpenAI's Decisions API."""

from __future__ import annotations

import os
from typing import Any, TypeGuard

import httpx
from pydantic import BaseModel, ValidationError
from pydantic_ai import BinaryContent

from ._decisions import _choices, _request_errors
from .exceptions import ExtractionError, ModelError, SchemaValidationError

_PREFIX = "openai-decisions:"
_URL = "https://api.openai.com/v1/decisions"
_TIMEOUT_SECONDS = 60.0


def is_openai_decisions_model(model: object) -> TypeGuard[str]:
    return isinstance(model, str) and model.startswith(_PREFIX)


def decide[T: BaseModel](
    schema: type[T], model: str, content: BinaryContent, prompt: str, instructions: str | None
) -> T:
    body = _request_body(schema, model, content, prompt, instructions)
    with _request_errors():
        response = httpx.post(_URL, headers=_headers(), json=body, timeout=_TIMEOUT_SECONDS)
    return _parse(schema, response)


async def decide_async[T: BaseModel](
    schema: type[T], model: str, content: BinaryContent, prompt: str, instructions: str | None
) -> T:
    body = _request_body(schema, model, content, prompt, instructions)
    with _request_errors():
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(_URL, headers=_headers(), json=body)
    return _parse(schema, response)


def _request_body(
    schema: type[BaseModel],
    model: str,
    content: BinaryContent,
    prompt: str,
    instructions: str | None,
) -> dict[str, Any]:
    if not 1 <= len(schema.model_fields) <= 400:
        raise ValueError("OpenAI Decisions requires between 1 and 400 schema fields.")
    questions = []
    for name, field in schema.model_fields.items():
        question: dict[str, Any] = {
            "name": name,
            "instructions": field.description or name.replace("_", " "),
        }
        if field.annotation is bool:
            question["type"] = "predicate"
        else:
            question["type"] = "choice"
            question["choices"] = [
                {"value": key, "description": key} for key in _choices(name, field.annotation)
            ]
        questions.append(question)

    if content.media_type.startswith("text/"):
        document = {"type": "input_text", "text": content.data.decode(errors="replace")}
    elif content.media_type.startswith("image/"):
        document = {"type": "input_image", "image_url": content.data_uri}
    else:
        raise ValueError(f"Decisions models read text and images, not {content.media_type}.")
    messages: list[dict[str, Any]] = [
        {"role": "developer", "content": instructions or prompt},
        {"role": "user", "content": [{"type": "input_text", "text": prompt}, document]},
    ]
    return {"model": model.removeprefix(_PREFIX), "input": messages, "questions": questions}


def _headers() -> dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ExtractionError("Set OPENAI_API_KEY to use an OpenAI decisions model.")
    return {"Authorization": f"Bearer {api_key}"}


def _parse[T: BaseModel](schema: type[T], response: httpx.Response) -> T:
    if response.is_error:
        raise ModelError(f"OpenAI returned {response.status_code}: {response.text}")
    try:
        answers = response.json()["answers"]
        if not isinstance(answers, list):
            raise ValueError("Expected an answers array.")
        values = {}
        for answer in answers:
            name = answer["name"]
            if name in values:
                raise ValueError(f"Duplicate answer for {name!r}.")
            field = schema.model_fields[name]
            if answer["type"] == "refusal":
                raise ValueError(f"The model refused to answer {name!r}.")
            if field.annotation is bool:
                probability = answer["probability"]
                if (
                    answer["type"] != "predicate"
                    or type(probability) not in (int, float)
                    or not 0 <= probability <= 1
                ):
                    raise ValueError(f"Invalid predicate answer for {name!r}.")
                values[name] = probability >= 0.5
            else:
                if answer["type"] != "choice":
                    raise ValueError(f"Expected a choice answer for {name!r}.")
                values[name] = _choices(name, field.annotation)[answer["choice"]]
        if values.keys() != schema.model_fields.keys():
            raise ValueError("Not every schema field received an answer.")
        return schema.model_validate(values)
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise SchemaValidationError(f"The answers did not match the schema: {exc}") from exc
