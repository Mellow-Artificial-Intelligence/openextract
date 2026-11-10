"""Fill choice and yes/no schemas with an OpenRouter decisions model, such as TypeSafe's Jev."""

from __future__ import annotations

import typing
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from ._content import Content
from ._validation_errors import invalid_response_message, validation_message
from .exceptions import SchemaValidationError

_YES_THRESHOLD = 0.5


def _request_body(
    schema: type[BaseModel],
    model: str,
    content: Content,
    prompt: str,
    instructions: str | None,
) -> dict[str, Any]:
    return {
        "model": model.removeprefix("openrouter:"),
        "state": [prompt, *([instructions] if instructions else []), _document(content)],
        "questions": {name: _question(name, field) for name, field in schema.model_fields.items()},
    }


def _document(content: Content) -> str | dict[str, Any]:
    if content.is_text:
        return content.text
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


def _parse[T: BaseModel](schema: type[T], payload: dict[str, Any]) -> T:
    try:
        answers = payload["answers"]
        if not isinstance(answers, dict) or answers.keys() != schema.model_fields.keys():
            raise ValueError("Expected exactly one answer for every schema field.")
        values = {
            name: _answer(name, field, answers[name]) for name, field in schema.model_fields.items()
        }
        return schema.model_validate(values, by_name=True)
    except ValidationError as exc:
        raise SchemaValidationError(validation_message(exc, 1)) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaValidationError(
            invalid_response_message(exc, context="decision answers")
        ) from exc


def _choice(name: str, annotation: object, value: object) -> object:
    choices = _choices(name, annotation)
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"Expected one of {list(choices)!r} for field {name!r}.")
    return choices[value]


def _answer(name: str, field: FieldInfo, answer: dict[str, Any]) -> object:
    if field.annotation is bool:
        probability = answer["noul"]
        if (
            answer["type"] != "noul"
            or type(probability) not in (int, float)
            or not 0 <= probability <= 1
        ):
            raise ValueError(
                f"Expected a yes/no answer with a numeric probability from 0 to 1 for {name!r}."
            )
        return probability >= _YES_THRESHOLD
    if answer["type"] != "choice":
        raise ValueError(f"Expected a choice answer for {name!r}.")
    return _choice(name, field.annotation, answer["choice"])
