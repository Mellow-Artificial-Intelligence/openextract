"""Fill choice and yes/no schemas with an OpenRouter decisions model, such as TypeSafe's Jev."""

from __future__ import annotations

import typing
from enum import Enum
from typing import Any, Literal, cast

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from ._content import Content
from ._validation_errors import invalid_response_message, validation_message
from .exceptions import SchemaValidationError

_YES_THRESHOLD = 0.5


def request_body(
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
    if check_media(content):
        return content.text
    return {"type": "image_url", "image_url": {"url": content.data_uri}}


def _question(name: str, field: FieldInfo) -> dict[str, Any]:
    instructions = field_instructions(name, field)
    if field.annotation is bool:
        return {"type": "noul", "instructions": instructions}
    criteria = {key: key for key in choices(name, field.annotation)}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def check_media(content: Content) -> bool:
    """Return whether ``content`` is text; raise unless it is text or an image."""
    if content.is_text:
        return True
    if content.media_type.startswith("image/"):
        return False
    raise ValueError(f"Decisions models read text and images, not {content.media_type}.")


def field_instructions(name: str, field: FieldInfo) -> str:
    return field.description or name.replace("_", " ")


def is_yes(
    name: str, answer_type: object, probability: object, *, expected_type: str, label: str
) -> bool:
    """Validate a yes/no answer's probability and apply the yes threshold."""
    numeric = type(probability) in (int, float)
    value = cast(float, probability)
    if answer_type != expected_type or not numeric or not 0 <= value <= 1:
        raise ValueError(
            f"Expected a {label} answer with a numeric probability from 0 to 1 for {name!r}."
        )
    return value >= _YES_THRESHOLD


def choices(name: str, annotation: object) -> dict[str, object]:
    """Map each option's wire key to the value the schema expects."""
    if typing.get_origin(annotation) is Literal:
        return {str(value): value for value in typing.get_args(annotation)}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {str(member.value): member for member in annotation}
    raise TypeError(
        f"Decisions models only fill bool, Literal, and Enum fields; {name!r} is {annotation!r}."
    )


def parse[T: BaseModel](schema: type[T], payload: dict[str, Any]) -> T:
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


def choice(name: str, annotation: object, value: object) -> object:
    options = choices(name, annotation)
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"Expected one of {list(options)!r} for field {name!r}.")
    return options[value]


def _answer(name: str, field: FieldInfo, answer: dict[str, Any]) -> object:
    if field.annotation is bool:
        return is_yes(name, answer["type"], answer["noul"], expected_type="noul", label="yes/no")
    if answer["type"] != "choice":
        raise ValueError(f"Expected a choice answer for {name!r}.")
    return choice(name, field.annotation, answer["choice"])
