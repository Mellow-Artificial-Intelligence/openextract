"""Fill choice and yes/no schemas using OpenAI's Decisions API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from ._content import Content
from ._decisions import check_media, choice, choices, field_instructions, is_yes
from ._validation_errors import invalid_response_message, validation_message
from .exceptions import SchemaValidationError


def request_body(
    schema: type[BaseModel],
    model: str,
    content: Content,
    prompt: str,
    instructions: str | None,
) -> dict[str, Any]:
    if not 1 <= len(schema.model_fields) <= 400:
        raise ValueError("OpenAI Decisions requires between 1 and 400 schema fields.")
    questions = []
    for name, field in schema.model_fields.items():
        question: dict[str, Any] = {"name": name, "instructions": field_instructions(name, field)}
        if field.annotation is bool:
            question["type"] = "predicate"
        else:
            question["type"] = "choice"
            question["choices"] = [
                {"value": key, "description": key} for key in choices(name, field.annotation)
            ]
        questions.append(question)

    if check_media(content):
        document = {"type": "input_text", "text": content.text}
    else:
        document = {"type": "input_image", "image_url": content.data_uri}
    messages: list[dict[str, Any]] = [
        {"role": "developer", "content": instructions or prompt},
        {"role": "user", "content": [{"type": "input_text", "text": prompt}, document]},
    ]
    return {"model": model, "input": messages, "questions": questions}


def parse[T: BaseModel](schema: type[T], payload: dict[str, Any]) -> T:
    try:
        answers = payload["answers"]
        if not isinstance(answers, list):
            raise ValueError("Expected an answers array.")
        values = {}
        for answer in answers:
            name = answer["name"]
            if not isinstance(name, str) or name not in schema.model_fields:
                raise ValueError("Received an answer for an unknown schema field.")
            if name in values:
                raise ValueError(f"Duplicate answer for {name!r}.")
            field = schema.model_fields[name]
            if answer["type"] == "refusal":
                raise ValueError(f"The model refused to answer {name!r}.")
            if field.annotation is bool:
                values[name] = is_yes(
                    name,
                    answer["type"],
                    answer["probability"],
                    expected_type="predicate",
                    label="predicate",
                )
            else:
                if answer["type"] != "choice":
                    raise ValueError(f"Expected a choice answer for {name!r}.")
                values[name] = choice(name, field.annotation, answer["choice"])
        if values.keys() != schema.model_fields.keys():
            raise ValueError("Not every schema field received an answer.")
        return schema.model_validate(values, by_name=True)
    except ValidationError as exc:
        raise SchemaValidationError(validation_message(exc, 1)) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaValidationError(
            invalid_response_message(exc, context="decision answers")
        ) from exc
