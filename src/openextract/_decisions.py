"""Fill ``bool``, ``Literal``, and ``Enum`` schemas through closed-set Decisions APIs.

Each field becomes one question. OpenRouter and Gemini share a named question and
answer format; OpenAI uses lists and its own answer types.
"""

import re
import typing
from enum import Enum
from typing import Any, Literal, TypeGuard

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from ._content import PROMPT, Content
from ._diagnostics import invalid_response_message, validation_message
from .exceptions import SchemaValidationError
from .models import ResolvedModel

_YES_THRESHOLD = 0.5
_GEMINI_IMAGES = {
    f"image/{subtype}" for subtype in ("png", "jpeg", "webp", "heic", "heif", "gif", "bmp", "tiff")
}


def request(
    schema: type[BaseModel], model: ResolvedModel, content: Content, instructions: str | None
) -> tuple[str, dict[str, Any]]:
    body = _BUILDERS[model.api](schema, content, instructions)
    if model.config.settings:
        raise ValueError("Model settings are not supported by decisions adapters.")
    return "/decisions", {"model": model.model_name, **body}


def parse[T: BaseModel](schema: type[T], model: ResolvedModel, payload: dict[str, Any]) -> T:
    """Validate one answer per field. Invalid answers fail without a validation retry."""
    is_openai = model.api == "openai-decisions"
    fields = schema.model_fields
    try:
        answers = _answer_map(payload["answers"], is_openai=is_openai)
        if answers.keys() != fields.keys():
            raise ValueError("Expected exactly one answer for every schema field.")
        values = {
            name: _parse_answer(name, fields[name], answer, is_openai=is_openai)
            for name, answer in answers.items()
        }
        return schema.model_validate(values, by_name=True)
    except ValidationError as exc:
        raise SchemaValidationError(validation_message(exc, 1)) from exc
    except (LookupError, TypeError, ValueError, AttributeError) as exc:
        message = invalid_response_message(exc, context="decision answers")
        raise SchemaValidationError(message) from exc


def _build_openrouter(
    schema: type[BaseModel], content: Content, instructions: str | None
) -> dict[str, Any]:
    _require_text_or_image(content)
    document: str | dict[str, Any] = (
        content.text
        if content.is_text
        else {"type": "image_url", "image_url": {"url": content.data_uri}}
    )
    return {
        "state": [*_guidance(instructions), document],
        "questions": _named_questions(schema),
    }


def _build_gemini(
    schema: type[BaseModel], content: Content, instructions: str | None
) -> dict[str, Any]:
    questions = _named_questions(schema)
    _check_gemini_limits(questions)
    if content.is_text:
        document = {"type": "text", "text": content.text}
    elif content.is_pdf or content.media_type in _GEMINI_IMAGES:
        kind = "document" if content.is_pdf else "image"
        document = {"type": kind, "data": content.base64, "mime_type": content.media_type}
    else:
        raise ValueError(
            f"Gemini Decisions supports text, images and PDF, not {content.media_type}."
        )
    texts = [{"type": "text", "text": text + "\n"} for text in _guidance(instructions)]
    return {"input": [*texts, document], "questions": questions}


def _build_openai(
    schema: type[BaseModel], content: Content, instructions: str | None
) -> dict[str, Any]:
    if not 1 <= len(schema.model_fields) <= 400:
        raise ValueError("OpenAI Decisions requires between 1 and 400 schema fields.")
    questions = [_openai_question(name, field) for name, field in schema.model_fields.items()]
    _require_text_or_image(content)
    document = (
        {"type": "input_text", "text": content.text}
        if content.is_text
        else {"type": "input_image", "image_url": content.data_uri}
    )
    return {
        "input": [
            {"role": "developer", "content": instructions or PROMPT},
            {"role": "user", "content": [{"type": "input_text", "text": PROMPT}, document]},
        ],
        "questions": questions,
    }


def _check_gemini_limits(questions: dict[str, dict[str, Any]]) -> None:
    if not 1 <= len(questions) <= 1000:
        raise ValueError("Gemini Decisions requires between 1 and 1000 schema fields.")
    for name, question in questions.items():
        options = question.get("criteria")
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None:
            raise ValueError(
                "Gemini Decisions field names must be 1–64 ASCII letters, digits, _ or -."
            )
        if options is not None and not 2 <= len(options) <= 255:
            raise ValueError("Gemini Decisions requires between 2 and 255 choices per field.")
        if options is not None and not all(map(_is_gemini_option, options)):
            raise ValueError("Invalid Gemini Decisions option key.")


def _named_questions(schema: type[BaseModel]) -> dict[str, dict[str, Any]]:
    return {name: _named_question(name, field) for name, field in schema.model_fields.items()}


def _guidance(instructions: str | None) -> list[str]:
    return [PROMPT, instructions] if instructions else [PROMPT]


def _named_question(name: str, field: FieldInfo) -> dict[str, Any]:
    """Phrase a field in the named format OpenRouter and Gemini share."""
    instructions = _field_instructions(name, field)
    if field.annotation is bool:
        return {"type": "noul", "instructions": instructions}
    criteria = {key: key for key in _options(name, field.annotation)}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _openai_question(name: str, field: FieldInfo) -> dict[str, Any]:
    instructions = _field_instructions(name, field)
    if field.annotation is bool:
        return {"name": name, "type": "predicate", "instructions": instructions}
    choices = [{"value": key, "description": key} for key in _options(name, field.annotation)]
    return {"name": name, "type": "choice", "instructions": instructions, "choices": choices}


def _field_instructions(name: str, field: FieldInfo) -> str:
    return field.description or name.replace("_", " ")


def _options(name: str, annotation: object) -> dict[str, object]:
    """Map each option's wire key to the value the schema expects."""
    if typing.get_origin(annotation) is Literal:
        return {str(value): value for value in typing.get_args(annotation)}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {str(member.value): member for member in annotation}
    raise TypeError(
        f"Decisions models only fill bool, Literal, and Enum fields; {name!r} is {annotation!r}."
    )


def _is_gemini_option(key: str) -> bool:
    """Google allows 1–128 trimmed printable characters in at most 512 UTF-8 bytes."""
    return (
        1 <= len(key) <= 128
        and len(key.encode()) <= 512
        and key == key.strip()
        and key.isprintable()
    )


def _require_text_or_image(content: Content) -> None:
    if not (content.is_text or content.is_image):
        raise ValueError(f"Decisions models read text and images, not {content.media_type}.")


def _answer_map(raw_answers: Any, *, is_openai: bool) -> dict[str, dict[str, Any]]:
    """Normalize OpenAI's answer list or OpenRouter/Gemini's answer map into a field map."""
    if not is_openai:
        if not isinstance(raw_answers, dict):
            raise TypeError("Expected an object of decision answers.")
        return raw_answers
    if not isinstance(raw_answers, list):
        raise TypeError("Expected a list of decision answers.")
    answers = {item["name"]: item for item in raw_answers}
    if len(answers) != len(raw_answers):
        raise ValueError("Expected at most one answer per schema field.")
    return answers


def _parse_answer(
    name: str, field: FieldInfo, answer: dict[str, Any], *, is_openai: bool
) -> object:
    """Turn an answer into the field's value; yes/no is true from a probability of one half."""
    if answer["type"] == "refusal":
        raise ValueError(f"The model refused to answer {name!r}.")
    if field.annotation is bool:
        yes_type, prob_key = ("predicate", "probability") if is_openai else ("noul", "noul")
        chance = answer[prob_key]
        if not (answer["type"] == yes_type and _is_probability(chance)):
            raise ValueError(
                f"Expected a yes/no answer with a numeric probability from 0 to 1 for {name!r}."
            )
        return chance >= _YES_THRESHOLD
    options = _options(name, field.annotation)
    if answer["type"] != "choice":
        raise ValueError(f"Expected a choice answer for {name!r}.")
    if not isinstance(answer["choice"], str) or answer["choice"] not in options:
        raise ValueError(f"Expected one of {list(options)!r} for field {name!r}.")
    return options[answer["choice"]]


def _is_probability(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


_BUILDERS = {
    "openai-decisions": _build_openai,
    "gemini-decisions": _build_gemini,
    "openrouter-decisions": _build_openrouter,
}
