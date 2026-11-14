"""Gemini Decisions REST encoding: https://aistudio.google.com/docs/decisions-api."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from ._content import Content
from ._decisions import question as schema_question

_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    "image/gif",
    "image/bmp",
    "image/tiff",
}


def request_body(
    schema: type[BaseModel],
    model: str,
    content: Content,
    prompt: str,
    instructions: str | None,
) -> dict[str, Any]:
    if not 1 <= len(schema.model_fields) <= 1000:
        raise ValueError("Gemini Decisions requires between 1 and 1000 schema fields.")
    questions = {}
    for name, field in schema.model_fields.items():
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None:
            raise ValueError(
                "Gemini Decisions field names must be 1–64 ASCII letters, digits, _ or -."
            )
        question = schema_question(name, field)
        if question["type"] == "choice":
            criteria = question["criteria"]
            if not 2 <= len(criteria) <= 255:
                raise ValueError("Gemini Decisions requires between 2 and 255 choices per field.")
            for key in criteria:
                if (
                    not 1 <= len(key) <= 128
                    or len(key.encode("utf-8")) > 512
                    or key != key.strip()
                    or any(
                        ord(char) < 32 or ord(char) == 127 or (char.isspace() and char != " ")
                        for char in key
                    )
                ):
                    raise ValueError("Invalid Gemini Decisions option key.")
        questions[name] = question

    if content.is_text:
        document = {"type": "text", "text": content.text}
    elif content.media_type in _IMAGE_TYPES or content.media_type == "application/pdf":
        document = {
            "type": "document" if content.media_type == "application/pdf" else "image",
            "data": content.base64,
            "mime_type": content.media_type,
        }
    else:
        raise ValueError(
            f"Gemini Decisions supports text, images and PDF, not {content.media_type}."
        )
    return {
        "model": model,
        "input": [
            {"type": "text", "text": prompt + "\n"},
            *([{"type": "text", "text": instructions + "\n"}] if instructions else []),
            document,
        ],
        "questions": questions,
    }
