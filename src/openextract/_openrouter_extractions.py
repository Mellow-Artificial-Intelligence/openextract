"""Typed field extraction through OpenRouter's alpha Extractions API."""

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from types import NoneType, UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ._content import Content
from .models import ResolvedModel, merge_settings

_SCALAR_TYPES = {
    str: "text",
    int: "number",
    float: "number",
    Decimal: "number",
    date: "date",
    time: "time",
    datetime: "datetime",
}
_LIST_ITEMS = ("text", "number", "date")
_ASSET_FORMATS = {
    "application/pdf": ("document", "pdf"),
    "image/jpeg": ("image", "jpeg"),
    "image/png": ("image", "png"),
    "image/webp": ("image", "webp"),
}
_REQUEST_FIELDS = {"model", "state", "assets", "fields"}


class OpenRouterField:
    """Set native Extractions field options in a Pydantic `Annotated` field."""

    def __init__(self, **options: Any) -> None:
        self.options = options


@dataclass(frozen=True)
class ExtractionResult[T: BaseModel]:
    """Validated data and the original OpenRouter answer and response metadata."""

    data: T
    answers: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def request(
    schema: type[BaseModel], model: ResolvedModel, content: Content, instructions: str | None
) -> tuple[str, dict[str, Any]]:
    if schema.__pydantic_root_model__:
        raise TypeError("OpenRouter Extractions requires a Pydantic model with named fields.")
    if not schema.model_fields:
        raise ValueError("OpenRouter Extractions requires at least one schema field.")
    fields = {
        name: _field_spec(name, field, instructions) for name, field in schema.model_fields.items()
    }
    body: dict[str, Any] = {"model": model.model_name, "fields": fields}
    merge_settings(body, model.config.settings, _REQUEST_FIELDS)
    if content.is_text:
        body["state"] = content.text
    elif asset := _ASSET_FORMATS.get(content.media_type):
        kind, file_format = asset
        document = {"type": kind, "format": file_format, "data_base64": content.base64}
        body["state"] = "document"
        body["assets"] = {"document": document}
    else:
        raise ValueError(
            "OpenRouter Extractions supports text, PDF, PNG, JPEG, and WebP. "
            f"Received {content.media_type}."
        )
    return "/extractions", body


def parse[T: BaseModel](
    schema: type[T], model: ResolvedModel, payload: dict[str, Any], *, include_metadata: bool
) -> T | ExtractionResult[T]:
    answers = payload.get("answers")
    if not isinstance(answers, dict) or answers.keys() != schema.model_fields.keys():
        raise ValueError("OpenRouter Extractions must return one answer for every schema field.")
    if not include_metadata and payload.get("truncated") is True:
        raise ValueError(
            "OpenRouter read only part of the source. Use include_metadata=True to inspect "
            "partial results or retry with a shorter source."
        )
    if not all(isinstance(answer, dict) and "value" in answer for answer in answers.values()):
        raise ValueError("Every OpenRouter Extractions answer must contain a value.")
    values = {name: answer["value"] for name, answer in answers.items()}
    data = schema.model_validate_json(json.dumps(values), by_alias=False, by_name=True)
    if not include_metadata:
        return data
    metadata = {key: value for key, value in payload.items() if key != "answers"}
    return ExtractionResult(data, answers, metadata)


def _field_spec(name: str, field: FieldInfo, instructions: str | None) -> dict[str, Any]:
    options = {}
    for metadata in field.metadata:
        if isinstance(metadata, OpenRouterField):
            options.update(metadata.options)
    spec = options if "type" in options else {**_infer_field(name, field.annotation), **options}
    spec.setdefault("instructions", field.description or name.replace("_", " "))
    if instructions:
        # Keep global instructions out of state so they don't shift source spans.
        spec["instructions"] = f"{spec['instructions']}\n{instructions}"
    return spec


def _infer_field(name: str, annotation: Any) -> dict[str, Any]:
    if get_origin(annotation) in (Union, UnionType):
        members = [member for member in get_args(annotation) if member is not NoneType]
        if len(members) == 1:
            annotation = members[0]
    annotation = _unannotated(annotation)
    if spec := _scalar_spec(annotation):
        return spec
    if get_origin(annotation) is list and len(items := get_args(annotation)) == 1:
        spec = _scalar_spec(_unannotated(items[0]))
        if spec and spec["type"] in _LIST_ITEMS:
            return {**spec, "type": "list", "items": spec["type"]}
    raise TypeError(
        f"OpenRouter Extractions cannot infer a field type for {name!r}. "
        "Use a supported scalar or list, set a compatible OpenRouterField(type=...), "
        "or use a structured-output model for nested schemas."
    )


def _unannotated(annotation: Any) -> Any:
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation


def _scalar_spec(annotation: Any) -> dict[str, Any] | None:
    if isinstance(annotation, type) and (kind := _SCALAR_TYPES.get(annotation)):
        return {"type": kind, "integer": True} if annotation is int else {"type": kind}
    return None
