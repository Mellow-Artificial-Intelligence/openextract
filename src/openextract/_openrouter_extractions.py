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
from .models import ResolvedModel

_SCALAR_TYPES = {
    str: "text",
    int: "number",
    float: "number",
    Decimal: "number",
    date: "date",
    time: "time",
    datetime: "datetime",
}
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


def request_body(
    schema: type[BaseModel], model: ResolvedModel, content: Content, instructions: str | None
) -> dict[str, Any]:
    if schema.__pydantic_root_model__:
        raise TypeError("OpenRouter Extractions requires a Pydantic model with named fields.")
    if not schema.model_fields:
        raise ValueError("OpenRouter Extractions requires at least one schema field.")
    settings = model.config.settings
    if reserved := _REQUEST_FIELDS & settings.keys():
        raise ValueError(f"settings cannot override request fields: {sorted(reserved)}.")
    body = {
        **settings,
        "model": model.model_name,
        "fields": {
            name: _field_spec(name, field, instructions)
            for name, field in schema.model_fields.items()
        },
    }
    if content.is_text:
        body["state"] = content.text
    elif asset := _ASSET_FORMATS.get(content.media_type):
        kind, file_format = asset
        body["state"] = "document"
        body["assets"] = {
            "document": {"type": kind, "format": file_format, "data_base64": content.base64}
        }
    else:
        raise ValueError(
            "OpenRouter Extractions supports text, PDF, PNG, JPEG, and WebP. "
            f"Received {content.media_type}."
        )
    return body


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
        types = [member for member in get_args(annotation) if member is not NoneType]
        if len(types) == 1:
            annotation = types[0]
    annotation = _unannotated(annotation)
    if spec := _scalar_spec(annotation):
        return spec
    if get_origin(annotation) is list:
        items = get_args(annotation)
        if (
            len(items) == 1
            and (spec := _scalar_spec(_unannotated(items[0])))
            and spec["type"] in ("text", "number", "date")
        ):
            return {**spec, "type": "list", "items": spec["type"]}
    raise TypeError(
        f"OpenRouter Extractions cannot infer a field type for {name!r}. "
        "Use a supported scalar or list, set a compatible OpenRouterField(type=...), "
        "or use a structured-output model for nested schemas."
    )


def _unannotated(annotation: Any) -> Any:
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _scalar_spec(annotation: Any) -> dict[str, Any] | None:
    if isinstance(annotation, type) and (kind := _SCALAR_TYPES.get(annotation)):
        return {"type": kind, **({"integer": True} if annotation is int else {})}
    return None


def parse[T: BaseModel](
    schema: type[T], payload: dict[str, Any], include_metadata: bool
) -> T | ExtractionResult[T]:
    answers = payload.get("answers")
    if not isinstance(answers, dict) or answers.keys() != schema.model_fields.keys():
        raise ValueError("OpenRouter Extractions must return one answer for every schema field.")
    if not include_metadata and payload.get("truncated") is True:
        raise ValueError(
            "OpenRouter read only part of the source. Use include_metadata=True to inspect "
            "partial results or retry with a shorter source."
        )
    values = {}
    for name, answer in answers.items():
        if not isinstance(answer, dict) or "value" not in answer:
            raise ValueError("Every OpenRouter Extractions answer must contain a value.")
        values[name] = answer["value"]
    data = schema.model_validate_json(json.dumps(values), by_alias=False, by_name=True)
    if include_metadata:
        metadata = {key: value for key, value in payload.items() if key != "answers"}
        return ExtractionResult(data, answers, metadata)
    return data
