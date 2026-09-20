"""JSON Schema → Pydantic model loading (CLI ``--schema`` subset)."""

from __future__ import annotations

import json
from pathlib import Path
from types import GenericAlias
from typing import Any, cast

from pydantic import BaseModel, create_model

_JSON_PRIMITIVES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _json_schema_model_name(schema: dict[str, Any], path: Path | None = None) -> str:
    """Pick a valid class name from the schema title or file stem."""
    title = schema.get("title")
    if isinstance(title, str) and title.isidentifier():
        return title
    if path is not None:
        stem = path.stem
        if stem.isidentifier():
            return stem
    return "JsonSchema"


def _schema_object(node: object) -> dict[str, Any] | None:
    """Narrow a JSON object to ``dict[str, Any]``."""
    if not isinstance(node, dict):
        return None
    return cast(dict[str, Any], node)


def _json_schema_annotation(node: object, *, name: str) -> Any:
    """Map a JSON Schema node to a Pydantic annotation."""
    schema = _schema_object(node)
    if schema is None:
        return Any
    schema_type = schema.get("type")
    if schema_type == "array":
        item = _json_schema_annotation(schema.get("items"), name=f"{name}Item")
        return GenericAlias(list, (item,))
    if schema_type == "object" or (schema_type is None and "properties" in schema):
        return _model_from_json_schema(schema, name)
    if isinstance(schema_type, str):
        return _JSON_PRIMITIVES.get(schema_type, Any)
    return Any


def _model_from_json_schema(schema: dict[str, Any], name: str) -> type[BaseModel]:
    """Build a Pydantic model from a JSON Schema object."""
    required_raw = schema.get("required")
    required = set(required_raw) if isinstance(required_raw, list) else set()
    properties: dict[str, Any] = _schema_object(schema.get("properties")) or {}
    fields: dict[str, Any] = {}
    for key, spec in properties.items():
        if not key.isidentifier():
            raise ValueError(
                f"cannot build a Pydantic model from '{name}': invalid field name {key!r}"
            )
        annotation = _json_schema_annotation(spec, name=f"{name}{key.title()}")
        fields[key] = (annotation, ...) if key in required else (annotation | None, None)
    try:
        return create_model(name, __module__="openextract._schema_json", **fields)
    except Exception as exc:
        raise ValueError(f"cannot build a Pydantic model from '{name}': {exc}") from exc


def _require_object_schema(data: object, origin: str) -> dict[str, Any]:
    """Reject non-object JSON Schema documents."""
    schema = _schema_object(data)
    if schema is None:
        raise ValueError(f"{origin} must be a JSON Schema object")
    schema_type = schema.get("type")
    if schema_type is not None and schema_type != "object":
        raise ValueError(f"{origin} must be a JSON Schema object")
    return schema


def _read_json_schema_file(path: Path) -> dict[str, Any]:
    """Load a JSON Schema object from ``path`` with clear file errors."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"schema file '{path}' not found") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read schema file '{path}': {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"schema file '{path}' is not valid JSON: {exc}") from exc
    return _require_object_schema(data, f"schema file '{path}'")


def _probe_json_schema_file(path: Path) -> dict[str, Any] | None:
    """Return a JSON Schema object if ``path`` already is one, else ``None``."""
    try:
        return _read_json_schema_file(path)
    except ValueError:
        return None


def _model_from_path(path: Path) -> type[BaseModel]:
    """Build a Pydantic model from a JSON Schema file."""
    schema = _read_json_schema_file(path)
    return _model_from_json_schema(schema, _json_schema_model_name(schema, path))


def _model_from_text(raw: str) -> type[BaseModel]:
    """Build a Pydantic model from JSON Schema text."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"schema is not valid JSON: {exc}") from exc
    schema = _require_object_schema(data, "schema")
    return _model_from_json_schema(schema, _json_schema_model_name(schema))


def schema_from_json(source: str | Path | dict[str, Any]) -> type[BaseModel]:
    """Build a Pydantic model from a JSON Schema file, object, or JSON text.

    ``source`` may be a filesystem path (``str`` or ``Path``), a JSON object
    ``dict``, or a JSON text string. The supported subset matches CLI
    ``--schema``: object schemas with ``properties`` / ``items`` / nested
    objects. This is not a full JSON Schema compiler.

    A ``str`` is treated as a file path when it has a ``.json`` suffix or
    names an existing file; otherwise it is parsed as JSON text.

    Raises:
        ValueError: If a path is missing or unreadable, the text is not valid
            JSON, or the document is not a JSON Schema object.
        TypeError: If ``source`` is not a path, ``dict``, or ``str``.
    """
    if isinstance(source, dict):
        schema = _require_object_schema(source, "schema")
        return _model_from_json_schema(schema, _json_schema_model_name(schema))
    if isinstance(source, Path):
        return _model_from_path(source.expanduser())
    if isinstance(source, str):
        path = Path(source).expanduser()
        if path.suffix.lower() == ".json" or path.is_file():
            return _model_from_path(path)
        return _model_from_text(source)
    raise TypeError(
        "schema_from_json() expected a path, JSON object dict, or JSON text, "
        f"got {type(source).__name__}"
    )
