"""Adapt Pydantic's JSON Schema to provider dialects; Pydantic still validates the result."""

import copy
import json
from collections.abc import Iterator
from typing import Any

type Schema = dict[str, Any]

_ANTHROPIC_FORMATS = {
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
}
# Anthropic rejects these keywords, so they move into descriptions and are enforced locally.
_ANTHROPIC_UNSUPPORTED = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
    "uniqueItems",
    "pattern",
}


def openai_strict(schema: Schema) -> Schema:
    """Require every declared field and close fixed objects, as OpenAI strict output expects."""
    schema = copy.deepcopy(schema)
    for node in _subschemas(schema):
        node.pop("default", None)
        _close(node, "Strict JSON Schema output", "use Model(api='chat') for tool-based output")
        if node.get("type") == "object":
            node["required"] = list(node.get("properties", {}))
    return schema


def anthropic_json(schema: Schema) -> Schema:
    """Close fixed objects, keep optional defaults, and describe constraints Anthropic rejects.

    Maps, recursion, and unsupported compositions raise rather than silently change meaning.
    """
    schema = copy.deepcopy(schema)
    _reject_recursion(schema, schema, ())
    for node in _subschemas(schema):
        _close(
            node, "Anthropic JSON output", "use output_mode='tool' or a list of key/value objects"
        )
        if node.keys() & {"allOf", "prefixItems", "patternProperties"}:
            raise ValueError("Anthropic JSON output does not support this schema composition.")
        if any(isinstance(value, (dict, list)) for value in node.get("enum", [])):
            raise ValueError("Anthropic JSON output does not support complex enum values.")
        constraints = {
            key: node.pop(key) for key in list(node) if _is_local_constraint(key, node[key])
        }
        if constraints:
            described = f"{node.get('description', '')} Constraints: {json.dumps(constraints)}"
            node["description"] = described.strip()
    return schema


def _subschemas(node: Schema) -> Iterator[Schema]:
    """Yield a schema, then everything nested in it as it stands once the caller adapts it."""
    yield node
    for keyword in ("properties", "$defs", "definitions"):
        for child in node.get(keyword, {}).values():
            yield from _subschemas(child)
    for keyword in ("items", "additionalProperties"):
        if isinstance(node.get(keyword), dict):
            yield from _subschemas(node[keyword])
    for keyword in ("anyOf", "allOf", "oneOf", "prefixItems"):
        for child in node.get(keyword, []):
            yield from _subschemas(child)


def _close(node: Schema, output: str, alternative: str) -> None:
    """Forbid undeclared properties and express tagged unions without a discriminator."""
    node.pop("discriminator", None)
    if "oneOf" in node:
        # The branches stay disjoint through their discriminator literals.
        node["anyOf"] = node.pop("oneOf")
    if node.get("type") == "object":
        if node.get("additionalProperties") not in (None, False):
            raise ValueError(f"{output} does not support open-ended dictionaries; {alternative}.")
        node["additionalProperties"] = False


def _is_local_constraint(keyword: str, value: Any) -> bool:
    return (
        keyword in _ANTHROPIC_UNSUPPORTED
        or (keyword == "minItems" and value not in (0, 1))
        or (keyword == "format" and value not in _ANTHROPIC_FORMATS)
    )


def _reject_recursion(root: Schema, node: Schema, followed: tuple[str, ...]) -> None:
    for schema in _subschemas(node):
        ref = schema.get("$ref")
        if ref is None:
            continue
        if not ref.startswith("#/") or ref in followed:
            raise ValueError("Anthropic JSON output does not support external or recursive refs.")
        target = root
        for key in ref[2:].split("/"):
            target = target[key.replace("~1", "/").replace("~0", "~")]
        _reject_recursion(root, target, (*followed, ref))
