"""Anthropic JSON output schema adaptation; final validation stays with Pydantic."""

from __future__ import annotations

import copy
import json
from typing import Any

_FORMATS = {
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
_LOCAL_CONSTRAINTS = {
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


def json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Close fixed objects, retain optional defaults, and describe local constraints.

    Reject maps and recursion rather than silently changing their meaning. Patterns
    are validated locally because Anthropic supports only a subset of regex syntax.
    """
    original = schema

    def visit(node: dict[str, Any], refs: tuple[str, ...] = ()) -> dict[str, Any]:
        node = copy.deepcopy(node)
        if "$ref" in node:
            ref = node["$ref"]
            if not ref.startswith("#/") or ref in refs:
                raise ValueError(
                    "Anthropic JSON output does not support external or recursive refs."
                )
            target = original
            for key in ref[2:].split("/"):
                target = target[key.replace("~1", "/").replace("~0", "~")]
            visit(target, (*refs, ref))
        if node.get("type") == "object":
            if node.get("additionalProperties") not in (None, False):
                raise ValueError(
                    "Anthropic JSON output does not support open-ended dictionaries; "
                    "use output_mode='tool' or a list of key/value objects."
                )
            node["additionalProperties"] = False
        if "allOf" in node or "prefixItems" in node or "patternProperties" in node:
            raise ValueError("Anthropic JSON output does not support this schema composition.")
        if any(isinstance(value, (dict, list)) for value in node.get("enum", [])):
            raise ValueError("Anthropic JSON output does not support complex enum values.")
        node.pop("discriminator", None)
        if "oneOf" in node:
            node["anyOf"] = node.pop("oneOf")
        constraints = {}
        for key in list(node):
            if (
                key in _LOCAL_CONSTRAINTS
                or (key == "minItems" and node[key] not in (0, 1))
                or (key == "format" and node[key] not in _FORMATS)
            ):
                constraints[key] = node.pop(key)
        if constraints:
            node["description"] = (
                node.get("description", "") + " Constraints: " + json.dumps(constraints)
            ).strip()
        for keyword in ("properties", "$defs", "definitions"):
            if keyword in node:
                node[keyword] = {key: visit(child, refs) for key, child in node[keyword].items()}
        if isinstance(node.get("items"), dict):
            node["items"] = visit(node["items"], refs)
        if "anyOf" in node:
            node["anyOf"] = [visit(child, refs) for child in node["anyOf"]]
        return node

    return visit(schema)
