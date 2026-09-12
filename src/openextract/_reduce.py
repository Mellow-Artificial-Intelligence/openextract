"""Reduce strategies that combine the outputs of parallel swarm agents."""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, assert_never, cast

from ._errors import _extraction_errors
from ._types import Citation, T

_MISSING = object()


class SwarmReduce(StrEnum):
    """How a swarm folds several agent outputs into one result.

    ``merge`` unions lists and fills fields from whichever agent produced a
    value, ``vote`` keeps the most frequently produced value per field, and
    ``first`` returns the first successful agent's output untouched.
    """

    MERGE = "merge"
    VOTE = "vote"
    FIRST = "first"


def normalize_reduce(reduce: SwarmReduce | str = "merge") -> SwarmReduce:
    """Return a valid :class:`SwarmReduce` or raise ``ValueError``."""
    try:
        return SwarmReduce(reduce)
    except ValueError:
        allowed = ", ".join(repr(item.value) for item in SwarmReduce)
        raise ValueError(f"reduce must be one of {allowed}; got {reduce!r}.") from None


def _is_empty(value: object) -> bool:
    """Treat ``None`` and the empty string as "this agent had nothing"."""
    return value is None or value == ""


def _stable_key(value: object) -> str:
    """Return an order-independent identity for ``value`` used to dedupe/count."""
    return json.dumps(value, sort_keys=True, default=str)


def _union(items: Sequence[Any]) -> list[Any]:
    """Concatenate list values while dropping structurally duplicate entries."""
    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        key = _stable_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _keys(values: Sequence[dict[str, Any]]) -> list[str]:
    """Return every key across ``values`` in first-seen order."""
    return list(dict.fromkeys(key for value in values for key in value))


def merge_values(values: Sequence[Any]) -> Any:
    """Combine agent values by unioning lists and filling fields field-wise.

    Dicts recurse per key so partial extractions complement each other. Scalars
    resolve to the first non-empty value, preserving agent order.
    """
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if all(isinstance(value, list) for value in values):
        return _union([item for value in values for item in value])
    if all(isinstance(value, dict) for value in values):
        return {key: merge_values([value.get(key) for value in values]) for key in _keys(values)}
    return next((value for value in values if not _is_empty(value)), values[0])


def vote_values(values: Sequence[Any]) -> Any:
    """Combine agent values by majority per field.

    Lists have no meaningful majority, so they fall back to
    :func:`merge_values`. Dicts recurse per key; scalars pick the most common
    non-empty value, breaking ties toward the earliest agent.
    """
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if all(isinstance(value, list) for value in values):
        return merge_values(values)
    if all(isinstance(value, dict) for value in values):
        return {key: vote_values([value.get(key) for value in values]) for key in _keys(values)}
    counts: dict[str, tuple[int, Any]] = {}
    best: tuple[int, Any] | None = None
    for value in values:
        if _is_empty(value):
            continue
        key = _stable_key(value)
        count = counts.get(key, (0, value))[0] + 1
        counts[key] = (count, value)
        if best is None or count > best[0]:
            best = (count, value)
    if best is not None:
        return best[1]
    return values[0]


def reduce_outputs(values: Sequence[T], reduce: SwarmReduce | str = "merge") -> T:
    """Fold successful swarm outputs into a single validated schema instance.

    ``values`` must be instances of the same Pydantic model. ``merge`` and
    ``vote`` reduce the dumped payloads and re-validate the combined value, so
    the returned object always satisfies the schema.

    Raises:
        ValueError: If ``values`` is empty or ``reduce`` is not a known strategy.
        SchemaValidationError: If the combined payload no longer validates.
    """
    strategy = normalize_reduce(reduce)
    if not values:
        raise ValueError("reduce_outputs requires at least one value.")
    if len(values) == 1 or strategy is SwarmReduce.FIRST:
        return values[0]
    schema = type(values[0])
    payloads = [value.model_dump() for value in values]
    combined = merge_values(payloads) if strategy is SwarmReduce.MERGE else vote_values(payloads)
    with _extraction_errors():
        return schema.model_validate(combined)


def reduce_citations(
    items: Sequence[tuple[T, tuple[Citation, ...]]],
    output: T,
    reduce: SwarmReduce | str = "merge",
) -> tuple[Citation, ...]:
    """Fold per-agent citations onto the reduced swarm ``output``.

    ``first`` keeps the leading agent's citations. ``merge`` and ``vote`` keep
    the first citation whose agent's field value equals ``output`` at that
    path, so a discarded minority vote does not survive. Per-agent citations
    are not mutated.
    """
    strategy = normalize_reduce(reduce)
    if not items:
        return ()
    if len(items) == 1:
        return items[0][1]
    match strategy:
        case SwarmReduce.FIRST:
            return items[0][1]
        case SwarmReduce.MERGE | SwarmReduce.VOTE:
            return _citations_matching_output(items, output)
    assert_never(strategy)


def _citations_matching_output(
    items: Sequence[tuple[T, tuple[Citation, ...]]],
    output: T,
) -> tuple[Citation, ...]:
    """Keep the first citation whose agent value equals ``output`` at that path."""
    reduced = output.model_dump()
    chosen: dict[str, Citation] = {}
    for value, citations in items:
        dumped = value.model_dump()
        for citation in citations:
            if citation.field in chosen:
                continue
            if _field_values_match(dumped, reduced, citation.field):
                chosen[citation.field] = citation
    return tuple(chosen.values())


def _field_values_match(agent: object, reduced: object, field: str) -> bool:
    """True when ``field`` exists on both dumps and the values are equal."""
    left = _lookup_field(agent, field)
    right = _lookup_field(reduced, field)
    return left is not _MISSING and right is not _MISSING and left == right


def _lookup_field(data: object, field: str) -> object:
    """Return the dumped value at a dotted/indexed citation path, if present."""
    node: object = data
    index = 0
    length = len(field)
    while index < length:
        if field[index] == ".":
            index += 1
            continue
        if field[index] == "[":
            close = field.find("]", index)
            if close < 0 or not isinstance(node, list):
                return _MISSING
            pos = int(field[index + 1 : close])
            if pos >= len(node):
                return _MISSING
            node = node[pos]
            index = close + 1
            continue
        end = index
        while end < length and field[end] not in ".[":
            end += 1
        key = field[index:end]
        if not isinstance(node, dict):
            return _MISSING
        mapping = cast(dict[str, object], node)
        if key not in mapping:
            return _MISSING
        node = mapping[key]
        index = end
    return node
