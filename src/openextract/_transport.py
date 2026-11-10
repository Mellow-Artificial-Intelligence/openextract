"""Shared sync/async HTTP transport and bounded retries."""

import asyncio
import math
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from ._provider_errors import provider_error
from .exceptions import ModelError
from .models import ResolvedModel

_RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}


def _delay(attempt: int, response: httpx.Response | None) -> float:
    value = response.headers.get("retry-after") if response is not None else None
    if value is not None:
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                seconds = float("nan")
        if math.isfinite(seconds):
            return min(max(seconds, 0.0), 30.0)
    return min(0.5 * 2 ** min(attempt, 4), 8.0)


def _check(response: httpx.Response, model: ResolvedModel) -> dict[str, Any]:
    secrets = list(model.config.headers.values())
    if model.config.api_key:
        secrets.append(model.config.api_key)
    for header in ("authorization", "x-api-key", "x-goog-api-key", "api-key"):
        value = response.request.headers.get(header)
        if value:
            secrets.extend((value, value.partition(" ")[2]))
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        failure = provider_error(
            payload if isinstance(payload, dict) else {}, response.status_code, secrets
        )
        assert failure is not None
        raise failure from exc
    # Parsing errors are handled at the extraction boundary, separately from HTTP failures.
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the model provider.")
    failure = provider_error(result, response.status_code, secrets)
    if failure is not None:
        raise failure
    return result


def post(
    model: ResolvedModel, path: str, body: Mapping[str, Any], client: httpx.Client
) -> dict[str, Any]:
    for attempt in range(model.config.max_retries + 1):
        response = None
        try:
            response = client.post(
                model.base_url + path,
                headers=model.headers,
                json=body,
                timeout=model.config.timeout,
                follow_redirects=False,
            )
        except httpx.RequestError as exc:
            if not isinstance(exc, httpx.TransportError) or attempt == model.config.max_retries:
                raise ModelError(f"The model request failed ({type(exc).__name__}).") from exc
        else:
            if response.status_code not in _RETRY_STATUSES or attempt == model.config.max_retries:
                return _check(response, model)
        time.sleep(_delay(attempt, response))
    raise AssertionError("Unreachable")  # pragma: no cover


async def post_async(
    model: ResolvedModel, path: str, body: Mapping[str, Any], client: httpx.AsyncClient
) -> dict[str, Any]:
    for attempt in range(model.config.max_retries + 1):
        response = None
        try:
            response = await client.post(
                model.base_url + path,
                headers=model.headers,
                json=body,
                timeout=model.config.timeout,
                follow_redirects=False,
            )
        except httpx.RequestError as exc:
            if not isinstance(exc, httpx.TransportError) or attempt == model.config.max_retries:
                raise ModelError(f"The model request failed ({type(exc).__name__}).") from exc
        else:
            if response.status_code not in _RETRY_STATUSES or attempt == model.config.max_retries:
                return _check(response, model)
        await asyncio.sleep(_delay(attempt, response))
    raise AssertionError("Unreachable")  # pragma: no cover
