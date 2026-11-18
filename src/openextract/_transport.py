"""POST model requests with bounded retries, in sync and async form."""

import asyncio
import math
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from ._diagnostics import has_error_envelope, provider_error, request_error
from .models import ResolvedModel

_RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}
_KEY_HEADERS = ("authorization", "x-api-key", "x-goog-api-key", "api-key")


def post(
    model: ResolvedModel, path: str, body: dict[str, Any], client: httpx.Client
) -> dict[str, Any]:
    attempt = 0
    while True:
        response = None
        try:
            response = client.post(model.base_url + path, **_request_options(model, body))
        except httpx.RequestError as exc:
            if _is_final(exc, attempt, model):
                raise request_error(exc, attempt + 1) from exc
        else:
            if _is_final(response, attempt, model):
                return _payload(response, model)
        time.sleep(_delay(attempt, response))
        attempt += 1


async def post_async(
    model: ResolvedModel, path: str, body: dict[str, Any], client: httpx.AsyncClient
) -> dict[str, Any]:
    attempt = 0
    while True:
        response = None
        try:
            response = await client.post(model.base_url + path, **_request_options(model, body))
        except httpx.RequestError as exc:
            if _is_final(exc, attempt, model):
                raise request_error(exc, attempt + 1) from exc
        else:
            if _is_final(response, attempt, model):
                return _payload(response, model)
        await asyncio.sleep(_delay(attempt, response))
        attempt += 1


def _request_options(model: ResolvedModel, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "headers": model.headers,
        "json": body,
        "timeout": model.config.timeout,
        "follow_redirects": False,
    }


def _is_final(
    outcome: httpx.Response | httpx.RequestError, attempt: int, model: ResolvedModel
) -> bool:
    """Whether to stop retrying: the budget is spent, or the outcome is not transient."""
    if attempt == model.config.max_retries:
        return True
    if isinstance(outcome, httpx.Response):
        return outcome.status_code not in _RETRY_STATUSES
    return not isinstance(outcome, httpx.TransportError)


def _delay(attempt: int, response: httpx.Response | None) -> float:
    """Honor ``Retry-After`` up to 30 seconds; otherwise back off exponentially up to 8."""
    retry_after = response.headers.get("retry-after") if response is not None else None
    if retry_after is not None and (seconds := _parse_retry_after(retry_after)) is not None:
        return min(max(seconds, 0.0), 30.0)
    return 0.5 * 2 ** min(attempt, 4)


def _parse_retry_after(value: str) -> float | None:
    try:
        seconds = float(value)
        return seconds if math.isfinite(seconds) else None
    except ValueError:
        pass
    try:
        return (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return None


def _payload(response: httpx.Response, model: ResolvedModel) -> dict[str, Any]:
    """Return the provider's JSON object, raising ``ModelError`` for a reported failure."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise provider_error(
            _error_body(response), response.status_code, _secrets(response, model)
        ) from exc
    # A malformed success body is reported as invalid model output by the caller.
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object from the model provider.")
    if has_error_envelope(payload):
        raise provider_error(payload, response.status_code, _secrets(response, model))
    return payload


def _error_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _secrets(response: httpx.Response, model: ResolvedModel) -> list[str]:
    secrets = [*model.config.headers.values(), model.config.api_key or ""]
    for header in _KEY_HEADERS:
        if value := response.request.headers.get(header):
            secrets.extend((value, value.partition(" ")[2]))
    return secrets
