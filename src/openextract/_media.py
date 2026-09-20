"""Resolve extraction inputs to ``(bytes, media_type)`` with SSRF and size caps."""

from __future__ import annotations

import asyncio
import ipaddress
import mimetypes
import os
import socket
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import BinaryIO, cast
from urllib.parse import urlparse

import httpx

from ._config import (
    _MAX_INPUT_BYTES_ENV,
    _allow_private_urls,
    _max_redirects,
    _resolve_max_input_bytes,
    _url_fetch_timeout,
)
from ._types import ExtractionInput, ExtractionInputLike, MediaSource, ResolvedSource
from .exceptions import InputFileError, InputTooLargeError, UrlFetchError

_DEFAULT_MEDIA_TYPE = "application/octet-stream"
_URL_PREFIXES = ("http://", "https://")
_INPUT_READ_CHUNK_SIZE = 64 * 1024
_BYTES_MEDIA_TYPE_REQUIRED = (
    "media_type is required when input_file is bytes or a file-like object; "
    "pass it explicitly, e.g. extract(..., media_type='application/pdf')."
)


def _get_media_type(file_path: str) -> str:
    """Return the MIME type for a file path (e.g. 'application/pdf')."""
    media_type, _ = mimetypes.guess_type(file_path)
    return media_type or _DEFAULT_MEDIA_TYPE


def _safe_source_context(source: str) -> str:
    """Return source context without URL credentials, query strings, or fragments."""
    if source.startswith(_URL_PREFIXES):
        parsed = urlparse(source)
        host = parsed.hostname or "unknown-host"
        try:
            parsed_port = parsed.port
        except ValueError:
            parsed_port = None
        port = f":{parsed_port}" if parsed_port is not None else ""
        return f"URL {parsed.scheme}://{host}{port}{parsed.path or '/'}"
    return f"path {Path(source).name!r}"


def _item_source_label(source: MediaSource, name: str | None) -> str | None:
    """Return a sanitized source label for diagnostics, preferring ``name``.

    URLs and paths are stripped of credentials, query strings, and fragments;
    raw bytes and file-like inputs have no safe label unless named.
    """
    if name is not None:
        return name
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if isinstance(source, str):
        return _safe_source_context(source)
    return None


# ---------------------------------------------------------------------------
# Byte-limit enforcement
# ---------------------------------------------------------------------------


def _input_file_error(exc: OSError, *, source: str) -> InputFileError:
    """Map an open/read OS failure to ``InputFileError`` with safe source context."""
    return InputFileError(f"Cannot read {source}: {exc.strerror or type(exc).__name__}")


@contextmanager
def _input_file_errors(*, source: str) -> Iterator[None]:
    """Convert open/read ``OSError`` values into ``InputFileError``."""
    try:
        yield
    except OSError as exc:
        raise _input_file_error(exc, source=source) from exc


def _input_too_large(*, limit: int, observed: int, source: str) -> InputTooLargeError:
    """Build the single ``InputTooLargeError`` message used by every reader."""
    return InputTooLargeError(
        f"{source} exceeds the configured size limit ({limit} bytes); "
        f"got at least {observed} bytes. Set {_MAX_INPUT_BYTES_ENV} or pass "
        "max_input_bytes=... if this is intentional."
    )


def _reject_declared_size(size: int | None, *, limit: int, source: str) -> None:
    """Fail before reading when an advertised size already exceeds the cap."""
    if size is not None and size > limit:
        raise _input_too_large(limit=limit, observed=size, source=source)


class _LimitedBuffer:
    """Accumulate chunks while failing fast once the byte cap is exceeded.

    Shared by the file-like, sync-response, and async-response readers so the
    cap is enforced identically no matter where the bytes come from.
    """

    def __init__(self, *, limit: int, source: str) -> None:
        self._chunks: list[bytes] = []
        self._limit = limit
        self._source = source
        self.total = 0

    def add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if self.total > self._limit:
            raise _input_too_large(limit=self._limit, observed=self.total, source=self._source)
        self._chunks.append(chunk)

    def value(self) -> bytes:
        return b"".join(self._chunks)

    def next_read_size(self) -> int:
        """Read at most one byte past the remaining budget to detect overruns."""
        return min(_INPUT_READ_CHUNK_SIZE, self._limit - self.total + 1)


def _enforce_max_input_bytes(data: bytes, *, limit: int, source: str) -> bytes:
    buffer = _LimitedBuffer(limit=limit, source=source)
    buffer.add(data)
    return buffer.value()


def _read_file_like_limited(stream: BinaryIO, *, limit: int, source: str) -> bytes:
    """Read a binary stream in bounded chunks, including non-seekable streams."""
    buffer = _LimitedBuffer(limit=limit, source=source)
    while True:
        chunk = stream.read(buffer.next_read_size())
        if not chunk:
            break
        buffer.add(chunk)
    return buffer.value()


# ---------------------------------------------------------------------------
# SSRF defenses
# ---------------------------------------------------------------------------


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True only for globally routable public unicast addresses.

    Treats IPv4-mapped IPv6 (``::ffff:a.b.c.d``) as its underlying IPv4 so
    that e.g. ``::ffff:127.0.0.1`` is correctly classified as loopback.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global


def _is_safe_host(host: str | None) -> bool:
    """Return True when ``host`` resolves only to public IP addresses.

    Refuses missing/empty hosts, IP literals in private/loopback/link-local/
    multicast/reserved ranges (including AWS/GCP metadata at
    ``169.254.169.254``), hostnames that resolve to any non-public IP, and
    hostnames that fail to resolve. Bypassed by ``OPENEXTRACT_ALLOW_PRIVATE_URLS``.
    """
    if _allow_private_urls():
        return True
    if not host:
        return False
    bare = host.strip("[]")
    try:
        return _is_public_ip(ipaddress.ip_address(bare))
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(bare, None)
    except OSError:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            resolved = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if not _is_public_ip(resolved):
            return False
    return True


def _unsafe_host_error(url: str) -> UrlFetchError:
    return UrlFetchError(f"Refusing to fetch URL with non-public host: {urlparse(url).hostname!r}")


def _require_safe_url(url: str) -> None:
    if not _is_safe_host(urlparse(url).hostname):
        raise _unsafe_host_error(url)


async def _require_safe_url_async(url: str) -> None:
    if not await asyncio.to_thread(_is_safe_host, urlparse(url).hostname):
        raise _unsafe_host_error(url)


# ---------------------------------------------------------------------------
# URL fetching
# ---------------------------------------------------------------------------


def _redirect_target(current: str, response: httpx.Response) -> str | None:
    """Return the next hop, or ``None`` when ``response`` is the final document."""
    if response.is_redirect:
        location = response.headers.get("location")
        if not location:
            raise UrlFetchError(f"Redirect from {current!r} missing Location header")
        return str(httpx.URL(current).join(location))
    response.raise_for_status()
    return None


def _too_many_redirects(limit: int) -> UrlFetchError:
    return UrlFetchError(f"Too many redirects (>{limit})")


def _declared_content_length(response: httpx.Response) -> int | None:
    raw = response.headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _response_buffer(response: httpx.Response, *, limit: int, source: str) -> _LimitedBuffer:
    """Start a capped buffer for ``response``, rejecting an oversized Content-Length."""
    _reject_declared_size(_declared_content_length(response), limit=limit, source=source)
    return _LimitedBuffer(limit=limit, source=source)


def _read_response_limited(response: httpx.Response, *, limit: int, source: str) -> bytes:
    buffer = _response_buffer(response, limit=limit, source=source)
    for chunk in response.iter_bytes(chunk_size=_INPUT_READ_CHUNK_SIZE):
        buffer.add(chunk)
    return buffer.value()


async def _read_response_limited_async(
    response: httpx.Response,
    *,
    limit: int,
    source: str,
) -> bytes:
    buffer = _response_buffer(response, limit=limit, source=source)
    async for chunk in response.aiter_bytes(chunk_size=_INPUT_READ_CHUNK_SIZE):
        buffer.add(chunk)
    return buffer.value()


def _read_url_with_client(
    url: str,
    client: httpx.Client,
    *,
    limit: int,
) -> tuple[bytes, Mapping[str, str]]:
    """Fetch a URL and stream its final response through the byte cap.

    The host is validated before every hop, so a redirect cannot walk the
    request onto a private address.
    """
    current = url
    redirect_limit = _max_redirects()
    for _ in range(redirect_limit):
        _require_safe_url(current)
        response = client.send(client.build_request("GET", current), stream=True)
        try:
            nxt = _redirect_target(current, response)
            if nxt is None:
                content = _read_response_limited(
                    response,
                    limit=limit,
                    source=_safe_source_context(current),
                )
                return content, dict(response.headers)
            current = nxt
        finally:
            response.close()
    raise _too_many_redirects(redirect_limit)


async def _read_url_with_client_async(
    url: str,
    client: httpx.AsyncClient,
    *,
    limit: int,
) -> tuple[bytes, Mapping[str, str]]:
    """Async counterpart to :func:`_read_url_with_client`."""
    current = url
    redirect_limit = _max_redirects()
    for _ in range(redirect_limit):
        await _require_safe_url_async(current)
        response = await client.send(client.build_request("GET", current), stream=True)
        try:
            nxt = _redirect_target(current, response)
            if nxt is None:
                content = await _read_response_limited_async(
                    response,
                    limit=limit,
                    source=_safe_source_context(current),
                )
                return content, dict(response.headers)
            current = nxt
        finally:
            await response.aclose()
    raise _too_many_redirects(redirect_limit)


@contextmanager
def _input_client(client: httpx.Client | None) -> Iterator[httpx.Client]:
    """Yield ``client``, or a short-lived one configured the same way."""
    if client is not None:
        yield client
        return
    with httpx.Client(follow_redirects=False, timeout=_url_fetch_timeout()) as owned:
        yield owned


@asynccontextmanager
async def _input_client_async(client: httpx.AsyncClient | None) -> AsyncIterator[httpx.AsyncClient]:
    """Async counterpart to :func:`_input_client`."""
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient(follow_redirects=False, timeout=_url_fetch_timeout()) as owned:
        yield owned


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def _media_from_content(
    file_path: str,
    content: bytes,
    headers: Mapping[str, str],
) -> tuple[bytes, str]:
    media_type = _get_media_type(file_path)
    if media_type == _DEFAULT_MEDIA_TYPE:
        header = headers.get("content-type", "").split(";", 1)[0].strip()
        if header:
            media_type = header
    return content, media_type


def _is_url(file_path: str) -> bool:
    return file_path.startswith(_URL_PREFIXES)


def _read_local_file(file_path: str, *, max_input_bytes: int) -> tuple[bytes, str]:
    """Read a local path through the byte cap, checking the stat size first."""
    path = Path(file_path)
    source = _safe_source_context(file_path)
    with _input_file_errors(source=source):
        _reject_declared_size(path.stat().st_size, limit=max_input_bytes, source=source)
        with path.open("rb") as stream:
            content = _read_file_like_limited(stream, limit=max_input_bytes, source=source)
    return content, _get_media_type(file_path)


def _read_from_path(
    file_path: str,
    *,
    max_input_bytes: int,
    client: httpx.Client | None = None,
) -> tuple[bytes, str]:
    """Read bytes from a local path or http(s) URL; return (bytes, media_type)."""
    if not _is_url(file_path):
        return _read_local_file(file_path, max_input_bytes=max_input_bytes)
    with _input_client(client) as http_client:
        content, headers = _read_url_with_client(file_path, http_client, limit=max_input_bytes)
    return _media_from_content(file_path, content, headers)


async def _read_from_path_async(
    file_path: str,
    client: httpx.AsyncClient | None,
    *,
    max_input_bytes: int,
) -> tuple[bytes, str]:
    """Async counterpart to :func:`_read_from_path`."""
    if not _is_url(file_path):
        return await asyncio.to_thread(
            _read_from_path,
            file_path,
            max_input_bytes=max_input_bytes,
        )
    async with _input_client_async(client) as http_client:
        content, headers = await _read_url_with_client_async(
            file_path,
            http_client,
            limit=max_input_bytes,
        )
    return _media_from_content(file_path, content, headers)


def _normalize_input(
    input_file: ExtractionInputLike,
    media_type: str | None,
) -> tuple[ResolvedSource, str | None]:
    """Unwrap an :class:`ExtractionInput` and coerce ``os.PathLike`` sources.

    The effective media type prefers an explicit per-item ``media_type``
    argument over the value carried on the ``ExtractionInput``. File-like
    objects (anything exposing ``.read()``) keep precedence over a coincidental
    ``os.PathLike`` implementation so streams are read, not opened as paths.
    """
    if isinstance(input_file, ExtractionInput):
        if media_type is None:
            media_type = input_file.media_type
        input_file = input_file.source
    if isinstance(input_file, os.PathLike) and not hasattr(input_file, "read"):
        input_file = os.fspath(input_file)
    # A PathLike that also exposes ``.read()`` is consumed as a stream, so it is
    # a valid member of ``ResolvedSource`` at runtime.
    return cast("ResolvedSource", input_file), media_type


def _unsupported_input() -> TypeError:
    return TypeError(
        "input_file must be a str path/URL, os.PathLike, bytes, or a file-like "
        "object with a .read() method."
    )


def _require_media_type(media_type: str | None) -> str:
    if media_type is None:
        raise TypeError(_BYTES_MEDIA_TYPE_REQUIRED)
    return media_type


def _get_media(
    input_file: ExtractionInputLike,
    media_type: str | None = None,
    *,
    max_input_bytes: int | None = None,
    client: httpx.Client | None = None,
) -> tuple[bytes, str]:
    """Resolve ``input_file`` to ``(bytes, media_type)``.

    ``str`` and ``os.PathLike`` are treated as a local path or http(s) URL.
    ``bytes`` and file-like objects (anything with a ``.read()`` method) are
    passed through. For the latter two, ``media_type`` is required.
    """
    input_file, media_type = _normalize_input(input_file, media_type)
    limit = _resolve_max_input_bytes(max_input_bytes)
    if isinstance(input_file, str):
        file_bytes, resolved_type = _read_from_path(
            input_file,
            max_input_bytes=limit,
            client=client,
        )
        return file_bytes, media_type or resolved_type

    if isinstance(input_file, bytes):
        resolved_type = _require_media_type(media_type)
        content = _enforce_max_input_bytes(input_file, limit=limit, source="bytes input")
        return content, resolved_type

    if hasattr(input_file, "read"):
        # Validate before touching the stream so an invalid call cannot consume it.
        resolved_type = _require_media_type(media_type)
        with _input_file_errors(source="file-like input"):
            content = _read_file_like_limited(input_file, limit=limit, source="file-like input")
        return content, resolved_type

    raise _unsupported_input()


async def _get_media_async(
    input_file: ExtractionInputLike,
    client: httpx.AsyncClient | None = None,
    media_type: str | None = None,
    *,
    max_input_bytes: int | None = None,
) -> tuple[bytes, str]:
    """Resolve media without blocking the event loop on disk, DNS, or stream I/O."""
    input_file, media_type = _normalize_input(input_file, media_type)
    limit = _resolve_max_input_bytes(max_input_bytes)
    if isinstance(input_file, str):
        file_bytes, resolved_type = await _read_from_path_async(
            input_file,
            client,
            max_input_bytes=limit,
        )
        return file_bytes, media_type or resolved_type

    if hasattr(input_file, "read"):
        return await asyncio.to_thread(
            _get_media,
            input_file,
            media_type,
            max_input_bytes=limit,
        )

    # ``bytes`` and unsupported inputs are cheap to handle inline; the sync
    # resolver owns the single copy of that validation.
    return _get_media(input_file, media_type=media_type, max_input_bytes=limit)
