"""Read a source into bytes and a media type."""

import base64
import mimetypes
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

import httpx

from .exceptions import ExtractionError

type Source = str | os.PathLike[str] | bytes

PROMPT = "Extract the requested information from this file."

# Office and OpenDocument types, which Python's built-in table only partly covers.
DOCUMENT_EXTENSIONS = {
    "application/pdf": ".pdf",
    "application/rtf": ".rtf",
    "application/msword": ".doc",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.oasis.opendocument.presentation": ".odp",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.template": ".dotx",
}
_DOCUMENT_TYPES = {extension: media for media, extension in DOCUMENT_EXTENSIONS.items()}
_TEXT_TYPES = {"application/json", "application/xml", "application/javascript"}
_GENERIC_TYPES = {"", "application/octet-stream", "binary/octet-stream"}
# Leading bytes, used when neither a file name nor a header names the type.
_SIGNATURES = {
    b"%PDF": "application/pdf",
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
    b"ID3": "audio/mpeg",
    b"fLaC": "audio/flac",
    b"OggS": "audio/ogg",
}
_RIFF_FORMATS = {b"WEBP": "image/webp", b"WAVE": "audio/wav"}
_URL_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class Content:
    """A loaded source and the encodings providers ask for."""

    data: bytes
    media_type: str

    @property
    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def data_uri(self) -> str:
        return f"data:{self.media_type};base64,{self.base64}"

    @property
    def is_text(self) -> bool:
        return self.media_type.startswith("text/") or self.media_type in _TEXT_TYPES

    @property
    def is_image(self) -> bool:
        return self.media_type.startswith("image/")

    @property
    def is_pdf(self) -> bool:
        return self.media_type == "application/pdf"

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


def load(source: Source, media_type: str | None) -> Content:
    """Read text, a URL, bytes, or a path. An explicit ``media_type`` always wins."""
    if _is_url(source):
        with _fetch_errors():
            response = httpx.get(source, follow_redirects=True, timeout=_URL_TIMEOUT_SECONDS)
            return _downloaded(response, media_type)
    if isinstance(source, bytes):
        return _content(source, media_type)
    if isinstance(source, str):
        return _content(source.encode(), media_type or "text/plain")
    path = os.fspath(source)
    return _content(Path(path).read_bytes(), media_type, name=path)


async def load_async(source: Source, media_type: str | None) -> Content:
    """Like :func:`load`, fetching URLs without blocking. Local files are read synchronously."""
    if not _is_url(source):
        return load(source, media_type)
    with _fetch_errors():
        async with httpx.AsyncClient(follow_redirects=True, timeout=_URL_TIMEOUT_SECONDS) as client:
            return _downloaded(await client.get(source), media_type)


def _is_url(source: Source) -> TypeGuard[str]:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _downloaded(response: httpx.Response, media_type: str | None) -> Content:
    response.raise_for_status()
    header = response.headers.get("content-type", "").split(";")[0].strip().lower()
    return _content(
        response.content,
        media_type,
        name=response.url.path,
        header=None if header in _GENERIC_TYPES else header,
    )


def _content(
    data: bytes, media_type: str | None, *, name: str = "", header: str | None = None
) -> Content:
    """Prefer the explicit type, then the file extension, the header, and the leading bytes."""
    resolved = media_type or _type_from_name(name) or header or _sniff(data)
    if resolved is None:
        raise ValueError("Could not infer the media type; pass media_type explicitly.")
    return Content(data, resolved)


def _type_from_name(name: str) -> str | None:
    return _DOCUMENT_TYPES.get(Path(name).suffix.lower()) or mimetypes.guess_type(name)[0]


def _sniff(data: bytes) -> str | None:
    """Guess a media type from the content, falling back to plain text for UTF-8."""
    for signature, media_type in _SIGNATURES.items():
        if data.startswith(signature):
            return media_type
    if data.startswith(b"RIFF"):
        return _RIFF_FORMATS.get(data[8:12])
    if data[4:8] == b"ftyp":
        return "video/mp4"
    try:
        data.decode()
    except UnicodeDecodeError:
        return None
    return "text/plain"


@contextmanager
def _fetch_errors() -> Generator[None]:
    try:
        yield
    except httpx.HTTPError as exc:
        detail = type(exc).__name__
        if isinstance(exc, httpx.HTTPStatusError):
            detail = f"HTTP {exc.response.status_code}"
        raise ExtractionError(
            f"Could not fetch the source URL ({detail}).\n"
            "Next step: Check the URL and network access. If the source needs authentication "
            "or a custom timeout, download it with your own HTTP client "
            "and pass its bytes to extract()."
        ) from exc
