"""Out-of-the-box extraction styles backed by the Pydantic AI harness."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import assert_never

from .exceptions import ProviderNotInstalledError

_CODE_VIRTUAL_ROOT = "/work"
# The pydantic-ai-harness version line the FileSystem/CodeMode integration was
# written against. The repo's locked dev environment cannot install it (it
# requires pydantic-ai-slim 2.x), so CI does not exercise the live harness.
_HARNESS_TARGET_VERSION = "0.18.x"
_BINARY_MEDIA_PREFIXES = (
    "image/",
    "audio/",
    "video/",
    # Office container families as reported by ``mimetypes.guess_type``, which
    # backs the repo's media-type detection for path and URL inputs.
    "application/vnd.openxmlformats-officedocument.",
    "application/vnd.oasis.opendocument.",
)
_BINARY_MEDIA_TYPES = frozenset(
    {
        "application/pdf",
        "application/zip",
        "application/gzip",
        "application/x-gzip",
        "application/x-tar",
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
    }
)
_TEXT_APPLICATION_TYPES = frozenset(
    {
        "application/csv",
        "application/graphql",
        "application/javascript",
        "application/json",
        "application/ld+json",
        "application/sql",
        "application/toml",
        "application/xml",
        "application/x-ndjson",
        "application/x-sh",
        "application/x-yaml",
        "application/yaml",
    }
)
_DOCUMENT_FILENAMES = {
    "csv": "document.csv",
    "html": "document.html",
    "javascript": "document.js",
    "json": "document.json",
    "ld+json": "document.json",
    "markdown": "document.md",
    "plain": "document.txt",
    "tab-separated-values": "document.tsv",
    "toml": "document.toml",
    "x-markdown": "document.md",
    "x-ndjson": "document.ndjson",
    "x-sh": "document.sh",
    "x-yaml": "document.yaml",
    "xml": "document.xml",
    "yaml": "document.yaml",
}


TABLE_INSTRUCTIONS = (
    "The document contains a table or line items (invoice, statement, or similar). "
    "Extract every row in order. Do not drop, merge, invent, or summarize rows. "
    "Map columns from the header row. Keep numeric amounts as written. "
    "Use null for fields that are not present. Continue a table across page breaks. "
    "Empty tables are an empty list, not omitted."
)

FORM_INSTRUCTIONS = (
    "The document is a form, receipt, or other labeled key-value document. "
    "Extract every labeled field. Keep labels and values as written. "
    "Do not invent missing fields; use null when a field is not present. "
    "Preserve nested sections. Do not summarize."
)


class ExtractionStyle(StrEnum):
    """How an extraction run inspects the input.

    ``direct``
        Pass the media bytes to the model in one shot (default).
    ``table``
        Same media path as ``direct``, with row-oriented guidance. PDFs reuse
        the local parse-then-window path so line items across pages merge.
    ``form``
        Same media path as ``direct``, with labeled-field guidance. PDFs reuse
        the local parse-then-window path so fields across pages merge.
    ``search``
        For text, give the model sandboxed file tools (read, regex search,
        glob) against a workspace copy of the document.
    ``code``
        For text, give the model a sandboxed ``run_code`` tool that can open
        and parse the document with Python.
    """

    DIRECT = "direct"
    TABLE = "table"
    FORM = "form"
    SEARCH = "search"
    CODE = "code"


def normalize_style(style: ExtractionStyle | str) -> ExtractionStyle:
    """Return a valid :class:`ExtractionStyle` or raise ``ValueError``."""
    try:
        return ExtractionStyle(style)
    except ValueError:
        allowed = ", ".join(repr(item.value) for item in ExtractionStyle)
        raise ValueError(f"style must be one of {allowed}; got {style!r}.") from None


def uses_workspace(style: ExtractionStyle) -> bool:
    """Return whether ``style`` materializes a text workspace (search/code)."""
    match style:
        case ExtractionStyle.SEARCH | ExtractionStyle.CODE:
            return True
        case ExtractionStyle.DIRECT | ExtractionStyle.TABLE | ExtractionStyle.FORM:
            return False
        case _:  # pragma: no cover - exhaustive ExtractionStyle
            assert_never(style)


def should_parse(cite: bool, style: ExtractionStyle, pages: object = None) -> bool:
    """Return whether this run should locally parse a PDF before extract.

    ``cite=True`` always parses so citations can be grounded. ``table`` and
    ``form`` also parse so rows and labeled fields are extracted per page
    window and merged. A ``pages`` filter parses so only those 1-based pages
    are sent to the model.
    """
    return (
        cite or pages is not None or style is ExtractionStyle.TABLE or style is ExtractionStyle.FORM
    )


def _with_prefixed_instructions(prefix: str, instructions: str | None) -> str:
    if instructions and instructions.strip():
        return f"{prefix}\n\n{instructions.strip()}"
    return prefix


def with_table_instructions(instructions: str | None) -> str:
    """Prepend table/line-item guidance without dropping caller instructions."""
    return _with_prefixed_instructions(TABLE_INSTRUCTIONS, instructions)


def with_form_instructions(instructions: str | None) -> str:
    """Prepend form/key-value guidance without dropping caller instructions."""
    return _with_prefixed_instructions(FORM_INSTRUCTIONS, instructions)


def with_style_instructions(instructions: str | None, style: ExtractionStyle) -> str | None:
    """Return caller instructions, with table or form guidance when applicable."""
    match style:
        case ExtractionStyle.TABLE:
            return with_table_instructions(instructions)
        case ExtractionStyle.FORM:
            return with_form_instructions(instructions)
        case ExtractionStyle.DIRECT | ExtractionStyle.SEARCH | ExtractionStyle.CODE:
            return instructions
        case _:  # pragma: no cover - exhaustive ExtractionStyle
            assert_never(style)


def _bare_media_type(media_type: str) -> str:
    return media_type.split(";", 1)[0].strip().lower()


def is_text_media_type(media_type: str) -> bool:
    """Return whether ``media_type`` is a known textual MIME type."""
    bare = _bare_media_type(media_type)
    return (
        bare.startswith("text/")
        or bare in _TEXT_APPLICATION_TYPES
        or bare.endswith(("+json", "+xml", "+yaml"))
    )


def is_binary_media_type(media_type: str) -> bool:
    """Return whether ``media_type`` is a known non-text MIME type."""
    bare = _bare_media_type(media_type)
    return bare.startswith(_BINARY_MEDIA_PREFIXES) or bare in _BINARY_MEDIA_TYPES


def document_filename(media_type: str) -> str:
    """Return a stable workspace filename for a textual media type."""
    subtype = _bare_media_type(media_type).rsplit("/", 1)[-1]
    return _DOCUMENT_FILENAMES.get(subtype, "document.txt")


def decode_text_document(data: bytes, media_type: str, *, style: ExtractionStyle) -> str:
    """Decode UTF-8 text for search/code styles; reject binary inputs."""
    if is_binary_media_type(media_type) and not is_text_media_type(media_type):
        raise ValueError(
            f"style {style.value!r} requires a text document; got media_type={media_type!r}. "
            "Use style='direct' for PDFs, images, audio, and video."
        )
    if b"\x00" in data:
        raise ValueError(
            f"style {style.value!r} requires a text document; the input contains NUL bytes. "
            "Use style='direct' instead."
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"style {style.value!r} requires UTF-8 text; the input is not valid UTF-8. "
            "Use style='direct' instead."
        ) from exc


def materialize_text_document(
    workspace: Path,
    data: bytes,
    media_type: str,
    *,
    style: ExtractionStyle,
) -> str:
    """Write the decoded document into ``workspace`` and return its filename."""
    filename = document_filename(media_type)
    (workspace / filename).write_text(
        decode_text_document(data, media_type, style=style),
        encoding="utf-8",
    )
    return filename


def _search_capabilities(workspace: Path) -> list[object]:
    try:
        filesystem = __import__("pydantic_ai_harness", fromlist=["FileSystem"]).FileSystem
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "style='search' requires pydantic-ai-harness. "
            "Install it with: pip install pydantic-ai-harness "
            "(or 'pip install pydantic-ai-harness[codemode]'). "
            f"The integration targets pydantic-ai-harness {_HARNESS_TARGET_VERSION}. "
            f"Original error: {exc}"
        ) from exc
    return [filesystem(root_dir=workspace)]


def _code_capabilities(workspace: Path) -> list[object]:
    try:
        harness = __import__("pydantic_ai_harness", fromlist=["CodeMode"])
        monty = __import__("pydantic_monty", fromlist=["MountDir"])
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "style='code' requires pydantic-ai-harness with the codemode extra. "
            "Install it with: pip install 'pydantic-ai-harness[codemode]'. "
            f"The integration targets pydantic-ai-harness {_HARNESS_TARGET_VERSION}. "
            f"Original error: {exc}"
        ) from exc
    return [
        harness.CodeMode(
            mount=monty.MountDir(
                virtual_path=_CODE_VIRTUAL_ROOT,
                host_path=workspace,
                mode="read-only",
            )
        )
    ]


def style_capabilities(style: ExtractionStyle, workspace: Path) -> list[object]:
    """Return Pydantic AI capabilities that implement ``style``."""
    match style:
        case ExtractionStyle.SEARCH:
            return _search_capabilities(workspace)
        case ExtractionStyle.CODE:
            return _code_capabilities(workspace)
        case ExtractionStyle.DIRECT | ExtractionStyle.TABLE | ExtractionStyle.FORM:
            return []
        case _:  # pragma: no cover - exhaustive ExtractionStyle
            assert_never(style)


def style_run_inputs(style: ExtractionStyle, filename: str) -> list[str]:
    """Return the user prompt for a search or code extraction."""
    match style:
        case ExtractionStyle.SEARCH:
            return [
                "Extract the requested information from the document "
                f"{filename!r} in the workspace. Use search_files, read_file, "
                "find_files, list_directory, and file_info to inspect it. Search "
                "before reading large files; do not assume unseen contents."
            ]
        case ExtractionStyle.CODE:
            return [
                "Extract the requested information by writing Python against "
                f"{_CODE_VIRTUAL_ROOT}/{filename}. Read the file with pathlib or "
                "open(), then parse, filter, and compute the structured result."
            ]
        case ExtractionStyle.DIRECT | ExtractionStyle.TABLE | ExtractionStyle.FORM:
            raise ValueError(f"style {style.value!r} does not use workspace run inputs.")
        case _:  # pragma: no cover - exhaustive ExtractionStyle
            assert_never(style)


@contextmanager
def prepared_style_run(
    style: ExtractionStyle,
    file_bytes: bytes,
    file_type: str,
) -> Iterator[tuple[list[object], list[str] | None]]:
    """Yield ``(extra_capabilities, run_inputs)`` for one extraction.

    ``direct``, ``table``, and ``form`` yield no extra capabilities and ``None``
    inputs so the caller can pass media as ``BinaryContent`` (or parsed page
    text). ``search`` and ``code`` materialize a UTF-8 workspace that lives
    until the context exits, including retries.
    """
    if not uses_workspace(style):
        yield [], None
        return
    with tempfile.TemporaryDirectory(prefix="openextract-") as tmp:
        workspace = Path(tmp)
        filename = materialize_text_document(workspace, file_bytes, file_type, style=style)
        yield style_capabilities(style, workspace), style_run_inputs(style, filename)
