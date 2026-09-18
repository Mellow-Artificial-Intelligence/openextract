"""One-shot extraction APIs and internal re-exports."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, TypeVar, cast

import httpx
from pydantic import BaseModel

from ._agent import (
    Agent,
    _build_agent,
    _install_hint,
    _model_identifier,
    _resolve_run_inputs,
    _route_model,
    _run_extraction,
    _run_extraction_async,
    _usage_from_result,
)
from ._agents import (
    DefinedAgent,
    RemoteAgent,
    flatten_agent,
    is_agent,
    resolve_output_schema,
)
from ._batch import (
    _run_with_shared_agent,
    _run_with_shared_agent_result,
    extract_many,
    extract_many_async,
    extract_many_with_results,
    extract_many_with_results_async,
    iter_extract_many_async,
)
from ._citations import prepare_cited_run
from ._config import (
    _DEFAULT_RETRY_MAX_BACKOFF,
    _max_redirects,
    _resolve_max_input_bytes,
    _url_fetch_timeout,
    _validate_cite_min_confidence,
    _validate_pages,
    _validate_retry_options,
)
from ._errors import (
    _extraction_errors,
    _is_transient_model_exception,
    _map_exception,
    _model_retry_after,
    _model_status_code,
    _parse_retry_after,
)
from ._media import (
    _get_media,
    _get_media_async,
    _get_media_type,
    _is_public_ip,
    _is_safe_host,
    _item_source_label,
    _read_from_path,
    _read_url_with_client,
    _safe_source_context,
)
from ._parse import ParsedDocument, maybe_parsed_inputs
from ._retry import _retry_delay
from ._session import AsyncExtractor, Extractor
from ._styles import (
    ExtractionStyle,
    compose_extract_instructions,
    normalize_language,
    normalize_style,
    prepared_style_run,
    should_parse,
)
from ._swarm import (
    extract_swarm,
    extract_swarm_async,
    extract_swarm_with_results,
    extract_swarm_with_results_async,
)
from ._types import (
    Citation,
    ExtractionInput,
    ExtractionInputLike,
    ExtractionResult,
    ExtractProgress,
    RetryPolicy,
    Usage,
    _resolve_item,
    total_usage,
)
from ._windows import extract_windows_async, extract_windows_sync

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models import Model

T = TypeVar("T", bound=BaseModel)


@contextmanager
def _bind_agent_inputs(
    schema: type[BaseModel],
    model: str | Model,
    instructions: str | None,
    file_bytes: bytes,
    file_type: str,
    style: ExtractionStyle,
    cite: bool,
    pages: Sequence[int] | None = None,
    language: str | None = None,
) -> Iterator[tuple[PydanticAgent, list, ParsedDocument | None]]:
    """Build the agent and run inputs after media has already been loaded."""
    run_schema, run_instructions = prepare_cited_run(
        schema, compose_extract_instructions(instructions, style, language), cite
    )
    parsed_inputs, parsed = maybe_parsed_inputs(
        file_bytes, file_type, parse=should_parse(cite, style, pages), pages=pages
    )
    with prepared_style_run(style, file_bytes, file_type) as (capabilities, style_inputs):
        with _extraction_errors():
            agent = _build_agent(
                run_schema,
                model,
                run_instructions,
                extra_capabilities=capabilities,
            )
        inputs = (
            parsed_inputs
            if parsed_inputs is not None and style_inputs is None
            else _resolve_run_inputs(file_bytes, file_type, style_inputs)
        )
        yield agent, inputs, parsed


@contextmanager
def _prepare_extraction(
    schema: type[BaseModel],
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int,
    style: ExtractionStyle,
    cite: bool = False,
    pages: Sequence[int] | None = None,
    language: str | None = None,
) -> Iterator[tuple[PydanticAgent, list, ParsedDocument | None]]:
    """Prepare one extraction while applying the public exception mapping."""
    with _extraction_errors():
        file_bytes, file_type = _get_media(
            input_file,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
        )
    with _bind_agent_inputs(
        schema, model, instructions, file_bytes, file_type, style, cite, pages, language
    ) as prepared:
        yield prepared


@asynccontextmanager
async def _prepare_extraction_async(
    schema: type[BaseModel],
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int,
    style: ExtractionStyle,
    cite: bool = False,
    client: httpx.AsyncClient | None = None,
    pages: Sequence[int] | None = None,
    language: str | None = None,
) -> AsyncIterator[tuple[PydanticAgent, list, ParsedDocument | None]]:
    """Prepare one async extraction while applying public exception mapping."""
    with _extraction_errors():
        file_bytes, file_type = await _get_media_async(
            input_file,
            client,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
        )
    with _bind_agent_inputs(
        schema, model, instructions, file_bytes, file_type, style, cite, pages, language
    ) as prepared:
        yield prepared


def _extract_once(
    agent: PydanticAgent,
    inputs: list,
) -> T:
    """Perform a single sync extraction attempt; return the schema instance."""
    result = _run_extraction(agent, inputs)
    return cast(T, result.output)


def _resolve_agent_call(
    schema: Any,
    model: Any,
    input_file: Any,
) -> tuple[type[T], Any, ExtractionInputLike]:
    """Support ``extract(agent, input_file)`` alongside ``extract(schema, model, input)``.

    An agent declaring ``output_schema`` already knows the shape it produces, so
    naming the schema again at the call site is noise.
    """
    if is_agent(schema):
        if input_file is not None:
            raise ValueError(
                "extract(agent, input_file) takes no separate model; "
                "the agent supplies both the model and the schema."
            )
        return cast("type[T]", resolve_output_schema(schema)), schema, model
    if input_file is None:
        raise ValueError("input_file is required.")
    return schema, model, input_file


def _plan_agent(
    model: Any,
    instructions: str | None,
    style: ExtractionStyle | str,
) -> tuple[Any, str | None, ExtractionStyle | str, bool]:
    """Resolve an agent to a one-shot model, or defer it to the swarm.

    An agent that flattens to a single local model is just a preconfigured
    one-shot call; anything wider (subagents, or a remote endpoint) is a swarm.
    Returns ``(model, instructions, style, use_swarm)``.
    """
    if not is_agent(model):
        return model, instructions, style, False
    members = flatten_agent(model)
    member = members[0] if len(members) == 1 else None
    if member is None or isinstance(member.model, RemoteAgent):
        return model, instructions, style, True
    return (
        member.model,
        member.instructions if member.instructions is not None else instructions,
        member.style if member.style is not None else style,
        False,
    )


def _oneshot_options(
    style: ExtractionStyle | str,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
    max_input_bytes: int | None,
) -> tuple[ExtractionStyle, int, int, float, float]:
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    return (
        normalize_style(style),
        _resolve_max_input_bytes(max_input_bytes),
        max_retries,
        retry_backoff,
        retry_max_backoff,
    )


def _resolve_oneshot(
    schema: Any,
    model: Any,
    input_file: Any,
    instructions: str | None,
    style: ExtractionStyle | str,
) -> tuple[type[T], Any, ExtractionInputLike, str | None, ExtractionStyle | str, bool]:
    """Resolve agent call forms and decide whether this run is a swarm."""
    schema, model, input_file = _resolve_agent_call(schema, model, input_file)
    model, instructions, style, use_swarm = _plan_agent(model, instructions, style)
    return schema, model, input_file, instructions, style, use_swarm


def _swarm_kwargs(
    *,
    style: ExtractionStyle | str,
    media_type: str | None,
    max_input_bytes: int | None,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
    cite: bool,
    cite_min_confidence: float | None,
    pages: Sequence[int] | None,
    language: str | None,
    on_progress: Callable[[ExtractProgress], None] | None,
) -> dict[str, Any]:
    """Keyword arguments shared by every oneshot-to-swarm dispatch."""
    return {
        "style": style,
        "media_type": media_type,
        "max_input_bytes": max_input_bytes,
        "max_retries": max_retries,
        "retry_backoff": retry_backoff,
        "retry_max_backoff": retry_max_backoff,
        "cite": cite,
        "cite_min_confidence": cite_min_confidence,
        "pages": pages,
        "language": language,
        "on_progress": on_progress,
    }


def _extract_sync(
    schema: type[T] | DefinedAgent | RemoteAgent,
    model: str | Model | DefinedAgent | RemoteAgent | ExtractionInputLike,
    input_file: ExtractionInputLike | None,
    instructions: str | None,
    *,
    style: ExtractionStyle | str,
    media_type: str | None,
    max_input_bytes: int | None,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
    cite: bool,
    cite_min_confidence: float | None,
    pages: Sequence[int] | None,
    language: str | None,
    with_usage: bool,
    on_progress: Callable[[ExtractProgress], None] | None,
) -> tuple[T, Usage, tuple[Citation, ...]]:
    """Shared sync oneshot path used by ``extract`` and ``extract_with_usage``."""
    cite_min_confidence = _validate_cite_min_confidence(cite_min_confidence)
    pages = _validate_pages(pages)
    language = normalize_language(language)
    schema, model, input_file, instructions, style, use_swarm = _resolve_oneshot(
        schema, model, input_file, instructions, style
    )
    if use_swarm:
        swarm_kw = _swarm_kwargs(
            style=style,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            cite=cite,
            cite_min_confidence=cite_min_confidence,
            pages=pages,
            language=language,
            on_progress=on_progress,
        )
        if with_usage:
            swarm = extract_swarm_with_results(schema, model, input_file, instructions, **swarm_kw)
            return swarm.output, swarm.usage, swarm.citations
        return (
            extract_swarm(schema, model, input_file, instructions, **swarm_kw),
            Usage(0, 0, 0),
            (),
        )
    style, limit, max_retries, retry_backoff, retry_max_backoff = _oneshot_options(
        style, max_retries, retry_backoff, retry_max_backoff, max_input_bytes
    )
    with _prepare_extraction(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
        style,
        cite,
        pages,
        language,
    ) as (agent, inputs, parsed):

        def _run(window: list) -> tuple[object, Usage]:
            if with_usage:
                result = _run_extraction(agent, window)
                return result.output, _usage_from_result(result)
            return _extract_once(agent, window), Usage(0, 0, 0)

        return extract_windows_sync(
            _run,
            inputs,
            parsed,
            schema,
            cite,
            cite_min_confidence=cite_min_confidence,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            on_progress=on_progress,
        )


async def _extract_async(
    schema: type[T] | DefinedAgent | RemoteAgent,
    model: str | Model | DefinedAgent | RemoteAgent | ExtractionInputLike,
    input_file: ExtractionInputLike | None,
    instructions: str | None,
    *,
    style: ExtractionStyle | str,
    media_type: str | None,
    max_input_bytes: int | None,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
    cite: bool,
    cite_min_confidence: float | None,
    pages: Sequence[int] | None,
    language: str | None,
    with_usage: bool,
    on_progress: Callable[[ExtractProgress], None] | None,
) -> tuple[T, Usage, tuple[Citation, ...]]:
    """Shared async oneshot path used by the async extract entry points."""
    cite_min_confidence = _validate_cite_min_confidence(cite_min_confidence)
    pages = _validate_pages(pages)
    language = normalize_language(language)
    schema, model, input_file, instructions, style, use_swarm = _resolve_oneshot(
        schema, model, input_file, instructions, style
    )
    if use_swarm:
        swarm_kw = _swarm_kwargs(
            style=style,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            cite=cite,
            cite_min_confidence=cite_min_confidence,
            pages=pages,
            language=language,
            on_progress=on_progress,
        )
        if with_usage:
            swarm = await extract_swarm_with_results_async(
                schema, model, input_file, instructions, **swarm_kw
            )
            return swarm.output, swarm.usage, swarm.citations
        return (
            await extract_swarm_async(schema, model, input_file, instructions, **swarm_kw),
            Usage(0, 0, 0),
            (),
        )
    style, limit, max_retries, retry_backoff, retry_max_backoff = _oneshot_options(
        style, max_retries, retry_backoff, retry_max_backoff, max_input_bytes
    )
    async with _prepare_extraction_async(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
        style,
        cite,
        pages=pages,
        language=language,
    ) as (agent, inputs, parsed):

        async def _run(window: list) -> tuple[object, Usage]:
            result = await _run_extraction_async(agent, window)
            return result.output, _usage_from_result(result) if with_usage else Usage(0, 0, 0)

        return await extract_windows_async(
            _run,
            inputs,
            parsed,
            schema,
            cite,
            cite_min_confidence=cite_min_confidence,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            on_progress=on_progress,
        )


def extract(
    schema: type[T] | DefinedAgent | RemoteAgent,
    model: str | Model | DefinedAgent | RemoteAgent | ExtractionInputLike,
    input_file: ExtractionInputLike | None = None,
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
    cite: bool = False,
    cite_min_confidence: float | None = None,
    pages: Sequence[int] | None = None,
    language: str | None = None,
    on_progress: Callable[[ExtractProgress], None] | None = None,
) -> T:
    """
    Extract structured data from a document, image, audio, or video file using an LLM.

    Also callable as ``extract(agent, input_file)`` when the agent declares an
    ``output_schema``, and as ``extract(schema, agent, input_file)`` for any
    agent. An agent that resolves to a single local model runs as a normal
    one-shot call with the agent's model, instructions, and style; an agent
    with subagents or a remote endpoint runs as a swarm and its outputs are
    merged.

    Args:
        schema: A Pydantic model class defining the expected output structure,
            or an agent from ``define_agent`` / ``define_remote_agent`` that
            declares its own ``output_schema``.
        model: The model identifier (e.g., 'xai:grok-4.3'), a configured
            pydantic-ai ``Model``, or an agent. When ``schema`` is an agent,
            this is the input instead.
        input_file: A local file path, an ``http(s)://`` URL, raw ``bytes``, or
            a binary file-like object with a ``.read()`` method. For ``bytes``
            and file-like inputs, ``media_type`` must be provided.
        instructions: Optional natural-language guidance for the LLM.
        style: Extraction strategy. ``direct`` (default) sends the media to the
            model in one shot. ``table`` adds row-oriented guidance; ``form``
            adds labeled-field guidance. Both reuse PDF parse-then-window.
            ``search`` gives the model file tools (grep/read) against a text
            document. ``code`` lets the model write Python against a text
            document via the Pydantic AI harness.
        media_type: Optional MIME type. Required for ``bytes`` and file-like
            inputs; overrides the guess for ``str`` inputs when provided.
        max_input_bytes: Maximum bytes to load for this input. ``None`` uses
            ``OPENEXTRACT_MAX_INPUT_BYTES`` or the 50 MiB default.
        max_retries: Number of additional attempts after a transient
            ``ModelError``. Defaults to 0 (no retries, single attempt).
        retry_backoff: Base backoff in seconds. Sleep between attempts is
            ``retry_backoff * (2 ** attempt) * (1 + random.uniform(0, 0.25))``,
            i.e. exponential backoff with up to 25% jitter.
        retry_max_backoff: Maximum delay in seconds for exponential backoff or
            a provider ``Retry-After`` value. Defaults to 60 seconds.
        cite: When ``True``, the model is asked for per-field source spans.
            PDFs are parsed locally first; boxes come from parser spans, not
            the model. ``extract`` still returns the schema instance; citations
            land on :class:`ExtractionResult` from the ``*_with_results`` APIs.
        cite_min_confidence: When ``cite=True``, keep only citations whose
            heuristic ``confidence`` is not ``None`` and is at least this
            threshold in ``[0, 1]``. ``None`` (default) keeps every citation.
            Invalid values raise ``ValueError`` at call time.
        pages: Optional 1-based PDF page numbers to extract. When set, only
            those pages are considered for local parse-then-window and
            citation grounding. Out-of-range numbers are ignored; if none
            remain, raises ``ValueError``. ``None`` (default) keeps every
            page. Invalid values raise ``ValueError`` at call time. Has no
            effect when the input is not locally parsed.
        language: Optional BCP-47-ish tag or plain name (``en``, ``es``,
            ``fr``). When set, the model is told the document's primary
            language and to preserve that language/script in field values.
            ``None`` (default) leaves instructions unchanged. Empty values
            raise ``ValueError``.
        on_progress: Optional callback invoked once per parse window immediately
            before that window is sent to the model. Receives
            :class:`ExtractProgress` (1-indexed ``current`` / ``total``, plus
            ``page`` / ``pages`` when the input was parsed). Default ``None``
            keeps the silent path. A raising callback aborts the extraction.

    Returns:
        An instance of the schema populated with extracted data.

    Raises:
        TypeError: If ``input_file`` is bytes or file-like and ``media_type``
            is not provided.
        ValueError: If ``input_file`` is omitted, or an agent is passed as
            ``schema`` together with a separate model.
        InputTooLargeError: If the resolved input exceeds ``max_input_bytes``.
        UrlFetchError: If the URL cannot be fetched or returns a non-2xx status.
        SchemaValidationError: If the model output doesn't match the schema.
        ModelError: If retries (if any) are exhausted.
        ProviderNotInstalledError: If a provider SDK or style extra is missing.
        ExtractionError: For other extraction failures.
        ValueError: If ``style`` is invalid, ``search``/``code`` is used with
            a non-text document, or ``cite_min_confidence`` is outside ``[0, 1]``.
            Also raised if ``pages`` is empty/invalid or matches no PDF page,
            or ``language`` is empty.
    """
    output, _usage, _citations = _extract_sync(
        schema,
        model,
        input_file,
        instructions,
        style=style,
        media_type=media_type,
        max_input_bytes=max_input_bytes,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_max_backoff=retry_max_backoff,
        cite=cite,
        cite_min_confidence=cite_min_confidence,
        pages=pages,
        language=language,
        with_usage=False,
        on_progress=on_progress,
    )
    return output


def extract_with_usage(
    schema: type[T] | DefinedAgent | RemoteAgent,
    model: str | Model | DefinedAgent | RemoteAgent | ExtractionInputLike,
    input_file: ExtractionInputLike | None = None,
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
    cite: bool = False,
    cite_min_confidence: float | None = None,
    pages: Sequence[int] | None = None,
    language: str | None = None,
    on_progress: Callable[[ExtractProgress], None] | None = None,
) -> tuple[T, Usage]:
    """Extract structured data and return ``(output, Usage)`` for token accounting.

    Same retry, agent, ``cite`` / ``cite_min_confidence``, ``pages``, and
    ``on_progress`` semantics as :func:`extract`.
    Returns a :class:`Usage` describing the tokens consumed by the successful
    model call, or summed across the agents when an agent fans out into a swarm.
    """
    output, usage, _citations = _extract_sync(
        schema,
        model,
        input_file,
        instructions,
        style=style,
        media_type=media_type,
        max_input_bytes=max_input_bytes,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_max_backoff=retry_max_backoff,
        cite=cite,
        cite_min_confidence=cite_min_confidence,
        pages=pages,
        language=language,
        with_usage=True,
        on_progress=on_progress,
    )
    return output, usage


async def extract_with_usage_async(
    schema: type[T] | DefinedAgent | RemoteAgent,
    model: str | Model | DefinedAgent | RemoteAgent | ExtractionInputLike,
    input_file: ExtractionInputLike | None = None,
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
    cite: bool = False,
    cite_min_confidence: float | None = None,
    pages: Sequence[int] | None = None,
    language: str | None = None,
    on_progress: Callable[[ExtractProgress], None] | None = None,
) -> tuple[T, Usage]:
    """Async sibling of :func:`extract_with_usage`; returns ``(output, Usage)``."""
    output, usage, _citations = await _extract_async(
        schema,
        model,
        input_file,
        instructions,
        style=style,
        media_type=media_type,
        max_input_bytes=max_input_bytes,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_max_backoff=retry_max_backoff,
        cite=cite,
        cite_min_confidence=cite_min_confidence,
        pages=pages,
        language=language,
        with_usage=True,
        on_progress=on_progress,
    )
    return output, usage


async def extract_async(
    schema: type[T] | DefinedAgent | RemoteAgent,
    model: str | Model | DefinedAgent | RemoteAgent | ExtractionInputLike,
    input_file: ExtractionInputLike | None = None,
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
    cite: bool = False,
    cite_min_confidence: float | None = None,
    pages: Sequence[int] | None = None,
    language: str | None = None,
    on_progress: Callable[[ExtractProgress], None] | None = None,
) -> T:
    """Async sibling of :func:`extract`; uses ``Agent.run`` instead of ``run_sync``.

    Accepts the same agent forms as :func:`extract`.
    """
    output, _usage, _citations = await _extract_async(
        schema,
        model,
        input_file,
        instructions,
        style=style,
        media_type=media_type,
        max_input_bytes=max_input_bytes,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_max_backoff=retry_max_backoff,
        cite=cite,
        cite_min_confidence=cite_min_confidence,
        pages=pages,
        language=language,
        with_usage=False,
        on_progress=on_progress,
    )
    return output


__all__ = [
    "Agent",
    "Extractor",
    "AsyncExtractor",
    "RetryPolicy",
    "ExtractionInput",
    "ExtractionResult",
    "Usage",
    "extract",
    "extract_async",
    "extract_with_usage",
    "extract_with_usage_async",
    "extract_many",
    "extract_many_async",
    "iter_extract_many_async",
    "extract_many_with_results",
    "extract_many_with_results_async",
    "total_usage",
    "_build_agent",
    "_extract_once",
    "_get_media",
    "_get_media_async",
    "_get_media_type",
    "_install_hint",
    "_is_public_ip",
    "_is_safe_host",
    "_is_transient_model_exception",
    "_item_source_label",
    "_map_exception",
    "_max_redirects",
    "_model_identifier",
    "_model_retry_after",
    "_model_status_code",
    "_parse_retry_after",
    "_plan_agent",
    "_prepare_extraction",
    "_resolve_agent_call",
    "_read_from_path",
    "_read_url_with_client",
    "_resolve_item",
    "_resolve_max_input_bytes",
    "_resolve_run_inputs",
    "_retry_delay",
    "_route_model",
    "_run_extraction",
    "_run_with_shared_agent",
    "_run_with_shared_agent_result",
    "_safe_source_context",
    "_url_fetch_timeout",
]
