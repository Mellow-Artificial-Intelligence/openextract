"""One-shot extraction APIs and internal re-exports."""

from __future__ import annotations

import time
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
    _session_model_settings,
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
    _resolve_url_timeout,
    _url_fetch_timeout,
    _validate_cite_min_confidence,
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
    normalize_style,
    prepared_style_run,
    should_parse,
)
from ._swarm import (
    SwarmResult,
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
    _extraction_result,
    _resolve_item,
    _resolve_item_options,
    total_usage,
)
from ._windows import extract_windows_async, extract_windows_sync

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings

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
    model_settings: ModelSettings | None = None,
    max_pages: int | None = None,
) -> Iterator[tuple[PydanticAgent, list, ParsedDocument | None]]:
    """Build the agent and run inputs after media has already been loaded."""
    run_schema, run_instructions = prepare_cited_run(
        schema, compose_extract_instructions(instructions, style, language), cite
    )
    parsed_inputs, parsed = maybe_parsed_inputs(
        file_bytes,
        file_type,
        parse=should_parse(cite, style, pages, max_pages),
        pages=pages,
        max_pages=max_pages,
    )
    with prepared_style_run(style, file_bytes, file_type) as (capabilities, style_inputs):
        with _extraction_errors():
            agent = _build_agent(
                run_schema,
                model,
                run_instructions,
                model_settings=model_settings,
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
    model_settings: ModelSettings | None = None,
    url_timeout: float | None = None,
    max_pages: int | None = None,
) -> Iterator[tuple[PydanticAgent, list, ParsedDocument | None]]:
    """Prepare one extraction while applying the public exception mapping."""
    with _extraction_errors():
        file_bytes, file_type = _get_media(
            input_file,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            url_timeout=url_timeout,
        )
    with _bind_agent_inputs(
        schema,
        model,
        instructions,
        file_bytes,
        file_type,
        style,
        cite,
        pages,
        language,
        model_settings=model_settings,
        max_pages=max_pages,
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
    model_settings: ModelSettings | None = None,
    url_timeout: float | None = None,
    max_pages: int | None = None,
) -> AsyncIterator[tuple[PydanticAgent, list, ParsedDocument | None]]:
    """Prepare one async extraction while applying public exception mapping."""
    with _extraction_errors():
        file_bytes, file_type = await _get_media_async(
            input_file,
            client,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            url_timeout=url_timeout,
        )
    with _bind_agent_inputs(
        schema,
        model,
        instructions,
        file_bytes,
        file_type,
        style,
        cite,
        pages,
        language,
        model_settings=model_settings,
        max_pages=max_pages,
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
    max_pages: int | None,
    language: str | None,
    model_settings: ModelSettings | None,
    timeout: float | None,
    url_timeout: float | None,
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
        "max_pages": max_pages,
        "language": language,
        "model_settings": model_settings,
        "timeout": timeout,
        "url_timeout": url_timeout,
        "on_progress": on_progress,
    }


def _oneshot_provenance(
    input_file: ExtractionInputLike,
    media_type: str | None,
) -> tuple[str | None, str | None]:
    """Requested media type and sanitized source label for a oneshot result."""
    source, item_media_type, name = _resolve_item(input_file, media_type)
    return item_media_type, _item_source_label(source, name)


def _result_from_swarm(
    swarm: SwarmResult[T],
    *,
    started: float,
    media_type: str | None,
    source: str | None,
) -> ExtractionResult[T]:
    """Summarize a oneshot swarm the same way ``extract_with_usage`` does."""
    successes = [agent for agent in swarm.agents if isinstance(agent, ExtractionResult)]
    return _extraction_result(
        swarm.output,
        swarm.usage,
        attempts=sum(agent.attempts for agent in successes),
        started=started,
        model=successes[0].model,
        media_type=media_type,
        source=source,
        citations=swarm.citations,
    )


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
    max_pages: int | None,
    language: str | None,
    model_settings: ModelSettings | None,
    timeout: float | None,
    url_timeout: float | None,
    with_usage: bool,
    on_progress: Callable[[ExtractProgress], None] | None,
    rich: bool = False,
) -> tuple[T, Usage, tuple[Citation, ...]] | ExtractionResult[T]:
    """Shared sync oneshot path used by ``extract`` and the usage/result helpers."""
    cite_min_confidence = _validate_cite_min_confidence(cite_min_confidence)
    run_settings = _session_model_settings(model_settings, timeout)
    url_timeout = _resolve_url_timeout(url_timeout)
    schema, model, input_file, instructions, style, use_swarm = _resolve_oneshot(
        schema, model, input_file, instructions, style
    )
    pages, max_pages, language = _resolve_item_options(input_file, pages, max_pages, language)
    started = time.perf_counter()
    item_media_type, source_label = _oneshot_provenance(input_file, media_type)
    need_usage = with_usage or rich
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
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            on_progress=on_progress,
        )
        if need_usage:
            swarm = extract_swarm_with_results(schema, model, input_file, instructions, **swarm_kw)
            if rich:
                return _result_from_swarm(
                    swarm,
                    started=started,
                    media_type=item_media_type,
                    source=source_label,
                )
            return swarm.output, swarm.usage, swarm.citations
        return (
            extract_swarm(schema, model, input_file, instructions, **swarm_kw),
            Usage(0, 0, 0),
            (),
        )
    style, limit, max_retries, retry_backoff, retry_max_backoff = _oneshot_options(
        style, max_retries, retry_backoff, retry_max_backoff, max_input_bytes
    )
    attempts = 0
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
        model_settings=run_settings,
        url_timeout=url_timeout,
        max_pages=max_pages,
    ) as (agent, inputs, parsed):

        def _run(window: list) -> tuple[object, Usage]:
            nonlocal attempts
            if need_usage:
                attempts += 1
                result = _run_extraction(agent, window)
                return result.output, _usage_from_result(result)
            return _extract_once(agent, window), Usage(0, 0, 0)

        output, usage, citations = extract_windows_sync(
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
    if rich:
        return _extraction_result(
            output,
            usage,
            attempts=attempts,
            started=started,
            model=_model_identifier(model, agent),
            media_type=item_media_type,
            source=source_label,
            citations=citations,
        )
    return output, usage, citations


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
    max_pages: int | None,
    language: str | None,
    model_settings: ModelSettings | None,
    timeout: float | None,
    url_timeout: float | None,
    with_usage: bool,
    on_progress: Callable[[ExtractProgress], None] | None,
    rich: bool = False,
) -> tuple[T, Usage, tuple[Citation, ...]] | ExtractionResult[T]:
    """Shared async oneshot path used by the async extract entry points."""
    cite_min_confidence = _validate_cite_min_confidence(cite_min_confidence)
    run_settings = _session_model_settings(model_settings, timeout)
    url_timeout = _resolve_url_timeout(url_timeout)
    schema, model, input_file, instructions, style, use_swarm = _resolve_oneshot(
        schema, model, input_file, instructions, style
    )
    pages, max_pages, language = _resolve_item_options(input_file, pages, max_pages, language)
    started = time.perf_counter()
    item_media_type, source_label = _oneshot_provenance(input_file, media_type)
    need_usage = with_usage or rich
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
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            on_progress=on_progress,
        )
        if need_usage:
            swarm = await extract_swarm_with_results_async(
                schema, model, input_file, instructions, **swarm_kw
            )
            if rich:
                return _result_from_swarm(
                    swarm,
                    started=started,
                    media_type=item_media_type,
                    source=source_label,
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
    attempts = 0
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
        model_settings=run_settings,
        url_timeout=url_timeout,
        max_pages=max_pages,
    ) as (agent, inputs, parsed):

        async def _run(window: list) -> tuple[object, Usage]:
            nonlocal attempts
            if need_usage:
                attempts += 1
            result = await _run_extraction_async(agent, window)
            return result.output, _usage_from_result(result) if need_usage else Usage(0, 0, 0)

        output, usage, citations = await extract_windows_async(
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
    if rich:
        return _extraction_result(
            output,
            usage,
            attempts=attempts,
            started=started,
            model=_model_identifier(model, agent),
            media_type=item_media_type,
            source=source_label,
            citations=citations,
        )
    return output, usage, citations


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
    max_pages: int | None = None,
    language: str | None = None,
    model_settings: ModelSettings | None = None,
    timeout: float | None = None,
    url_timeout: float | None = None,
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
            land on :class:`ExtractionResult` from :func:`extract_with_result`
            and the ``*_with_results`` APIs.
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
        max_pages: Optional positive cap on 1-based PDF page numbers. When
            set, only pages with number ``<= N`` are considered after any
            ``pages`` filter (``pages`` ``None`` is treated as ``1..N``).
            Invalid values raise ``ValueError`` at call time. If no pages
            remain, raises ``ValueError``. Accepted for non-paginated inputs
            with no effect.
        language: Optional BCP-47-ish tag or plain name (``en``, ``es``,
            ``fr``). When set, the model is told the document's primary
            language and to preserve that language/script in field values.
            ``None`` (default) leaves instructions unchanged. Empty values
            raise ``ValueError``.
        model_settings: Optional Pydantic AI ``ModelSettings`` passed to the
            constructed agent. ``None`` (default) leaves provider defaults.
        timeout: Optional model request timeout in seconds. Overrides a
            ``timeout`` entry in ``model_settings``. ``None`` (default) leaves
            the provider default. Invalid values raise ``ValueError`` before
            any model call.
        url_timeout: Optional HTTP timeout in seconds for fetching ``http(s)``
            URL inputs. ``None`` (default) uses ``OPENEXTRACT_URL_TIMEOUT`` or
            30 seconds. Invalid values raise ``ValueError`` before any fetch
            or model call. Same contract as session ``url_timeout``.
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
        InputFileError: If a local path or file-like input cannot be opened
            or read.
        InputTooLargeError: If the resolved input exceeds ``max_input_bytes``.
        UrlFetchError: If the URL cannot be fetched or returns a non-2xx status.
        SchemaValidationError: If the model output doesn't match the schema.
        ModelError: If retries (if any) are exhausted.
        ProviderNotInstalledError: If a provider SDK or style extra is missing.
        ExtractionError: For other extraction failures.
        ValueError: If ``style`` is invalid, ``search``/``code`` is used with
            a non-text document, or ``cite_min_confidence`` is outside ``[0, 1]``.
            Also raised if ``pages`` is empty/invalid or matches no PDF page,
            ``max_pages`` is invalid or filters out every requested page,
            ``language`` is empty, ``timeout`` is not a finite positive
            number of seconds, or ``url_timeout`` is not a finite positive
            number of seconds.
    """
    output, _usage, _citations = cast(
        "tuple[T, Usage, tuple[Citation, ...]]",
        _extract_sync(
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
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            with_usage=False,
            on_progress=on_progress,
        ),
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
    max_pages: int | None = None,
    language: str | None = None,
    model_settings: ModelSettings | None = None,
    timeout: float | None = None,
    url_timeout: float | None = None,
    on_progress: Callable[[ExtractProgress], None] | None = None,
) -> tuple[T, Usage]:
    """Extract structured data and return ``(output, Usage)`` for token accounting.

    Same retry, agent, ``cite`` / ``cite_min_confidence``, ``pages``,
    ``max_pages``, ``model_settings``, ``timeout``, ``url_timeout``, and ``on_progress``
    semantics as :func:`extract`.
    Returns a :class:`Usage` describing the tokens consumed by the successful
    model call, or summed across the agents when an agent fans out into a swarm.
    """
    output, usage, _citations = cast(
        "tuple[T, Usage, tuple[Citation, ...]]",
        _extract_sync(
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
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            with_usage=True,
            on_progress=on_progress,
        ),
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
    max_pages: int | None = None,
    language: str | None = None,
    model_settings: ModelSettings | None = None,
    timeout: float | None = None,
    url_timeout: float | None = None,
    on_progress: Callable[[ExtractProgress], None] | None = None,
) -> tuple[T, Usage]:
    """Async sibling of :func:`extract_with_usage`; returns ``(output, Usage)``."""
    output, usage, _citations = cast(
        "tuple[T, Usage, tuple[Citation, ...]]",
        await _extract_async(
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
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            with_usage=True,
            on_progress=on_progress,
        ),
    )
    return output, usage


def extract_with_result(
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
    max_pages: int | None = None,
    language: str | None = None,
    model_settings: ModelSettings | None = None,
    timeout: float | None = None,
    url_timeout: float | None = None,
    on_progress: Callable[[ExtractProgress], None] | None = None,
) -> ExtractionResult[T]:
    """Extract structured data and return an :class:`ExtractionResult`.

    Same arguments and retry, agent, ``cite`` / ``cite_min_confidence``,
    ``pages``, ``max_pages``, ``model_settings``, ``timeout``, ``url_timeout``, and
    ``on_progress`` semantics as :func:`extract`. The result carries the schema instance plus token
    usage, attempt count, duration, model/media metadata, a sanitized source
    label, and citations when ``cite=True`` — the same fields
    :func:`extract_many_with_results` fills.

    When an agent fans out into a swarm, usage is summed across successful
    agents and citations are the reduced swarm set, matching
    :func:`extract_with_usage`. Use :func:`extract_swarm_with_results` when
    per-agent results are needed.
    """
    return cast(
        ExtractionResult[T],
        _extract_sync(
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
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            with_usage=True,
            on_progress=on_progress,
            rich=True,
        ),
    )


async def extract_with_result_async(
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
    max_pages: int | None = None,
    language: str | None = None,
    model_settings: ModelSettings | None = None,
    timeout: float | None = None,
    url_timeout: float | None = None,
    on_progress: Callable[[ExtractProgress], None] | None = None,
) -> ExtractionResult[T]:
    """Async sibling of :func:`extract_with_result`."""
    return cast(
        ExtractionResult[T],
        await _extract_async(
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
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            with_usage=True,
            on_progress=on_progress,
            rich=True,
        ),
    )


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
    max_pages: int | None = None,
    language: str | None = None,
    model_settings: ModelSettings | None = None,
    timeout: float | None = None,
    url_timeout: float | None = None,
    on_progress: Callable[[ExtractProgress], None] | None = None,
) -> T:
    """Async sibling of :func:`extract`; uses ``Agent.run`` instead of ``run_sync``.

    Accepts the same agent forms as :func:`extract`.
    """
    output, _usage, _citations = cast(
        "tuple[T, Usage, tuple[Citation, ...]]",
        await _extract_async(
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
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            with_usage=False,
            on_progress=on_progress,
        ),
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
    "extract_with_result",
    "extract_with_result_async",
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
    "_resolve_item_options",
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
