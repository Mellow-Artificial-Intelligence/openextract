"""Bounded concurrent extraction over many inputs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import httpx

from ._agent import (
    _build_agent,
    _model_identifier,
    _resolve_run_inputs,
    _run_extraction_async,
    _session_model_settings,
    _usage_from_result,
)
from ._citations import prepare_cited_run
from ._config import (
    _DEFAULT_RETRY_MAX_BACKOFF,
    _resolve_max_input_bytes,
    _resolve_url_timeout,
    _select_pages,
    _validate_cite_min_confidence,
    _validate_max_concurrency,
    _validate_retry_options,
)
from ._errors import _extraction_errors
from ._media import _get_media_async, _item_source_label
from ._parse import maybe_parsed_inputs
from ._styles import (
    ExtractionStyle,
    compose_extract_instructions,
    normalize_language,
    normalize_style,
    prepared_style_run,
    should_parse,
    uses_workspace,
)
from ._types import (
    ExtractionInputLike,
    ExtractionResult,
    ExtractProgress,
    OnProgress,
    T,
    Usage,
    _extraction_result,
    _resolve_item,
)
from ._windows import extract_windows_async

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings


@dataclass(frozen=True)
class _BatchOptions:
    """Validated knobs shared by every batch entry point.

    Bundling them keeps the internal plumbing (`_iter_extractions`,
    `_gather_extractions`, `_run_batch_sync`) from re-declaring the same long
    parameter list, and guarantees validation happens exactly once per call.
    """

    instructions: str | None
    style: ExtractionStyle
    media_type: str | None
    max_input_bytes: int
    max_concurrency: int
    return_exceptions: bool
    max_retries: int
    retry_backoff: float
    retry_max_backoff: float
    rich: bool
    cite: bool
    url_timeout: float
    cite_min_confidence: float | None = None
    pages: tuple[int, ...] | None = None
    max_pages: int | None = None
    language: str | None = None
    model_settings: ModelSettings | None = None
    on_progress: OnProgress | None = None

    @classmethod
    def resolve(
        cls,
        instructions: str | None,
        *,
        style: ExtractionStyle | str,
        media_type: str | None,
        max_input_bytes: int | None,
        max_concurrency: int,
        return_exceptions: bool,
        max_retries: int,
        retry_backoff: float,
        retry_max_backoff: float,
        rich: bool,
        cite: bool = False,
        cite_min_confidence: float | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
        language: str | None = None,
        model_settings: ModelSettings | None = None,
        timeout: float | None = None,
        url_timeout: float | None = None,
        on_progress: OnProgress | None = None,
    ) -> _BatchOptions:
        """Validate and normalize the public batch arguments."""
        _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
        _validate_max_concurrency(max_concurrency)
        selected, cap = _select_pages(pages, max_pages)
        return cls(
            instructions=instructions,
            style=normalize_style(style),
            media_type=media_type,
            max_input_bytes=_resolve_max_input_bytes(max_input_bytes),
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            rich=rich,
            cite=cite,
            url_timeout=_resolve_url_timeout(url_timeout),
            cite_min_confidence=_validate_cite_min_confidence(cite_min_confidence),
            pages=selected,
            max_pages=cap,
            language=normalize_language(language),
            model_settings=_session_model_settings(model_settings, timeout),
            on_progress=on_progress,
        )


async def _run_with_shared_agent(
    agent: PydanticAgent,
    inputs: list,
) -> object:
    """Run prepared inputs through a pre-built shared ``Agent``.

    Mirrors ``extract_async``'s error mapping so callers get the same
    ``ExtractionError`` subclasses as the per-item path. Returns the validated
    schema instance.
    """
    result = await _run_extraction_async(agent, inputs)
    return result.output


async def _run_with_shared_agent_result(
    agent: PydanticAgent,
    inputs: list,
) -> Any:
    """Run prepared inputs through a shared ``Agent`` and return the raw result.

    The raw pydantic-ai result exposes ``.output`` and ``.usage()`` so the rich
    batch path can build :class:`ExtractionResult` diagnostics.
    """
    return await _run_extraction_async(agent, inputs)


async def _cancel_tasks(tasks: Iterable[asyncio.Task[object]]) -> None:
    """Cancel and await every task so no batch work outlives its caller."""
    task_list = list(tasks)
    for task in task_list:
        task.cancel()
    if task_list:
        await asyncio.gather(*task_list, return_exceptions=True)


def _require_no_running_loop(name: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{name}() cannot be called from a running event loop; use await {name}_async(...) instead."
    )


async def _iter_extractions(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    options: _BatchOptions,
) -> AsyncIterator[tuple[int, T | ExtractionResult[T] | Exception]]:
    """Yield indexed batch results in completion order with bounded work.

    When ``options.rich`` is true each successful item is yielded as an
    :class:`ExtractionResult[T]`; otherwise the bare schema instance is yielded.
    """
    file_iterator = iter(input_files)
    try:
        first_item = next(file_iterator)
    except StopIteration:
        return

    # Building the Agent (and its provider HTTP client) is ~32 ms; sharing one
    # across the batch saves ~32 ms × (N-1) per call. The Agent is stateless
    # between runs and stays inside this event loop, so this is safe. Search and
    # code styles bind capabilities to a per-item workspace, so those items
    # each get their own agent.
    run_schema, run_instructions = prepare_cited_run(
        schema,
        compose_extract_instructions(options.instructions, options.style, options.language),
        options.cite,
    )
    shared_agent = (
        _build_agent(run_schema, model, run_instructions, model_settings=options.model_settings)
        if not uses_workspace(options.style)
        else None
    )
    stop = asyncio.Event()
    pending: dict[asyncio.Task[object], int] = {}
    next_index = 0
    exhausted = False

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=options.url_timeout,
    ) as client:

        async def _run_item(item: ExtractionInputLike) -> object:
            source, item_media_type, name = _resolve_item(item, options.media_type)
            started = time.perf_counter()
            attempts = 0
            try:
                with _extraction_errors():
                    file_bytes, file_type = await _get_media_async(
                        source,
                        client,
                        media_type=item_media_type,
                        max_input_bytes=options.max_input_bytes,
                    )
                parsed_inputs, parsed = maybe_parsed_inputs(
                    file_bytes,
                    file_type,
                    parse=should_parse(
                        options.cite, options.style, options.pages, options.max_pages
                    ),
                    pages=options.pages,
                    max_pages=options.max_pages,
                )
                with prepared_style_run(options.style, file_bytes, file_type) as (
                    capabilities,
                    style_inputs,
                ):
                    inputs = (
                        parsed_inputs
                        if parsed_inputs is not None and style_inputs is None
                        else _resolve_run_inputs(file_bytes, file_type, style_inputs)
                    )
                    if shared_agent is None:
                        with _extraction_errors():
                            run_agent = _build_agent(
                                run_schema,
                                model,
                                run_instructions,
                                model_settings=options.model_settings,
                                extra_capabilities=capabilities,
                            )
                    else:
                        run_agent = shared_agent

                    async def _run(window: list) -> tuple[object, Usage]:
                        nonlocal attempts
                        attempts += 1
                        # A sibling may have failed while this item was being prepared
                        # or waiting to retry. Do not begin another model call afterward.
                        if stop.is_set():
                            raise asyncio.CancelledError
                        if options.rich:
                            result = await _run_with_shared_agent_result(run_agent, window)
                            return result.output, _usage_from_result(result)
                        output = await _run_with_shared_agent(run_agent, window)
                        return output, Usage(0, 0, 0)

                    output, usage, citations = await extract_windows_async(
                        _run,
                        inputs,
                        parsed,
                        schema,
                        options.cite,
                        cite_min_confidence=options.cite_min_confidence,
                        max_retries=options.max_retries,
                        retry_backoff=options.retry_backoff,
                        retry_max_backoff=options.retry_max_backoff,
                        on_progress=options.on_progress,
                    )
                if options.rich:
                    return _extraction_result(
                        output,
                        usage,
                        attempts=attempts,
                        started=started,
                        model=_model_identifier(model, run_agent),
                        media_type=item_media_type,
                        source=_item_source_label(source, name),
                        citations=citations,
                    )
                return output
            except Exception:
                if not options.return_exceptions:
                    stop.set()
                raise

        def _schedule(item: ExtractionInputLike) -> None:
            nonlocal next_index
            task = asyncio.create_task(_run_item(item))
            pending[task] = next_index
            next_index += 1

        def _fill_slots() -> None:
            nonlocal exhausted
            while len(pending) < options.max_concurrency and not exhausted:
                if next_index == 0:
                    item = first_item
                else:
                    try:
                        item = next(file_iterator)
                    except StopIteration:
                        exhausted = True
                        break
                _schedule(item)

        try:
            _fill_slots()
            while pending:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                completed: list[tuple[int, T | ExtractionResult[T] | Exception]] = []
                failures: list[Exception] = []
                child_cancelled = False

                # Stable index ordering makes simultaneous completions deterministic.
                for task in sorted(done, key=pending.__getitem__):
                    index = pending.pop(task)
                    if task.cancelled():
                        child_cancelled = True
                        continue
                    try:
                        result = task.result()
                    except Exception as exc:
                        if options.return_exceptions:
                            completed.append((index, exc))
                        else:
                            failures.append(exc)
                    else:
                        completed.append((index, cast("T | ExtractionResult[T]", result)))

                if failures:
                    await _cancel_tasks(pending)
                    pending.clear()
                    raise failures[0]
                if child_cancelled:
                    raise asyncio.CancelledError

                # Refill only after every completion has been checked for a
                # fail-fast error. Pending tasks therefore stay O(concurrency).
                _fill_slots()
                for indexed_result in completed:
                    yield indexed_result
        finally:
            await _cancel_tasks(pending)
            pending.clear()


async def _gather_extractions(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    options: _BatchOptions,
) -> list:
    """Drain the streaming batch runner and restore input order."""
    indexed_results = [
        item async for item in _iter_extractions(schema, model, input_files, options)
    ]
    indexed_results.sort(key=lambda item: item[0])
    return [result for _, result in indexed_results]


def _run_batch_sync(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    options: _BatchOptions,
    *,
    name: str,
) -> list:
    """Run a batch from sync code on a private event loop."""
    _require_no_running_loop(name)
    return asyncio.run(_gather_extractions(schema, model, input_files, options))


@overload
def extract_many(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: Literal[False] = False,
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
) -> list[T]: ...


@overload
def extract_many(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: Literal[True],
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
) -> list[T | Exception]: ...


@overload
def extract_many(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool,
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
) -> list[T | Exception]: ...


def extract_many(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
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
) -> list:
    """Run :func:`extract` over many inputs concurrently from sync code.

    Each item may be a raw path/URL/``bytes``/file-like ``os.PathLike`` or an
    :class:`ExtractionInput` carrying a per-item ``media_type``.

    Args:
        schema: A Pydantic model class defining the expected output structure.
        model: The model identifier.
        input_files: Iterable of paths, URLs, ``os.PathLike``, bytes,
            file-like objects, or :class:`ExtractionInput` items.
        instructions: Optional natural-language guidance.
        style: Extraction strategy, as documented on :func:`extract`.
        media_type: Optional MIME type applied to every item that does not carry
            its own. Required when ``input_files`` contains ``bytes`` or
            file-like objects without a per-item ``media_type``; optional
            override for path/URL items.
        max_input_bytes: Per-item byte limit. ``None`` uses
            ``OPENEXTRACT_MAX_INPUT_BYTES`` or the 50 MiB default.
        max_concurrency: Maximum number of in-flight extractions.
        return_exceptions: If True, exceptions are returned in-place instead of
            raised (mirrors :func:`asyncio.gather`).
        max_retries: Per-item retries after ``ModelError`` (same semantics as
            :func:`extract`).
        retry_backoff: Base backoff seconds between per-item retries.
        retry_max_backoff: Maximum per-item retry delay in seconds.
        cite: When ``True``, request per-field citations. Bare batch APIs still
            return schema instances; ``extract_many_with_results*`` attach them
            to :class:`ExtractionResult.citations`.
        cite_min_confidence: When ``cite=True``, drop citations below this
            ``[0, 1]`` heuristic threshold after grounding. ``None`` keeps
            every citation. Invalid values raise ``ValueError`` at call time.
        pages: Optional 1-based PDF page numbers, same contract as
            :func:`extract`.
        max_pages: Optional positive page-number cap, same contract as
            :func:`extract`.
        language: Optional document language hint, same contract as
            :func:`extract`.
        model_settings: Optional Pydantic AI ``ModelSettings``, same contract
            as :func:`extract`.
        timeout: Optional model request timeout in seconds, same contract as
            :func:`extract`. Invalid values raise ``ValueError`` before any
            model call.
        url_timeout: Optional HTTP timeout in seconds for fetching ``http(s)``
            URL inputs, same contract as :func:`extract`. Invalid values raise
            ``ValueError`` before any fetch or model call.
        on_progress: Optional per-window callback, same contract as
            :func:`extract`. Concurrent items may interleave events.

    Returns:
        A list of results (or exceptions, when ``return_exceptions=True``) in
        input order.

    Raises:
        ValueError: If ``max_concurrency`` is less than 1, ``max_retries`` is
            negative, a backoff value is negative or non-finite,
            ``cite_min_confidence`` is outside ``[0, 1]``, ``pages`` is
            empty/invalid, ``max_pages`` is invalid or filters out every
            requested page, ``language`` is empty, ``timeout`` is not a
            finite positive number of seconds, or ``url_timeout`` is not a
            finite positive number of seconds.
        RuntimeError: If called from a running event loop. Use
            :func:`extract_many_async` in async code instead.
    """
    return _run_batch_sync(
        schema,
        model,
        input_files,
        _BatchOptions.resolve(
            instructions,
            style=style,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            rich=False,
            cite=cite,
            cite_min_confidence=cite_min_confidence,
            pages=pages,
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            on_progress=on_progress,
        ),
        name="extract_many",
    )


@overload
async def extract_many_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: Literal[False] = False,
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
) -> list[T]: ...


@overload
async def extract_many_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: Literal[True],
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
) -> list[T | Exception]: ...


@overload
async def extract_many_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool,
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
) -> list[T | Exception]: ...


async def extract_many_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
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
) -> list:
    """Async sibling of :func:`extract_many`."""
    return await _gather_extractions(
        schema,
        model,
        input_files,
        _BatchOptions.resolve(
            instructions,
            style=style,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            rich=False,
            cite=cite,
            cite_min_confidence=cite_min_confidence,
            pages=pages,
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            on_progress=on_progress,
        ),
    )


@overload
def iter_extract_many_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: Literal[False] = False,
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
) -> AsyncIterator[tuple[int, T]]: ...


@overload
def iter_extract_many_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: Literal[True],
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
) -> AsyncIterator[tuple[int, T | Exception]]: ...


@overload
def iter_extract_many_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool,
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
) -> AsyncIterator[tuple[int, T | Exception]]: ...


def iter_extract_many_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
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
) -> AsyncIterator[tuple[int, T | Exception]]:
    """Stream ``(input_index, result)`` pairs in completion order.

    Unlike :func:`extract_many_async`, this API does not eagerly consume
    ``input_files`` and does not wait for the complete batch before yielding.
    At most ``max_concurrency`` items are scheduled at once. If
    ``return_exceptions`` is true, item failures are yielded as the result;
    otherwise the first failure cancels and awaits all outstanding work.
    ``max_input_bytes`` applies the same per-item cap as the list APIs.

    The function itself is synchronous because it returns an async iterator::

        async for index, result in iter_extract_many_async(...):
            ...
    """
    # The shared generator is typed for the richest yield (ExtractionResult);
    # without ``rich`` it only ever yields ``T`` or an exception, so narrow it.
    return cast(
        AsyncIterator[tuple[int, T | Exception]],
        _iter_extractions(
            schema,
            model,
            input_files,
            _BatchOptions.resolve(
                instructions,
                style=style,
                media_type=media_type,
                max_input_bytes=max_input_bytes,
                max_concurrency=max_concurrency,
                return_exceptions=return_exceptions,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                retry_max_backoff=retry_max_backoff,
                rich=False,
                cite=cite,
                cite_min_confidence=cite_min_confidence,
                pages=pages,
                max_pages=max_pages,
                language=language,
                model_settings=model_settings,
                timeout=timeout,
                url_timeout=url_timeout,
                on_progress=on_progress,
            ),
        ),
    )


@overload
def extract_many_with_results(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: Literal[False] = False,
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
) -> list[ExtractionResult[T]]: ...


@overload
def extract_many_with_results(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: Literal[True],
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
) -> list[ExtractionResult[T] | Exception]: ...


@overload
def extract_many_with_results(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool,
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
) -> list[ExtractionResult[T] | Exception]: ...


def extract_many_with_results(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
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
) -> list:
    """Run a batch and return per-item :class:`ExtractionResult` diagnostics.

    Takes exactly the same arguments as :func:`extract_many` — see that
    function for their meaning, the ordering and concurrency guarantees, the
    retry semantics, and the errors raised — but returns richer results
    carrying token usage, attempt counts, timing, model/media metadata, and a
    sanitized source label. Use :func:`total_usage` to aggregate token usage
    across the returned results.

    Returns:
        A list of ``ExtractionResult`` (or ``Exception`` when
        ``return_exceptions=True``) in input order.
    """
    return _run_batch_sync(
        schema,
        model,
        input_files,
        _BatchOptions.resolve(
            instructions,
            style=style,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            rich=True,
            cite=cite,
            cite_min_confidence=cite_min_confidence,
            pages=pages,
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            on_progress=on_progress,
        ),
        name="extract_many_with_results",
    )


@overload
async def extract_many_with_results_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: Literal[False] = False,
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
) -> list[ExtractionResult[T]]: ...


@overload
async def extract_many_with_results_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: Literal[True],
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
) -> list[ExtractionResult[T] | Exception]: ...


@overload
async def extract_many_with_results_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool,
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
) -> list[ExtractionResult[T] | Exception]: ...


async def extract_many_with_results_async(
    schema: type[T],
    model: str | Model,
    input_files: Iterable[ExtractionInputLike],
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
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
) -> list:
    """Async sibling of :func:`extract_many_with_results`."""
    return await _gather_extractions(
        schema,
        model,
        input_files,
        _BatchOptions.resolve(
            instructions,
            style=style,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            rich=True,
            cite=cite,
            cite_min_confidence=cite_min_confidence,
            pages=pages,
            max_pages=max_pages,
            language=language,
            model_settings=model_settings,
            timeout=timeout,
            url_timeout=url_timeout,
            on_progress=on_progress,
        ),
    )
