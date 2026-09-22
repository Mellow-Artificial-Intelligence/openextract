"""Reusable sync and async extraction sessions."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import httpx
from pydantic import BaseModel

from ._agent import (
    _build_agent,
    _build_run_inputs,
    _model_identifier,
    _run_extraction_async,
    _session_model_settings,
    _usage_from_result,
)
from ._batch import _cancel_tasks
from ._citations import prepare_cited_run, split_cited_output
from ._config import (
    _apply_max_pages,
    _resolve_max_input_bytes,
    _resolve_url_timeout,
    _validate_cite_min_confidence,
    _validate_max_concurrency,
    _validate_max_pages,
    _validate_pages,
)
from ._errors import _extraction_errors
from ._media import _get_media, _get_media_async, _item_source_label
from ._parse import ParsedDocument, maybe_parsed_inputs, parsed_window_inputs
from ._retry import _run_with_retries_async, _run_with_retries_sync
from ._styles import (
    ExtractionStyle,
    compose_extract_instructions,
    materialize_text_document,
    normalize_language,
    normalize_style,
    should_parse,
    style_capabilities,
    style_run_inputs,
    uses_workspace,
)
from ._types import (
    Citation,
    ExtractionInputLike,
    ExtractionResult,
    ExtractProgress,
    OnProgress,
    RetryPolicy,
    T,
    Usage,
    _extraction_result,
    _resolve_item,
)
from ._warnings import extraction_warnings
from ._windows import emit_progress, extract_windows_async, extract_windows_sync

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models import Model
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.settings import ModelSettings


def _call_provenance(
    input_file: ExtractionInputLike, media_type: str | None
) -> tuple[str | None, str | None]:
    """Requested media type and sanitized source label for a session result."""
    source, item_media_type, name = _resolve_item(input_file, media_type)
    return item_media_type, _item_source_label(source, name)


async def _run_session_batch[R](
    input_files: Iterable[ExtractionInputLike],
    run_item: Callable[[ExtractionInputLike], Awaitable[R]],
    *,
    max_concurrency: int,
    return_exceptions: bool,
) -> list:
    """Run session extractions with bounded concurrency, restoring input order.

    Reuses the caller's session agent via ``run_item``. ``max_concurrency``
    limits in-flight items; ``return_exceptions`` matches oneshot
    ``extract_many`` (in-place errors vs fail-fast cancel).
    """
    _validate_max_concurrency(max_concurrency)
    files = list(input_files)
    if not files:
        return []
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one(item: ExtractionInputLike) -> R:
        async with semaphore:
            return await run_item(item)

    tasks = [asyncio.create_task(_one(item)) for item in files]
    try:
        return list(await asyncio.gather(*tasks, return_exceptions=return_exceptions))
    except BaseException:
        await _cancel_tasks(tasks)
        raise


class _ExtractorSession[T: BaseModel]:
    """Configuration, lifecycle bookkeeping, and output validation for sessions.

    Both :class:`Extractor` and :class:`AsyncExtractor` are constructed through
    this class, so the public constructor signature is declared exactly once.
    Subclasses contribute only the transport they own (a sync or async HTTP
    client, plus the loop or runner the agent runs on).
    """

    def __init__(
        self,
        schema: type[T],
        model: str | Model | None = None,
        instructions: str | None = None,
        *,
        style: ExtractionStyle | str = "direct",
        agent: PydanticAgent | None = None,
        model_settings: ModelSettings | None = None,
        timeout: float | None = None,
        instrument: bool | InstrumentationSettings = False,
        retry_policy: RetryPolicy | None = None,
        max_input_bytes: int | None = None,
        url_timeout: float | None = None,
        cite: bool = False,
        cite_min_confidence: float | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
        language: str | None = None,
        on_progress: OnProgress | None = None,
    ) -> None:
        resolved_style = normalize_style(style)
        language = normalize_language(language)
        if agent is not None:
            if model is not None:
                raise ValueError("model and agent are mutually exclusive; provide exactly one.")
            if (
                instructions is not None
                or language is not None
                or model_settings is not None
                or timeout is not None
            ):
                raise ValueError(
                    "instructions, language, model_settings, and timeout must be "
                    "configured on an injected agent."
                )
            if instrument is not False:
                raise ValueError("instrument must be configured on an injected agent.")
            if resolved_style is not ExtractionStyle.DIRECT:
                raise ValueError("style other than 'direct' cannot be used with an injected agent.")
            if cite:
                raise ValueError("cite cannot be used with an injected agent.")
            configured_agent = agent
            session_settings = None
            run_schema, run_instructions = schema, instructions
        else:
            if model is None:
                raise TypeError("model is required unless agent is provided.")
            session_settings = _session_model_settings(model_settings, timeout)
            run_schema, run_instructions = prepare_cited_run(
                schema, compose_extract_instructions(instructions, resolved_style, language), cite
            )
            if not uses_workspace(resolved_style):
                configured_agent = _build_agent(
                    run_schema,
                    model,
                    run_instructions,
                    model_settings=session_settings,
                    instrument=instrument,
                )
            else:
                configured_agent = None

        if retry_policy is None:
            retry_policy = RetryPolicy()
        elif not isinstance(retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be a RetryPolicy instance.")

        self._schema = schema
        self._cite = cite
        self._cite_min_confidence = _validate_cite_min_confidence(cite_min_confidence)
        self._pages = _validate_pages(pages)
        self._max_pages = _validate_max_pages(max_pages)
        _apply_max_pages(self._pages, self._max_pages)
        self._run_schema = run_schema
        self._run_instructions = run_instructions
        self._model = model
        self._instructions = instructions
        self._style = resolved_style
        self._model_settings = session_settings
        self._instrument = instrument
        self._agent = configured_agent
        self._retry_policy = retry_policy
        self._max_input_bytes = _resolve_max_input_bytes(max_input_bytes)
        self._url_timeout = _resolve_url_timeout(url_timeout)
        self._on_progress = on_progress
        self._entered = False
        self._closed = False
        self._style_workspace: tempfile.TemporaryDirectory[str] | None = None
        self._style_run_index = 0

    def _page_filter(
        self, pages: Sequence[int] | None, max_pages: int | None
    ) -> tuple[tuple[int, ...] | None, int | None]:
        """Resolve constructor defaults vs per-call ``pages`` / ``max_pages``."""
        selected = self._pages if pages is None else _validate_pages(pages)
        cap = self._max_pages if max_pages is None else _validate_max_pages(max_pages)
        return _apply_max_pages(selected, cap), cap

    def _validate_output(self, output: object) -> T:
        with _extraction_errors():
            return self._schema.model_validate(output)

    def _split_run(
        self, result: Any, parsed: ParsedDocument | None = None
    ) -> tuple[T, tuple[Citation, ...], tuple[str, ...]]:
        output, citations, warnings = split_cited_output(
            result.output,
            self._schema,
            cite=self._cite,
            parsed=parsed,
            cite_min_confidence=self._cite_min_confidence,
        )
        return (
            self._validate_output(result.output if not self._cite else output),
            citations,
            warnings,
        )

    def _output_from_run(self, result: Any, parsed: ParsedDocument | None = None) -> T:
        output, _citations, _warnings = self._split_run(result, parsed)
        return output

    def _output_and_usage(
        self, result: Any, parsed: ParsedDocument | None = None
    ) -> tuple[T, Usage]:
        return self._output_from_run(result, parsed), _usage_from_result(result)

    def _result_from_run(
        self,
        result: Any,
        parsed: ParsedDocument | None = None,
        *,
        attempts: int,
        started: float,
        media_type: str | None,
        source: str | None,
        agent: object,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> ExtractionResult[T]:
        output, citations, cite_warnings = self._split_run(result, parsed)
        return self._session_result(
            output,
            _usage_from_result(result),
            citations,
            attempts=attempts,
            started=started,
            media_type=media_type,
            source=source,
            agent=agent,
            warnings=extraction_warnings(parsed, pages, max_pages, cite_warnings),
        )

    def _session_result(
        self,
        output: T,
        usage: Usage,
        citations: tuple[Citation, ...],
        *,
        attempts: int,
        started: float,
        media_type: str | None,
        source: str | None,
        agent: object,
        warnings: tuple[str, ...] = (),
    ) -> ExtractionResult[T]:
        return _extraction_result(
            output,
            usage,
            attempts=attempts,
            started=started,
            model=_model_identifier(self._model, agent),
            media_type=media_type,
            source=source,
            citations=citations,
            warnings=warnings,
        )

    def _finish_projected(
        self,
        output: T,
        usage: Usage,
        citations: tuple[Citation, ...],
        *,
        with_usage: bool,
        with_result: bool,
        attempts: int,
        started: float,
        media_type: str | None,
        source: str | None,
        agent: object,
        warnings: tuple[str, ...] = (),
    ) -> T | tuple[T, Usage] | ExtractionResult[T]:
        if with_result:
            return self._session_result(
                output,
                usage,
                citations,
                attempts=attempts,
                started=started,
                media_type=media_type,
                source=source,
                agent=agent,
                warnings=warnings,
            )
        if with_usage:
            return output, usage
        return output

    async def _project_run[R](
        self,
        agent: PydanticAgent,
        inputs: list,
        parsed: ParsedDocument | None,
        project: Callable[..., R],
        *,
        with_usage: bool,
        with_result: bool,
        callback: OnProgress | None,
        item_media_type: str | None,
        source_label: str | None,
        started: float,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> R:
        """Retrying async extraction after media and run inputs are prepared."""
        windows = parsed_window_inputs(parsed, inputs)
        attempts = 0
        if len(windows) == 1:
            emit_progress(callback, 1, 1, parsed)

            async def _once() -> R:
                nonlocal attempts
                if with_result:
                    attempts += 1
                result = await _run_extraction_async(agent, windows[0])
                if with_result:
                    return cast(
                        R,
                        self._result_from_run(
                            result,
                            parsed,
                            attempts=attempts,
                            started=started,
                            media_type=item_media_type,
                            source=source_label,
                            agent=agent,
                            pages=pages,
                            max_pages=max_pages,
                        ),
                    )
                return project(result, parsed)

            return await _run_with_retries_async(
                _once,
                max_retries=self._retry_policy.max_retries,
                retry_backoff=self._retry_policy.backoff,
                retry_max_backoff=self._retry_policy.max_backoff,
            )

        async def _run(window: list) -> tuple[object, Usage]:
            nonlocal attempts
            if with_result:
                attempts += 1
            result = await _run_extraction_async(agent, window)
            return result.output, _usage_from_result(result)

        output, usage, citations, cite_warnings = await extract_windows_async(
            _run,
            inputs,
            parsed,
            self._schema,
            self._cite,
            cite_min_confidence=self._cite_min_confidence,
            max_retries=self._retry_policy.max_retries,
            retry_backoff=self._retry_policy.backoff,
            retry_max_backoff=self._retry_policy.max_backoff,
            on_progress=callback,
        )
        return cast(
            R,
            self._finish_projected(
                self._validate_output(cast(Any, output)),
                usage,
                citations,
                with_usage=with_usage,
                with_result=with_result,
                attempts=attempts,
                started=started,
                media_type=item_media_type,
                source=source_label,
                agent=agent,
                warnings=extraction_warnings(parsed, pages, max_pages, cite_warnings),
            ),
        )

    # -- lifecycle bookkeeping shared by the sync and async sessions ---------

    def _ensure_enterable(self) -> None:
        class_name = type(self).__name__
        if self._closed:
            raise RuntimeError(f"{class_name} is closed and cannot be reused.")
        if self._entered:
            raise RuntimeError(f"{class_name} is already entered.")

    def _ensure_open(self) -> None:
        if not self._entered:
            raise RuntimeError(
                f"{type(self).__name__} must be used as a context manager before extraction."
            )

    def _close_before_enter(self) -> bool:
        """Mark a session that was never entered as closed; return whether it was."""
        if self._entered:
            return False
        self._closed = True
        return True

    def _finalize_close(self) -> None:
        """Drop per-session state after the transport has been torn down."""
        self._discard_style_state()
        self._entered = False
        self._closed = True

    def _enter_style_workspace(self) -> None:
        """Create the session workspace and its agent for search/code styles.

        The agent (and its provider HTTP client) is built once per session and
        lives until the session closes, matching the direct-style lifecycle.
        """
        if not uses_workspace(self._style):
            return
        assert self._model is not None
        self._style_workspace = tempfile.TemporaryDirectory(prefix="openextract-")
        with _extraction_errors():
            self._agent = _build_agent(
                self._run_schema,
                self._model,
                self._run_instructions,
                model_settings=self._model_settings,
                instrument=self._instrument,
                extra_capabilities=style_capabilities(
                    self._style, Path(self._style_workspace.name)
                ),
            )

    def _discard_style_state(self) -> None:
        """Drop the style agent and remove the workspace owned by the session."""
        if self._style_workspace is not None:
            self._style_workspace.cleanup()
            self._style_workspace = None
            self._agent = None

    @contextmanager
    def _style_document(self, file_bytes: bytes, file_type: str) -> Iterator[list]:
        """Materialize one document in the session workspace for a single call.

        Each call gets its own subdirectory so concurrent async extractions
        never collide. The subdirectory is removed when the call finishes; the
        workspace itself lives until the session closes, so documents persist
        across retries within a call.
        """
        assert self._style_workspace is not None
        self._style_run_index += 1
        run_dir = Path(self._style_workspace.name) / f"run{self._style_run_index}"
        run_dir.mkdir()
        try:
            filename = materialize_text_document(run_dir, file_bytes, file_type, style=self._style)
            yield style_run_inputs(self._style, f"{run_dir.name}/{filename}")
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    @contextmanager
    def _session_agent_inputs(
        self,
        file_bytes: bytes,
        file_type: str,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> Iterator[tuple[PydanticAgent, list, ParsedDocument | None]]:
        """Pair the session agent with per-call run inputs for one extraction."""
        assert self._agent is not None
        parsed_inputs, parsed = maybe_parsed_inputs(
            file_bytes,
            file_type,
            parse=should_parse(self._cite, self._style, pages, max_pages),
            pages=pages,
            max_pages=max_pages,
        )
        if not uses_workspace(self._style):
            inputs = (
                parsed_inputs
                if parsed_inputs is not None
                else _build_run_inputs(file_bytes, file_type)
            )
            yield self._agent, inputs, parsed
            return
        with self._style_document(file_bytes, file_type) as inputs:
            yield self._agent, inputs, parsed


class Extractor(_ExtractorSession[T]):
    """Reusable synchronous extraction session.

    An ``Extractor`` is bound to the thread that enters it and is not
    thread-safe. Use one session per thread and close it deterministically with
    a ``with`` block. ``search``/``code`` sessions build one harness agent and
    temporary workspace on enter and remove both on close.
    """

    _client: httpx.Client | None = None
    _runner: asyncio.Runner | None = None
    _thread_id: int | None = None

    def __enter__(self) -> Extractor[T]:
        self._ensure_enterable()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Extractor cannot be entered from a running event loop; use AsyncExtractor instead."
            )

        client = httpx.Client(follow_redirects=False, timeout=self._url_timeout)
        # Keep the session loop private so entering an Extractor does not replace
        # or clear the caller's process-wide current event loop.
        runner = asyncio.Runner(loop_factory=asyncio.new_event_loop)
        runner.__enter__()
        try:
            self._enter_style_workspace()
            assert self._agent is not None
            runner.run(self._agent.__aenter__())
        except BaseException:
            self._discard_style_state()
            client.close()
            runner.close()
            raise

        self._client = client
        self._runner = runner
        self._thread_id = threading.get_ident()
        self._entered = True
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return self._close(exc_info)

    def _close(self, exc_info: tuple[object, ...]) -> bool:
        if self._close_before_enter():
            return False
        if self._thread_id != threading.get_ident():
            raise RuntimeError("Extractor can only be closed from the thread that entered it.")
        assert self._runner is not None
        assert self._client is not None
        assert self._agent is not None
        suppressed = False
        try:
            suppressed = bool(self._runner.run(self._agent.__aexit__(*exc_info)))
        finally:
            self._client.close()
            self._runner.close()
            self._finalize_close()
            self._client = None
            self._runner = None
        return suppressed

    def close(self) -> None:
        """Close the owned agent/provider and input HTTP client."""
        self._close((None, None, None))

    def _ensure_sync_open(self) -> httpx.Client:
        self._ensure_open()
        if self._thread_id != threading.get_ident():
            raise RuntimeError("Extractor can only be used from the thread that entered it.")
        assert self._client is not None
        return self._client

    def _run_agent(self, agent: PydanticAgent, inputs: list):
        assert self._runner is not None
        return self._runner.run(_run_extraction_async(agent, inputs))

    @contextmanager
    def _prepare_session_extraction(
        self,
        input_file: ExtractionInputLike,
        media_type: str | None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> Iterator[tuple[PydanticAgent, list, ParsedDocument | None]]:
        """Resolve media and yield ``(agent, inputs, parsed)`` for one session call."""
        client = self._ensure_sync_open()
        with _extraction_errors():
            file_bytes, file_type = _get_media(
                input_file,
                media_type=media_type,
                max_input_bytes=self._max_input_bytes,
                client=client,
            )
        with self._session_agent_inputs(file_bytes, file_type, pages, max_pages) as prepared:
            yield prepared

    def _extract_projected[R](
        self,
        input_file: ExtractionInputLike,
        media_type: str | None,
        project: Callable[..., R],
        *,
        with_usage: bool = False,
        with_result: bool = False,
        on_progress: OnProgress | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> R:
        """Run one retrying extraction and map the raw result through ``project``."""
        callback = self._on_progress if on_progress is None else on_progress
        selected, cap = self._page_filter(pages, max_pages)
        started = time.perf_counter() if with_result else 0.0
        attempts = 0
        item_media_type, source_label = (
            _call_provenance(input_file, media_type) if with_result else (media_type, None)
        )
        with self._prepare_session_extraction(input_file, media_type, selected, cap) as (
            agent,
            inputs,
            parsed,
        ):
            windows = parsed_window_inputs(parsed, inputs)
            if len(windows) == 1:
                emit_progress(callback, 1, 1, parsed)

                def _once() -> R:
                    nonlocal attempts
                    if with_result:
                        attempts += 1
                    result = self._run_agent(agent, windows[0])
                    if with_result:
                        return cast(
                            R,
                            self._result_from_run(
                                result,
                                parsed,
                                attempts=attempts,
                                started=started,
                                media_type=item_media_type,
                                source=source_label,
                                agent=agent,
                                pages=selected,
                                max_pages=cap,
                            ),
                        )
                    return project(result, parsed)

                return _run_with_retries_sync(
                    _once,
                    max_retries=self._retry_policy.max_retries,
                    retry_backoff=self._retry_policy.backoff,
                    retry_max_backoff=self._retry_policy.max_backoff,
                )

            def _run(window: list) -> tuple[object, Usage]:
                nonlocal attempts
                if with_result:
                    attempts += 1
                result = self._run_agent(agent, window)
                return result.output, _usage_from_result(result)

            output, usage, citations, cite_warnings = extract_windows_sync(
                _run,
                inputs,
                parsed,
                self._schema,
                self._cite,
                cite_min_confidence=self._cite_min_confidence,
                max_retries=self._retry_policy.max_retries,
                retry_backoff=self._retry_policy.backoff,
                retry_max_backoff=self._retry_policy.max_backoff,
                on_progress=callback,
            )
            return cast(
                R,
                self._finish_projected(
                    self._validate_output(output),
                    usage,
                    citations,
                    with_usage=with_usage,
                    with_result=with_result,
                    attempts=attempts,
                    started=started,
                    media_type=item_media_type,
                    source=source_label,
                    agent=agent,
                    warnings=extraction_warnings(parsed, selected, cap, cite_warnings),
                ),
            )

    async def _extract_projected_async[R](
        self,
        input_file: ExtractionInputLike,
        media_type: str | None,
        project: Callable[..., R],
        *,
        with_usage: bool = False,
        with_result: bool = False,
        on_progress: OnProgress | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> R:
        """Async session extract used by :meth:`extract_many` on the session loop."""
        callback = self._on_progress if on_progress is None else on_progress
        selected, cap = self._page_filter(pages, max_pages)
        started = time.perf_counter() if with_result else 0.0
        item_media_type, source_label = (
            _call_provenance(input_file, media_type) if with_result else (media_type, None)
        )
        with self._prepare_session_extraction(input_file, media_type, selected, cap) as (
            agent,
            inputs,
            parsed,
        ):
            return await self._project_run(
                agent,
                inputs,
                parsed,
                project,
                with_usage=with_usage,
                with_result=with_result,
                callback=callback,
                item_media_type=item_media_type,
                source_label=source_label,
                started=started,
                pages=selected,
                max_pages=cap,
            )

    def _extract_many_projected[R](
        self,
        input_files: Iterable[ExtractionInputLike],
        project: Callable[..., R],
        *,
        media_type: str | None,
        max_concurrency: int,
        return_exceptions: bool,
        with_result: bool = False,
        on_progress: OnProgress | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list:
        """Bounded concurrent extracts on the session's private event loop."""
        _validate_max_concurrency(max_concurrency)
        self._ensure_sync_open()
        assert self._runner is not None

        async def _one(item: ExtractionInputLike) -> R:
            return await self._extract_projected_async(
                item,
                media_type,
                project,
                with_result=with_result,
                on_progress=on_progress,
                pages=pages,
                max_pages=max_pages,
            )

        return self._runner.run(
            _run_session_batch(
                input_files,
                _one,
                max_concurrency=max_concurrency,
                return_exceptions=return_exceptions,
            )
        )

    def extract(
        self,
        input_file: ExtractionInputLike,
        *,
        media_type: str | None = None,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> T:
        """Extract one input using the session's reusable agent and clients."""
        return self._extract_projected(
            input_file,
            media_type,
            self._output_from_run,
            on_progress=on_progress,
            pages=pages,
            max_pages=max_pages,
        )

    def extract_with_usage(
        self,
        input_file: ExtractionInputLike,
        *,
        media_type: str | None = None,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> tuple[T, Usage]:
        """Extract one input and return its successful-call token usage."""
        return self._extract_projected(
            input_file,
            media_type,
            self._output_and_usage,
            with_usage=True,
            on_progress=on_progress,
            pages=pages,
            max_pages=max_pages,
        )

    def extract_with_result(
        self,
        input_file: ExtractionInputLike,
        *,
        media_type: str | None = None,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> ExtractionResult[T]:
        """Extract one input and return an :class:`ExtractionResult`.

        Citations follow the session ``cite`` / ``cite_min_confidence`` settings.
        """
        return cast(
            ExtractionResult[T],
            self._extract_projected(
                input_file,
                media_type,
                self._output_from_run,
                with_result=True,
                on_progress=on_progress,
                pages=pages,
                max_pages=max_pages,
            ),
        )

    @overload
    def extract_many(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: Literal[False] = False,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[T]: ...

    @overload
    def extract_many(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: Literal[True],
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[T | Exception]: ...

    @overload
    def extract_many(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: bool,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[T | Exception]: ...

    def extract_many(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: bool = False,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list:
        """Extract many inputs using this session's reusable agent and clients.

        Mirrors oneshot :func:`extract_many` for ``input_files``,
        ``max_concurrency``, ``return_exceptions``, and ``on_progress``.
        ``pages`` / ``max_pages`` are per-call overrides, same as
        :meth:`extract`. Session ``cite``, style, language, retry policy,
        model settings, and URL timeout apply to every item.
        """
        return self._extract_many_projected(
            input_files,
            self._output_from_run,
            media_type=media_type,
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
            on_progress=on_progress,
            pages=pages,
            max_pages=max_pages,
        )

    @overload
    def extract_many_with_results(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: Literal[False] = False,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[ExtractionResult[T]]: ...

    @overload
    def extract_many_with_results(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: Literal[True],
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[ExtractionResult[T] | Exception]: ...

    @overload
    def extract_many_with_results(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: bool,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[ExtractionResult[T] | Exception]: ...

    def extract_many_with_results(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: bool = False,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list:
        """Batch counterpart of :meth:`extract_with_result`.

        Same arguments and ordering/concurrency contract as
        :meth:`extract_many`. Each success is an :class:`ExtractionResult`.
        """
        return self._extract_many_projected(
            input_files,
            self._output_from_run,
            media_type=media_type,
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
            with_result=True,
            on_progress=on_progress,
            pages=pages,
            max_pages=max_pages,
        )


class AsyncExtractor(_ExtractorSession[T]):
    """Reusable async extraction session bound to one event loop.

    ``search``/``code`` sessions build one harness agent and temporary
    workspace on enter and remove both on close; concurrent calls each write
    their document into a private workspace subdirectory.
    """

    _client: httpx.AsyncClient | None = None
    _loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> AsyncExtractor[T]:
        self._ensure_enterable()
        client = httpx.AsyncClient(follow_redirects=False, timeout=self._url_timeout)
        try:
            self._enter_style_workspace()
            assert self._agent is not None
            await self._agent.__aenter__()
        except BaseException:
            self._discard_style_state()
            await client.aclose()
            raise
        self._client = client
        self._loop = asyncio.get_running_loop()
        self._entered = True
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return await self._close(exc_info)

    async def _close(self, exc_info: tuple[object, ...]) -> bool:
        if self._close_before_enter():
            return False
        if self._loop is not asyncio.get_running_loop():
            raise RuntimeError(
                "AsyncExtractor can only be closed from the event loop that entered it."
            )
        assert self._client is not None
        assert self._agent is not None
        suppressed = False
        try:
            suppressed = bool(await self._agent.__aexit__(*exc_info))
        finally:
            await self._client.aclose()
            self._finalize_close()
            self._client = None
            self._loop = None
        return suppressed

    async def aclose(self) -> None:
        """Close the owned agent/provider and input HTTP client."""
        await self._close((None, None, None))

    def _ensure_async_open(self) -> httpx.AsyncClient:
        self._ensure_open()
        if self._loop is not asyncio.get_running_loop():
            raise RuntimeError(
                "AsyncExtractor can only be used from the event loop that entered it."
            )
        assert self._client is not None
        return self._client

    @asynccontextmanager
    async def _prepare_session_extraction(
        self,
        input_file: ExtractionInputLike,
        media_type: str | None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[tuple[PydanticAgent, list, ParsedDocument | None]]:
        """Resolve media and yield ``(agent, inputs, parsed)`` for one session call."""
        client = self._ensure_async_open()
        with _extraction_errors():
            file_bytes, file_type = await _get_media_async(
                input_file,
                client,
                media_type=media_type,
                max_input_bytes=self._max_input_bytes,
            )
        with self._session_agent_inputs(file_bytes, file_type, pages, max_pages) as prepared:
            yield prepared

    async def _extract_projected[R](
        self,
        input_file: ExtractionInputLike,
        media_type: str | None,
        project: Callable[..., R],
        *,
        with_usage: bool = False,
        with_result: bool = False,
        on_progress: OnProgress | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> R:
        """Async counterpart to :meth:`Extractor._extract_projected`."""
        callback = self._on_progress if on_progress is None else on_progress
        selected, cap = self._page_filter(pages, max_pages)
        started = time.perf_counter() if with_result else 0.0
        item_media_type, source_label = (
            _call_provenance(input_file, media_type) if with_result else (media_type, None)
        )
        async with self._prepare_session_extraction(input_file, media_type, selected, cap) as (
            agent,
            inputs,
            parsed,
        ):
            return await self._project_run(
                agent,
                inputs,
                parsed,
                project,
                with_usage=with_usage,
                with_result=with_result,
                callback=callback,
                item_media_type=item_media_type,
                source_label=source_label,
                started=started,
                pages=selected,
                max_pages=cap,
            )

    async def _extract_many_projected[R](
        self,
        input_files: Iterable[ExtractionInputLike],
        project: Callable[..., R],
        *,
        media_type: str | None,
        max_concurrency: int,
        return_exceptions: bool,
        with_result: bool = False,
        on_progress: OnProgress | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list:
        """Bounded concurrent extracts on the session event loop."""
        _validate_max_concurrency(max_concurrency)
        self._ensure_async_open()

        async def _one(item: ExtractionInputLike) -> R:
            return await self._extract_projected(
                item,
                media_type,
                project,
                with_result=with_result,
                on_progress=on_progress,
                pages=pages,
                max_pages=max_pages,
            )

        return await _run_session_batch(
            input_files,
            _one,
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
        )

    async def extract(
        self,
        input_file: ExtractionInputLike,
        *,
        media_type: str | None = None,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> T:
        """Extract one input using the session's reusable agent and clients."""
        return await self._extract_projected(
            input_file,
            media_type,
            self._output_from_run,
            on_progress=on_progress,
            pages=pages,
            max_pages=max_pages,
        )

    async def extract_with_usage(
        self,
        input_file: ExtractionInputLike,
        *,
        media_type: str | None = None,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> tuple[T, Usage]:
        """Extract one input and return its successful-call token usage."""
        return await self._extract_projected(
            input_file,
            media_type,
            self._output_and_usage,
            with_usage=True,
            on_progress=on_progress,
            pages=pages,
            max_pages=max_pages,
        )

    async def extract_with_result(
        self,
        input_file: ExtractionInputLike,
        *,
        media_type: str | None = None,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> ExtractionResult[T]:
        """Extract one input and return an :class:`ExtractionResult`.

        Citations follow the session ``cite`` / ``cite_min_confidence`` settings.
        """
        return cast(
            ExtractionResult[T],
            await self._extract_projected(
                input_file,
                media_type,
                self._output_from_run,
                with_result=True,
                on_progress=on_progress,
                pages=pages,
                max_pages=max_pages,
            ),
        )

    @overload
    async def extract_many(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: Literal[False] = False,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[T]: ...

    @overload
    async def extract_many(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: Literal[True],
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[T | Exception]: ...

    @overload
    async def extract_many(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: bool,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[T | Exception]: ...

    async def extract_many(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: bool = False,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list:
        """Extract many inputs using this session's reusable agent and clients.

        Same contract as :meth:`Extractor.extract_many`. Naming matches the
        other async session methods (``extract``, not ``extract_async``).
        """
        return await self._extract_many_projected(
            input_files,
            self._output_from_run,
            media_type=media_type,
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
            on_progress=on_progress,
            pages=pages,
            max_pages=max_pages,
        )

    @overload
    async def extract_many_with_results(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: Literal[False] = False,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[ExtractionResult[T]]: ...

    @overload
    async def extract_many_with_results(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: Literal[True],
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[ExtractionResult[T] | Exception]: ...

    @overload
    async def extract_many_with_results(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: bool,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list[ExtractionResult[T] | Exception]: ...

    async def extract_many_with_results(
        self,
        input_files: Iterable[ExtractionInputLike],
        *,
        media_type: str | None = None,
        max_concurrency: int = 5,
        return_exceptions: bool = False,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
        max_pages: int | None = None,
    ) -> list:
        """Batch counterpart of :meth:`extract_with_result`.

        Same arguments and ordering/concurrency contract as
        :meth:`extract_many`. Each success is an :class:`ExtractionResult`.
        """
        return await self._extract_many_projected(
            input_files,
            self._output_from_run,
            media_type=media_type,
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
            with_result=True,
            on_progress=on_progress,
            pages=pages,
            max_pages=max_pages,
        )
