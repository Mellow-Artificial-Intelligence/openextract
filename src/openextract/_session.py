"""Reusable sync and async extraction sessions."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
from pydantic import BaseModel

from ._agent import (
    _build_agent,
    _build_run_inputs,
    _run_extraction_async,
    _session_model_settings,
    _usage_from_result,
)
from ._citations import prepare_cited_run, split_cited_output
from ._config import (
    _resolve_max_input_bytes,
    _resolve_url_timeout,
    _validate_cite_min_confidence,
    _validate_pages,
)
from ._errors import _extraction_errors
from ._media import _get_media, _get_media_async
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
from ._types import ExtractionInputLike, ExtractProgress, OnProgress, RetryPolicy, T, Usage
from ._windows import emit_progress, extract_windows_async, extract_windows_sync

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models import Model
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.settings import ModelSettings


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

    def _validate_output(self, output: object) -> T:
        with _extraction_errors():
            return self._schema.model_validate(output)

    def _output_from_run(self, result: Any, parsed: ParsedDocument | None = None) -> T:
        output, _citations = split_cited_output(
            result.output,
            self._schema,
            cite=self._cite,
            parsed=parsed,
            cite_min_confidence=self._cite_min_confidence,
        )
        return self._validate_output(result.output if not self._cite else output)

    def _output_and_usage(
        self, result: Any, parsed: ParsedDocument | None = None
    ) -> tuple[T, Usage]:
        return self._output_from_run(result, parsed), _usage_from_result(result)

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
        self, file_bytes: bytes, file_type: str, pages: Sequence[int] | None = None
    ) -> Iterator[tuple[PydanticAgent, list, ParsedDocument | None]]:
        """Pair the session agent with per-call run inputs for one extraction."""
        assert self._agent is not None
        parsed_inputs, parsed = maybe_parsed_inputs(
            file_bytes,
            file_type,
            parse=should_parse(self._cite, self._style, pages),
            pages=pages,
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
        with self._session_agent_inputs(file_bytes, file_type, pages) as prepared:
            yield prepared

    def _extract_projected[R](
        self,
        input_file: ExtractionInputLike,
        media_type: str | None,
        project: Callable[..., R],
        *,
        with_usage: bool = False,
        on_progress: OnProgress | None = None,
        pages: Sequence[int] | None = None,
    ) -> R:
        """Run one retrying extraction and map the raw result through ``project``."""
        callback = self._on_progress if on_progress is None else on_progress
        selected = self._pages if pages is None else _validate_pages(pages)
        with self._prepare_session_extraction(input_file, media_type, selected) as (
            agent,
            inputs,
            parsed,
        ):
            windows = parsed_window_inputs(parsed, inputs)
            if len(windows) == 1:
                emit_progress(callback, 1, 1, parsed)
                return _run_with_retries_sync(
                    lambda: project(self._run_agent(agent, windows[0]), parsed),
                    max_retries=self._retry_policy.max_retries,
                    retry_backoff=self._retry_policy.backoff,
                    retry_max_backoff=self._retry_policy.max_backoff,
                )

            def _run(window: list) -> tuple[object, Usage]:
                result = self._run_agent(agent, window)
                return result.output, _usage_from_result(result)

            output, usage, _citations = extract_windows_sync(
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
            validated = self._validate_output(output)
            if with_usage:
                return cast(R, (validated, usage))
            return cast(R, validated)

    def extract(
        self,
        input_file: ExtractionInputLike,
        *,
        media_type: str | None = None,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
    ) -> T:
        """Extract one input using the session's reusable agent and clients."""
        return self._extract_projected(
            input_file, media_type, self._output_from_run, on_progress=on_progress, pages=pages
        )

    def extract_with_usage(
        self,
        input_file: ExtractionInputLike,
        *,
        media_type: str | None = None,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
    ) -> tuple[T, Usage]:
        """Extract one input and return its successful-call token usage."""
        return self._extract_projected(
            input_file,
            media_type,
            self._output_and_usage,
            with_usage=True,
            on_progress=on_progress,
            pages=pages,
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
        with self._session_agent_inputs(file_bytes, file_type, pages) as prepared:
            yield prepared

    async def _extract_projected[R](
        self,
        input_file: ExtractionInputLike,
        media_type: str | None,
        project: Callable[..., R],
        *,
        with_usage: bool = False,
        on_progress: OnProgress | None = None,
        pages: Sequence[int] | None = None,
    ) -> R:
        """Async counterpart to :meth:`Extractor._extract_projected`."""
        callback = self._on_progress if on_progress is None else on_progress
        selected = self._pages if pages is None else _validate_pages(pages)
        async with self._prepare_session_extraction(input_file, media_type, selected) as (
            agent,
            inputs,
            parsed,
        ):
            windows = parsed_window_inputs(parsed, inputs)
            if len(windows) == 1:
                emit_progress(callback, 1, 1, parsed)

                async def _once() -> R:
                    return project(await _run_extraction_async(agent, windows[0]), parsed)

                return await _run_with_retries_async(
                    _once,
                    max_retries=self._retry_policy.max_retries,
                    retry_backoff=self._retry_policy.backoff,
                    retry_max_backoff=self._retry_policy.max_backoff,
                )

            async def _run(window: list) -> tuple[object, Usage]:
                result = await _run_extraction_async(agent, window)
                return result.output, _usage_from_result(result)

            output, usage, _citations = await extract_windows_async(
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
            validated = self._validate_output(output)
            if with_usage:
                return cast(R, (validated, usage))
            return cast(R, validated)

    async def extract(
        self,
        input_file: ExtractionInputLike,
        *,
        media_type: str | None = None,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
    ) -> T:
        """Extract one input using the session's reusable agent and clients."""
        return await self._extract_projected(
            input_file, media_type, self._output_from_run, on_progress=on_progress, pages=pages
        )

    async def extract_with_usage(
        self,
        input_file: ExtractionInputLike,
        *,
        media_type: str | None = None,
        on_progress: Callable[[ExtractProgress], None] | None = None,
        pages: Sequence[int] | None = None,
    ) -> tuple[T, Usage]:
        """Extract one input and return its successful-call token usage."""
        return await self._extract_projected(
            input_file,
            media_type,
            self._output_and_usage,
            with_usage=True,
            on_progress=on_progress,
            pages=pages,
        )
