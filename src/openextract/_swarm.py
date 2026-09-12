"""Run several agents over one input and reduce their outputs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import httpx

from ._agent import (
    _build_agent,
    _model_identifier,
    _resolve_run_inputs,
    _run_extraction_async,
    _usage_from_result,
)
from ._agents import AgentInput, DefinedAgent, RemoteAgent, SwarmMember, flatten_agent
from ._batch import _require_no_running_loop
from ._citations import prepare_cited_run, split_cited_output
from ._config import (
    _DEFAULT_RETRY_MAX_BACKOFF,
    _resolve_max_input_bytes,
    _url_fetch_timeout,
    _validate_max_concurrency,
    _validate_retry_options,
    _validate_swarm_size,
)
from ._errors import _extraction_errors
from ._media import _get_media_async, _item_source_label
from ._parse import maybe_parsed_inputs
from ._reduce import SwarmReduce, normalize_reduce, reduce_citations, reduce_outputs
from ._remote import run_remote_extraction
from ._styles import ExtractionStyle, normalize_style, prepared_style_run
from ._types import (
    Citation,
    ExtractionInputLike,
    ExtractionResult,
    T,
    Usage,
    _resolve_item,
    total_usage,
)
from ._windows import extract_windows_async

if TYPE_CHECKING:
    pass

_DEFAULT_SWARM_CONCURRENCY = 5


# One agent or a list of them, as accepted by the ``agents`` argument.
type SwarmAgents = AgentInput | Sequence[AgentInput]


@dataclass(frozen=True)
class SwarmResult[T]:
    """The reduced output of a swarm plus every agent's individual result.

    Attributes:
        output: The reduced schema instance.
        agents: Per-agent :class:`ExtractionResult` or the exception that agent
            raised, in agent order.
        usage: Token usage summed across the agents that succeeded.
        reduce: The strategy that produced ``output``.
        citations: Per-field source spans for ``output`` when ``cite=True``;
            empty otherwise. Each successful agent's own citations remain on
            that :class:`ExtractionResult`. ``first`` keeps the leading agent's
            citations; ``merge`` and ``vote`` keep the first citation whose
            agent's field value equals ``output``.
    """

    output: T
    agents: tuple[ExtractionResult[T] | Exception, ...]
    usage: Usage
    reduce: SwarmReduce
    citations: tuple[Citation, ...] = ()


def resolve_swarm_members(
    agents: SwarmAgents,
    size: int | None = None,
) -> list[SwarmMember]:
    """Expand the ``agents`` argument into one :class:`SwarmMember` per agent.

    A single agent plus ``size`` fans that agent out ``size`` times. A list of
    agents is used as-is, and ``size`` may not contradict its length. Defined
    agents are flattened first, so a parent with subagents contributes one
    member per leaf.

    Raises:
        ValueError: If ``agents`` is empty, ``size`` is outside 1..16, or
            ``size`` disagrees with an explicit multi-agent list.
    """
    raw = _as_agent_list(agents)
    if not raw:
        raise ValueError("agents must include at least one model.")
    members = [member for item in raw for member in flatten_agent(item)]
    if not members:
        raise ValueError("agents must include at least one model.")
    if len(members) == 1:
        return [members[0]] * _validate_swarm_size(1 if size is None else size)
    if size is not None and size != len(members):
        raise ValueError(
            "size cannot be combined with a multi-agent list; "
            "pass one model plus size, or the full agent list."
        )
    _validate_swarm_size(len(members))
    return members


def _as_agent_list(agents: SwarmAgents) -> list[AgentInput]:
    """Accept one agent or a sequence of agents without splitting a model id."""
    if isinstance(agents, str | SwarmMember | DefinedAgent | RemoteAgent) or not isinstance(
        agents, Sequence
    ):
        return [cast(AgentInput, agents)]
    return [cast(AgentInput, agent) for agent in agents]


def _agent_instructions(base: str | None, index: int, total: int) -> str | None:
    """Add an independence/recall role to each agent when the swarm has peers."""
    if total == 1:
        return base
    role = (
        f"You are extraction agent {index + 1} of {total}. Work independently. "
        "Prefer recall: extract every matching record you can justify from the source."
    )
    return f"{base.strip()}\n\n{role}" if base and base.strip() else role


def _remote_label(agent: RemoteAgent) -> str:
    """Identify a remote agent in results without resolving a lazy URL."""
    return agent.url if isinstance(agent.url, str) else agent.description


async def _run_member(
    schema: type[T],
    member: SwarmMember,
    index: int,
    total: int,
    file_bytes: bytes,
    file_type: str,
    *,
    instructions: str | None,
    style: ExtractionStyle,
    media_type: str | None,
    source_label: str | None,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
    cite: bool,
) -> ExtractionResult[T]:
    """Run one swarm agent over already-loaded media and build its result."""
    started = time.perf_counter()
    attempts = 0
    member_style = style if member.style is None else normalize_style(member.style)
    member_instructions = _agent_instructions(
        instructions if member.instructions is None else member.instructions, index, total
    )
    run_schema, member_instructions = prepare_cited_run(schema, member_instructions, cite)
    parsed_inputs, parsed = maybe_parsed_inputs(file_bytes, file_type, parse=cite)
    if isinstance(member.model, RemoteAgent):
        output, usage, attempts = await run_remote_extraction(
            run_schema,
            member.model,
            file_bytes,
            file_type,
            instructions=member_instructions,
            style=member_style,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        )
        output, citations = split_cited_output(output, schema, cite=cite, parsed=parsed)
        return ExtractionResult(
            output=output,
            usage=usage,
            attempts=attempts,
            duration=time.perf_counter() - started,
            model=_remote_label(member.model),
            media_type=media_type,
            source=source_label,
            warnings=(),
            citations=citations,
        )
    with prepared_style_run(member_style, file_bytes, file_type) as (capabilities, style_inputs):
        with _extraction_errors():
            agent = _build_agent(
                run_schema,
                member.model,
                member_instructions,
                extra_capabilities=capabilities,
            )
        inputs = (
            parsed_inputs
            if parsed_inputs is not None and style_inputs is None
            else _resolve_run_inputs(file_bytes, file_type, style_inputs)
        )

        async def _run(window: list) -> tuple[object, Usage]:
            nonlocal attempts
            attempts += 1
            result = await _run_extraction_async(agent, window)
            return result.output, _usage_from_result(result)

        output, usage, citations = await extract_windows_async(
            _run,
            inputs,
            parsed,
            schema,
            cite,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        )
    return ExtractionResult(
        output=output,
        usage=usage,
        attempts=attempts,
        duration=time.perf_counter() - started,
        model=_model_identifier(member.model, agent),
        media_type=media_type,
        source=source_label,
        warnings=(),
        citations=citations,
    )


async def _run_swarm(
    schema: type[T],
    agents: SwarmAgents,
    input_file: ExtractionInputLike,
    instructions: str | None,
    *,
    size: int | None,
    style: ExtractionStyle | str,
    reduce: SwarmReduce | str,
    media_type: str | None,
    max_input_bytes: int | None,
    max_concurrency: int | None,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
    on_agent_start: Callable[[int, int], None] | None,
    on_agent: Callable[[int, int, ExtractionResult[T] | Exception], None] | None,
    cite: bool = False,
) -> SwarmResult[T]:
    """Load the input once, fan it out across agents, and reduce the outputs."""
    members = resolve_swarm_members(agents, size)
    strategy = normalize_reduce(reduce)
    resolved_style = normalize_style(style)
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    concurrency = (
        min(_DEFAULT_SWARM_CONCURRENCY, len(members))
        if max_concurrency is None
        else max_concurrency
    )
    _validate_max_concurrency(concurrency)
    limit = _resolve_max_input_bytes(max_input_bytes)
    source, item_media_type, name = _resolve_item(input_file, media_type)

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=_url_fetch_timeout(),
    ) as client:
        with _extraction_errors():
            file_bytes, file_type = await _get_media_async(
                source,
                client,
                media_type=item_media_type,
                max_input_bytes=limit,
            )

    source_label = _item_source_label(source, name)
    results: list[ExtractionResult[T] | Exception] = [
        cast("ExtractionResult[T] | Exception", None)
    ] * len(members)
    counter = iter(range(len(members)))

    async def _worker() -> None:
        for index in counter:
            if on_agent_start is not None:
                on_agent_start(index, len(members))
            try:
                results[index] = await _run_member(
                    schema,
                    members[index],
                    index,
                    len(members),
                    file_bytes,
                    file_type,
                    instructions=instructions,
                    style=resolved_style,
                    media_type=item_media_type,
                    source_label=source_label,
                    max_retries=max_retries,
                    retry_backoff=retry_backoff,
                    retry_max_backoff=retry_max_backoff,
                    cite=cite,
                )
            except Exception as exc:
                results[index] = exc
            if on_agent is not None:
                on_agent(index, len(members), results[index])

    await asyncio.gather(*(_worker() for _ in range(min(concurrency, len(members)))))

    successes = [item for item in results if not isinstance(item, Exception)]
    if not successes:
        raise cast(Exception, results[0])
    output = reduce_outputs([item.output for item in successes], strategy)
    return SwarmResult(
        output=output,
        agents=tuple(results),
        usage=total_usage(successes),
        reduce=strategy,
        citations=reduce_citations(
            [(item.output, item.citations) for item in successes],
            output,
            strategy,
        ),
    )


def _swarm_sync(name: str, **kwargs: Any) -> Any:
    """Run the async swarm from sync code, refusing to nest inside a loop."""
    _require_no_running_loop(name)
    return asyncio.run(_run_swarm(**kwargs))


def extract_swarm(
    schema: type[T],
    agents: SwarmAgents,
    input_file: ExtractionInputLike,
    instructions: str | None = None,
    *,
    size: int | None = None,
    style: ExtractionStyle | str = "direct",
    reduce: SwarmReduce | str = "merge",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
    cite: bool = False,
) -> T:
    """Run several agents over one input and return the reduced result.

    The input is loaded once and shared by every agent, so a swarm costs one
    fetch and N model calls. Agents run concurrently and independently; each
    one is told its position so it does not assume a peer covered a section.

    Args:
        schema: A Pydantic model class defining the expected output structure.
        agents: One agent or a list of agents. An agent is a model identifier,
            a configured pydantic-ai ``Model``, or a :class:`SwarmMember` with
            per-agent ``instructions`` / ``style``.
        input_file: A local path, ``http(s)://`` URL, ``os.PathLike``, raw
            ``bytes``, a binary file-like object, or an ``ExtractionInput``.
        instructions: Optional guidance applied to agents without their own.
        size: Number of copies to run when ``agents`` is a single agent
            (1..16). Invalid with a multi-agent list unless it matches its
            length.
        style: Swarm-wide extraction style; a :class:`SwarmMember` may override
            it.
        reduce: How to fold the agent outputs — ``merge`` (default), ``vote``,
            or ``first``.
        media_type: Optional MIME type. Required for ``bytes`` and file-like
            inputs.
        max_input_bytes: Byte limit for the input. ``None`` uses
            ``OPENEXTRACT_MAX_INPUT_BYTES`` or the 50 MiB default.
        max_concurrency: Maximum agents in flight. Defaults to ``min(5, agents)``.
        max_retries: Per-agent retries after a transient ``ModelError``.
        retry_backoff: Base backoff seconds between per-agent retries.
        retry_max_backoff: Maximum per-agent retry delay in seconds.
        cite: When ``True``, each agent is asked for per-field source spans.
            ``extract_swarm`` still returns the reduced schema instance;
            citations land on :class:`SwarmResult` from the ``*_with_results``
            APIs.

    Returns:
        The reduced schema instance.

    Raises:
        ValueError: If ``agents`` is empty, ``size`` is out of range or
            disagrees with the agent list, or a retry/concurrency option is
            invalid. Raised before any model call.
        ExtractionError: The first agent's failure, when every agent failed.
        RuntimeError: If called from a running event loop. Use
            :func:`extract_swarm_async` in async code instead.
    """
    return cast(
        "SwarmResult[T]",
        _swarm_sync(
            "extract_swarm",
            schema=schema,
            agents=agents,
            input_file=input_file,
            instructions=instructions,
            size=size,
            style=style,
            reduce=reduce,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            on_agent_start=None,
            on_agent=None,
            cite=cite,
        ),
    ).output


async def extract_swarm_async(
    schema: type[T],
    agents: SwarmAgents,
    input_file: ExtractionInputLike,
    instructions: str | None = None,
    *,
    size: int | None = None,
    style: ExtractionStyle | str = "direct",
    reduce: SwarmReduce | str = "merge",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
    cite: bool = False,
) -> T:
    """Async sibling of :func:`extract_swarm`."""
    result = await _run_swarm(
        schema,
        agents,
        input_file,
        instructions,
        size=size,
        style=style,
        reduce=reduce,
        media_type=media_type,
        max_input_bytes=max_input_bytes,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_max_backoff=retry_max_backoff,
        on_agent_start=None,
        on_agent=None,
        cite=cite,
    )
    return result.output


def extract_swarm_with_results(
    schema: type[T],
    agents: SwarmAgents,
    input_file: ExtractionInputLike,
    instructions: str | None = None,
    *,
    size: int | None = None,
    style: ExtractionStyle | str = "direct",
    reduce: SwarmReduce | str = "merge",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
    on_agent_start: Callable[[int, int], None] | None = None,
    on_agent: Callable[[int, int, ExtractionResult[T] | Exception], None] | None = None,
    cite: bool = False,
) -> SwarmResult[T]:
    """Run a swarm and return the reduced output plus per-agent diagnostics.

    Mirrors :func:`extract_swarm` and additionally reports every agent's
    :class:`ExtractionResult` (or its exception), the summed token usage, the
    reduce strategy that produced the output, and reduced ``citations`` when
    ``cite=True``. ``on_agent_start`` and ``on_agent`` are called with
    ``(index, total)`` and ``(index, total, result)`` for progress reporting.
    """
    return cast(
        "SwarmResult[T]",
        _swarm_sync(
            "extract_swarm_with_results",
            schema=schema,
            agents=agents,
            input_file=input_file,
            instructions=instructions,
            size=size,
            style=style,
            reduce=reduce,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
            on_agent_start=on_agent_start,
            on_agent=on_agent,
            cite=cite,
        ),
    )


async def extract_swarm_with_results_async(
    schema: type[T],
    agents: SwarmAgents,
    input_file: ExtractionInputLike,
    instructions: str | None = None,
    *,
    size: int | None = None,
    style: ExtractionStyle | str = "direct",
    reduce: SwarmReduce | str = "merge",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
    on_agent_start: Callable[[int, int], None] | None = None,
    on_agent: Callable[[int, int, ExtractionResult[T] | Exception], None] | None = None,
    cite: bool = False,
) -> SwarmResult[T]:
    """Async sibling of :func:`extract_swarm_with_results`."""
    return await _run_swarm(
        schema,
        agents,
        input_file,
        instructions,
        size=size,
        style=style,
        reduce=reduce,
        media_type=media_type,
        max_input_bytes=max_input_bytes,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_max_backoff=retry_max_backoff,
        on_agent_start=on_agent_start,
        on_agent=on_agent,
        cite=cite,
    )
