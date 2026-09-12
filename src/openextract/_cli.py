"""Command-line interface for openextract."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from collections.abc import AsyncGenerator, Sequence
from typing import Any, BinaryIO, cast

from dotenv import load_dotenv
from pydantic import BaseModel

from ._agents import DefinedAgent, RemoteAgent, is_agent, load_agent, load_agents
from ._batch import _BatchOptions, _iter_extractions, extract_many_with_results
from ._config import _validate_max_concurrency
from ._extract import _plan_agent, extract, extract_with_usage
from ._reduce import SwarmReduce
from ._styles import ExtractionStyle
from ._swarm import extract_swarm, extract_swarm_with_results
from ._types import Citation, ExtractionInput, ExtractionInputLike, ExtractionResult
from .exceptions import (
    ExtractionError,
    ModelError,
    ProviderNotInstalledError,
    RemoteAgentError,
    SchemaValidationError,
    UrlFetchError,
)

_EXIT_PARTIAL_FAILURE = 7
_EXIT_REMOTE_AGENT = 8
_EXIT_INTERRUPTED = 130  # 128 + SIGINT, the conventional Ctrl-C exit code
_EXIT_BROKEN_PIPE = 141  # 128 + SIGPIPE, the conventional broken-pipe exit code
_MANIFEST_KEYS = frozenset({"source", "media_type", "name"})


def _resolve_schema(schema_path: str) -> type[BaseModel]:
    """Resolve a ``module:ClassName`` string to a Pydantic ``BaseModel`` subclass."""
    if ":" not in schema_path:
        raise ValueError(
            f"Invalid schema path '{schema_path}'. Expected format 'module:ClassName'."
        )

    module_part, class_name = schema_path.split(":", 1)
    if not module_part or not class_name:
        raise ValueError(
            f"Invalid schema path '{schema_path}'. Expected format 'module:ClassName'."
        )

    module = importlib.import_module(module_part)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"Class '{class_name}' not found in module '{module_part}'.") from exc

    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise ValueError(f"'{schema_path}' does not refer to a Pydantic BaseModel subclass.")

    return cls


def _split_list(value: str | None) -> list[str]:
    """Split a comma-separated CLI list, dropping blank entries."""
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _resolve_agents(args: argparse.Namespace) -> list[DefinedAgent | RemoteAgent]:
    """Load the agents named by ``--agent`` / ``--agents``."""
    if args.agent and args.agents:
        raise ValueError("use --agent or --agents, not both")
    if args.agent:
        return [load_agent(args.agent)]
    if args.agents:
        return load_agents(args.agents)
    return []


def _agent_schema(agents: list[DefinedAgent | RemoteAgent]) -> type[BaseModel] | None:
    """Return the first output schema declared by the loaded agents, if any."""
    for agent in agents:
        if agent.output_schema is not None:
            return agent.output_schema
    return None


def _swarm_agents(
    args: argparse.Namespace,
    agents: list[DefinedAgent | RemoteAgent],
) -> list | None:
    """Build the swarm agent list, or ``None`` when this is a one-shot run.

    ``--agents`` and ``--models`` list the agents outright; ``--swarm`` fans a
    single model out. A lone ``--agent`` is not a swarm — it may still fan out
    on its own if it declares subagents, which ``extract`` handles.
    """
    models = _split_list(args.models)
    if len(agents) > 1:
        return agents
    if len(models) > 1:
        return models
    if args.swarm > 1:
        return agents or models or [args.model]
    return None


def _validate_swarm_args(
    args: argparse.Namespace,
    agents: list[DefinedAgent | RemoteAgent],
    items: list[ExtractionInputLike],
) -> None:
    """Reject swarm flag combinations before any input is loaded."""
    if args.swarm < 1:
        raise ValueError("--swarm must be a positive integer")
    models = _split_list(args.models)
    if agents or len(models) > 1 or args.swarm > 1:
        if args.manifest is not None or len(items) != 1:
            raise ValueError(
                "--swarm, --models, --agent, and --agents apply to a single input; "
                "omit them for batch files and manifests"
            )
        if args.output == "jsonl":
            raise ValueError(
                "--swarm, --models, --agent, and --agents produce a single result; "
                "use --output json or repr"
            )
    if len(models) > 1 and args.swarm > 1 and args.swarm != len(models):
        raise ValueError("--swarm does not match the number of --models")
    if args.models and args.model:
        raise ValueError("use --model or --models, not both")


def _resolve_input_files(
    raw_inputs: list[str],
    *,
    media_type: str | None,
) -> list[str | bytes | BinaryIO]:
    """Resolve CLI paths, URLs, or stdin ``-`` to values accepted by ``extract``."""
    if "-" in raw_inputs:
        if len(raw_inputs) > 1:
            raise ValueError("stdin (-) cannot be combined with other input files")
        if not media_type:
            raise ValueError("--media-type is required when reading from stdin (-)")
        return [sys.stdin.buffer]

    return cast(list[str | bytes | BinaryIO], raw_inputs)


def _parse_manifest_entry(text: str, line_number: int) -> ExtractionInput:
    """Validate one manifest line and build its ``ExtractionInput``."""
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest line {line_number}: invalid JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"manifest line {line_number}: expected a JSON object")
    unknown = sorted(set(record) - _MANIFEST_KEYS)
    if unknown:
        raise ValueError(f"manifest line {line_number}: unknown keys: {', '.join(unknown)}")
    source = record.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(f"manifest line {line_number}: 'source' must be a non-empty string")
    if source == "-":
        raise ValueError(f"manifest line {line_number}: stdin (-) is not supported in manifests")
    media_type = record.get("media_type")
    if media_type is not None and not isinstance(media_type, str):
        raise ValueError(f"manifest line {line_number}: 'media_type' must be a string")
    name = record.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"manifest line {line_number}: 'name' must be a string")
    return ExtractionInput(source=source, media_type=media_type, name=name)


def _load_manifest(path: str) -> list[ExtractionInput]:
    """Parse a JSONL manifest file into per-item extraction inputs."""
    entries: list[ExtractionInput] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            entries.append(_parse_manifest_entry(text, line_number))
    if not entries:
        raise ValueError(f"manifest '{path}' contains no entries")
    return entries


def _resolve_cli_inputs(
    args: argparse.Namespace,
) -> tuple[list[ExtractionInputLike], list[str]]:
    """Resolve positional inputs or a manifest into items plus display labels."""
    if args.manifest is not None:
        if args.input_files:
            raise ValueError("--manifest cannot be combined with positional input files")
        entries = _load_manifest(args.manifest)
        return list(entries), [entry.name or cast(str, entry.source) for entry in entries]
    if not args.input_files:
        raise ValueError("provide one or more input files, or --manifest")
    resolved = _resolve_input_files(args.input_files, media_type=args.media_type)
    return cast(list[ExtractionInputLike], resolved), list(args.input_files)


def _usage_payload(usage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _citations_payload(citations: Sequence[Citation]) -> list[dict[str, Any]]:
    """Serialize citations as ``{field, quote, page}`` plus optional ``bbox``."""
    payload: list[dict[str, Any]] = []
    for citation in citations:
        item: dict[str, Any] = {
            "field": citation.field,
            "quote": citation.quote,
            "page": citation.page,
        }
        if citation.bbox is not None:
            item["bbox"] = list(citation.bbox)
        payload.append(item)
    return payload


def _result_dump(result: object) -> dict[str, Any]:
    if isinstance(result, ExtractionResult):
        return cast(BaseModel, result.output).model_dump()
    return cast(BaseModel, result).model_dump()


def _result_citations(result: object) -> list[dict[str, Any]]:
    if isinstance(result, ExtractionResult):
        return _citations_payload(result.citations)
    return []


def _swarm_citations(swarm: Any) -> list[dict[str, Any]]:
    citations: list[Citation] = []
    for agent in swarm.agents:
        if isinstance(agent, ExtractionResult):
            citations.extend(agent.citations)
    return _citations_payload(citations)


def _failure_record(label: str, error: BaseException) -> dict[str, Any]:
    return {
        "input": label,
        "error": str(error),
        "error_type": type(error).__name__,
    }


def _item_record(
    label: str,
    result: object,
    *,
    with_usage: bool,
    with_citations: bool,
) -> dict[str, Any]:
    """Build the labeled record for one completed batch item."""
    if isinstance(result, BaseException):
        return _failure_record(label, result)
    record: dict[str, Any] = {"input": label, "result": _result_dump(result)}
    if with_usage:
        record["usage"] = _usage_payload(cast(ExtractionResult[Any], result).usage)
    if with_citations:
        record["citations"] = _result_citations(result)
    return record


def _array_entry(
    label: str,
    result: object,
    *,
    with_usage: bool,
    with_citations: bool,
) -> Any:
    """Build one JSON-array entry, keeping the legacy bare shape for successes."""
    if with_usage or isinstance(result, BaseException):
        return _item_record(label, result, with_usage=with_usage, with_citations=with_citations)
    if with_citations:
        return {"result": _result_dump(result), "citations": _result_citations(result)}
    return _result_dump(result)


def _print_json(payload: Any, *, as_repr: bool) -> None:
    if as_repr:
        print(repr(payload), flush=True)
        return
    print(json.dumps(payload, indent=2, default=str), flush=True)


def _emit_json_line(record: dict[str, Any]) -> None:
    """Write one JSONL record and flush so consumers see it immediately."""
    print(json.dumps(record, default=str), flush=True)


def _discard_stdout() -> None:
    """Point stdout at ``os.devnull`` so interpreter-exit flushes stay quiet."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
    except OSError:
        pass


def _print_batch_payload(
    ordered: list[tuple[int, object]],
    labels: list[str],
    usage_totals: dict[str, int],
    *,
    with_usage: bool,
    with_citations: bool,
    as_repr: bool,
) -> None:
    """Emit the buffered batch payload in input order."""
    ordered.sort(key=lambda pair: pair[0])
    entries = [
        _array_entry(
            labels[index],
            result,
            with_usage=with_usage,
            with_citations=with_citations,
        )
        for index, result in ordered
    ]
    payload: Any = {"results": entries, "usage": usage_totals} if with_usage else entries
    _print_json(payload, as_repr=as_repr)


async def _run_batch_async(
    schema_cls: type[BaseModel],
    items: list[ExtractionInputLike],
    labels: list[str],
    options: _BatchOptions,
    args: argparse.Namespace,
    model: str,
) -> int:
    """Stream the batch, emitting records (JSONL) or buffering them (array)."""
    with_usage = args.usage
    with_citations = args.cite
    jsonl = args.output == "jsonl"
    # _iter_extractions is an async generator function; its AsyncIterator return
    # annotation hides the aclose() needed for deterministic finalization.
    stream = cast(
        "AsyncGenerator[tuple[int, object], None]",
        _iter_extractions(schema_cls, model, items, options),
    )
    total = len(items)
    ordered: list[tuple[int, object]] = []
    completed = 0
    failed = 0
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        async for index, result in stream:
            completed += 1
            if isinstance(result, BaseException):
                failed += 1
            elif with_usage:
                usage = cast(ExtractionResult[Any], result).usage
                usage_totals["input_tokens"] += usage.input_tokens
                usage_totals["output_tokens"] += usage.output_tokens
                usage_totals["total_tokens"] += usage.total_tokens
            if jsonl:
                record = _item_record(
                    labels[index],
                    result,
                    with_usage=with_usage,
                    with_citations=with_citations,
                )
                _emit_json_line({"index": index, **record})
            else:
                ordered.append((index, result))
            if args.progress:
                print(
                    f"progress: {completed}/{total} completed ({failed} failed): {labels[index]}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        # Deterministically finalize the generator so outstanding work is
        # cancelled before the event loop closes, even on emit failures.
        await stream.aclose()

    if jsonl:
        if with_usage:
            _emit_json_line({"summary": {"inputs": total, "failed": failed, "usage": usage_totals}})
    else:
        _print_batch_payload(
            ordered,
            labels,
            usage_totals,
            with_usage=with_usage,
            with_citations=with_citations,
            as_repr=args.output == "repr",
        )

    if failed:
        print(
            f"warning: {failed} of {total} input(s) failed; see output for details",
            file=sys.stderr,
        )
        return _EXIT_PARTIAL_FAILURE
    return 0


def _run_batch(
    schema_cls: type[BaseModel],
    items: list[ExtractionInputLike],
    labels: list[str],
    args: argparse.Namespace,
    model: str,
) -> int:
    """Validate batch options before any model call, then run the batch."""
    options = _BatchOptions.resolve(
        args.instructions,
        style=args.style,
        media_type=args.media_type,
        max_input_bytes=args.max_input_bytes,
        max_concurrency=args.max_concurrency,
        return_exceptions=args.continue_on_error,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        retry_max_backoff=args.retry_max_backoff,
        rich=args.usage or args.cite,
        cite=args.cite,
    )
    return asyncio.run(_run_batch_async(schema_cls, items, labels, options, args, model))


def _print_single_payload(payload: Any, args: argparse.Namespace) -> None:
    """Print a single-result payload in the requested output format."""
    if args.output == "repr":
        _print_json(payload, as_repr=True)
    elif not args.usage and isinstance(payload, BaseModel):
        print(payload.model_dump_json(indent=2))
    else:
        _print_json(payload, as_repr=False)


def _run_single(
    schema_cls: type[BaseModel],
    input_file: ExtractionInputLike,
    args: argparse.Namespace,
    model: str | DefinedAgent | RemoteAgent,
) -> int:
    """Run a single extraction with the legacy output shapes."""
    shared: dict[str, Any] = {
        "instructions": args.instructions,
        "style": args.style,
        "media_type": args.media_type,
        "max_input_bytes": args.max_input_bytes,
        "max_retries": args.max_retries,
        "retry_backoff": args.retry_backoff,
        "retry_max_backoff": args.retry_max_backoff,
        "cite": args.cite,
    }
    if args.cite:
        run_model: Any = model
        if is_agent(model):
            run_model, shared["instructions"], shared["style"], use_swarm = _plan_agent(
                model, args.instructions, args.style
            )
            if use_swarm:
                return _run_swarm(schema_cls, [model], input_file, args)
        rich = extract_many_with_results(schema_cls, run_model, [input_file], **shared)[0]
        payload: Any = {"result": rich.output.model_dump()}
        if args.usage:
            payload["usage"] = _usage_payload(rich.usage)
        payload["citations"] = _citations_payload(rich.citations)
    elif args.usage:
        result, usage = extract_with_usage(
            schema=schema_cls,
            model=model,
            input_file=input_file,
            **shared,
        )
        payload = {"result": result.model_dump(), "usage": _usage_payload(usage)}
    else:
        payload = extract(
            schema=schema_cls,
            model=model,
            input_file=input_file,
            **shared,
        )

    _print_single_payload(payload, args)
    return 0


def _run_swarm(
    schema_cls: type[BaseModel],
    swarm_agents: list,
    input_file: ExtractionInputLike,
    args: argparse.Namespace,
) -> int:
    """Run a swarm over one input and print its payload."""
    options: dict[str, Any] = {
        "instructions": args.instructions,
        "size": args.swarm if len(swarm_agents) == 1 else None,
        "style": args.style,
        "reduce": args.reduce,
        "media_type": args.media_type,
        "max_input_bytes": args.max_input_bytes,
        "max_retries": args.max_retries,
        "retry_backoff": args.retry_backoff,
        "retry_max_backoff": args.retry_max_backoff,
        "cite": args.cite,
    }
    if args.usage or args.cite:
        swarm = extract_swarm_with_results(schema_cls, swarm_agents, input_file, **options)
        payload: Any = {"result": swarm.output.model_dump()}
        if args.usage:
            payload["usage"] = _usage_payload(swarm.usage)
            payload["agents"] = len(swarm.agents)
            payload["reduce"] = swarm.reduce.value
        if args.cite:
            payload["citations"] = _swarm_citations(swarm)
    else:
        payload = extract_swarm(schema_cls, swarm_agents, input_file, **options)
    _print_single_payload(payload, args)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openextract",
        description="Extract structured data from files or URLs using an LLM.",
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        metavar="input_file",
        help=(
            "One or more paths/URLs, or '-' to read bytes from stdin. Omit when using --manifest."
        ),
    )
    parser.add_argument(
        "--schema",
        default=None,
        help=(
            "Pydantic model import path in 'module:ClassName' form. Optional "
            "when --agent/--agents supplies an output schema."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="pydantic-ai model identifier (e.g. 'xai:grok-4.3').",
    )
    parser.add_argument(
        "--models",
        default=None,
        metavar="ID,ID",
        help="Comma-separated model identifiers, one per swarm agent.",
    )
    parser.add_argument(
        "--agent",
        default=None,
        metavar="SPEC",
        help="Agent to extract with: a directory, a Python file, or 'module:attribute'.",
    )
    parser.add_argument(
        "--agents",
        default=None,
        metavar="SPEC,SPEC",
        help="Comma-separated agent specs, one per swarm agent.",
    )
    parser.add_argument(
        "--swarm",
        type=int,
        default=1,
        metavar="N",
        help="Run N parallel agents over a single input (default 1).",
    )
    parser.add_argument(
        "--reduce",
        choices=tuple(item.value for item in SwarmReduce),
        default=SwarmReduce.MERGE.value,
        help="How a swarm folds agent outputs: 'merge' (default), 'vote', or 'first'.",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Optional natural-language instructions for the model.",
    )
    parser.add_argument(
        "--style",
        choices=tuple(item.value for item in ExtractionStyle),
        default=ExtractionStyle.DIRECT.value,
        help=(
            "Extraction style: 'direct' (default) sends media to the model; "
            "'search' uses file tools on text; 'code' writes Python against text."
        ),
    )
    parser.add_argument(
        "--media-type",
        default=None,
        metavar="MIME",
        help=(
            "MIME type (required for stdin; optional override for paths/URLs "
            "and fallback for manifest entries without their own)."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        metavar="FILE",
        help=(
            'JSONL file of inputs, one {"source": ..., "media_type"?: ..., '
            '"name"?: ...} object per line. Mutually exclusive with positional '
            "input files; always uses batch semantics."
        ),
    )
    parser.add_argument(
        "--usage",
        action="store_true",
        help=(
            "Include token usage: single inputs keep the {result, usage} shape; "
            "batches report per-item and aggregate usage."
        ),
    )
    parser.add_argument(
        "--cite",
        action="store_true",
        help=(
            "Request per-field source citations (field, quote, page, optional bbox). "
            "JSON/jsonl include a citations array. Same cite=True path as the library."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Batch only: keep going when an input fails; report per-item errors "
            "inline and exit 7 if any failed (default: abort on first failure)."
        ),
    )
    parser.add_argument(
        "--output",
        choices=("json", "jsonl", "repr"),
        default="json",
        help=(
            "Output format: 'json' (default), 'jsonl' (one record per completed "
            "input, written incrementally), or 'repr'."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        metavar="N",
        help="Maximum in-flight extractions for batch runs (default 5).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Batch only: report per-item completion progress on stderr.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        metavar="N",
        help="Retry up to N times on ModelError with exponential backoff (default 0).",
    )
    parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum bytes to load per input (default: OPENEXTRACT_MAX_INPUT_BYTES or 52428800)."
        ),
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Base backoff in seconds for retry (default 1.0).",
    )
    parser.add_argument(
        "--retry-max-backoff",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Maximum retry delay in seconds, including Retry-After (default 60.0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the openextract CLI. Returns the exit code."""
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        agents = _resolve_agents(args)
        schema_cls = _resolve_schema(args.schema) if args.schema else _agent_schema(agents)
        if schema_cls is None:
            raise ValueError(
                "--schema is required unless --agent/--agents supplies an output schema"
            )
        if not args.model and not args.models and not agents:
            raise ValueError("--model, --models, or --agent/--agents is required")
    except (ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        items, labels = _resolve_cli_inputs(args)
        _validate_swarm_args(args, agents, items)
    except OSError as exc:
        print(f"error: cannot read manifest: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    swarm_agents = _swarm_agents(args, agents)
    single_model = agents[0] if agents else (args.model or _split_list(args.models)[0])

    try:
        _validate_max_concurrency(args.max_concurrency)
        if swarm_agents is not None:
            return _run_swarm(schema_cls, swarm_agents, items[0], args)
        if args.manifest is not None or len(items) > 1 or args.output == "jsonl":
            return _run_batch(schema_cls, items, labels, args, cast(str, single_model))
        return _run_single(schema_cls, items[0], args, single_model)
    except BrokenPipeError:
        _discard_stdout()
        return _EXIT_BROKEN_PIPE
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return _EXIT_INTERRUPTED
    except UrlFetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SchemaValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except RemoteAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_REMOTE_AGENT
    except ProviderNotInstalledError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 6
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
