"""Run LlamaIndex ExtractBench through openextract with any model.

ExtractBench scores schema-guided extraction on 370 enterprise documents.
This wrapper registers an openextract pipeline so you can evaluate any
``pydantic-ai`` model identifier without editing ExtractBench itself.

Quick start (6 documents, cents on a hosted API)::

    uv run python scripts/extractbench.py --model openai:gpt-5 --test

Full run (370 documents; metered APIs can cost tens to hundreds of dollars)::

    uv run python scripts/extractbench.py --model xai:grok-4.3
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from openextract import (
    Citation,
    ExtractionError,
    Extractor,
    InputTooLargeError,
    ModelError,
    ProviderNotInstalledError,
    RetryPolicy,
    SchemaValidationError,
    Usage,
    reduce_outputs,
)
from openextract._agent import _install_hint, _route_model, _usage_model_settings
from openextract._citations import (
    citations_from_payload,
    field_citations_for_extractbench,
    json_schema_with_citations,
    with_citation_instructions,
)
from openextract._errors import _is_output_retry_error
from openextract._media import _get_media_type
from openextract._parse import (
    align_citations_to_window,
    ground_citations,
    maybe_parsed_inputs,
    parse_windows,
)
from openextract._types import _sum_usage

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".extractbench"
VENV_DIR = CACHE_DIR / "venv"
# Pinned to a commit so scores stay comparable across bootstraps and upstream
# changes cannot break the runner (it reaches into private ExtractBench
# internals). Bump the SHA deliberately when adopting a newer ExtractBench.
EXTRACTBENCH_GIT = os.environ.get(
    "OPENEXTRACT_EXTRACTBENCH_GIT",
    "git+https://github.com/run-llama/ExtractBench.git@28dadf58fcd1ab366808ac8e2dfc8716fa9ad5aa",
)
DEFAULT_INSTRUCTIONS = (
    "You are extracting structured data from a document according to the "
    "provided JSON schema. Return only the JSON that matches the schema. Use "
    "null for fields not present in the document. When the schema includes a "
    "list field, populate every relevant row visible in the document — do not "
    "return an empty list when rows are present."
)
DEFAULT_MAX_INPUT_BYTES = 500 * 1024 * 1024
DEFAULT_CALL_TIMEOUT = 240.0
DEFAULT_WINDOW_CONCURRENCY = 4
# ExtractBench InferenceRunner wraps each file at 1800s and retries twice.
# Window-pool wait for Veralto (41) / long (66) is above that, so the file
# cap must be ≥ pool budget. 96 windows covers the 6-doc long split with headroom.
EXTRACTBENCH_FILE_TIMEOUT_FLOOR = 1800.0
DEFAULT_FILE_TIMEOUT_WINDOWS = 96
# Two extra batches: +1 still killed pueblo at 960s under GLM variance.
WINDOW_POOL_SLACK_BATCHES = 2
# After the pool budget, wait briefly for in-flight windows before discarding.
WINDOW_POOL_LEFTOVER_GRACE = 15.0
# pydantic-ai default output_retries=1 is the Veralto "exceeded maximum
# output retries (1)" / empty-tool-call kill. Raise it on window extract.
WINDOW_OUTPUT_RETRIES = 3
# Extra Extractor attempt when pydantic-ai still raises after output retries.
WINDOW_OUTPUT_ATTEMPTS = 2
_REF_PREFIXES = ("#/$defs/", "#/definitions/")


class ExtractedDocument(BaseModel):
    """Passthrough container so ExtractBench JSON is not strict-revalidated."""

    model_config = ConfigDict(extra="allow")


def pipeline_name_for_model(model: str, override: str | None = None) -> str:
    """Stable ExtractBench pipeline name for a pydantic-ai model id."""
    if override:
        return override
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return f"openextract_{slug}" if slug else "openextract"


def add_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively set ``additionalProperties: false`` on every object schema."""
    out = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
            for value in (node.get("properties") or {}).values():
                walk(value)
            if "items" in node:
                walk(node["items"])
            for key in ("anyOf", "oneOf", "allOf"):
                for branch in node.get(key) or []:
                    walk(branch)
            for key in ("$defs", "definitions"):
                for definition in (node.get(key) or {}).values():
                    walk(definition)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(out)
    return out


def inline_json_schema_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace local ``$ref`` pointers with the referenced definitions.

    Recursive references are left unresolved so ``StructuredDict`` can fall
    back to prompt-only schema guidance instead of looping.
    """
    out = copy.deepcopy(schema)
    defs = dict(out.get("$defs") or {})
    defs.update(out.get("definitions") or {})

    def resolve(node: Any, stack: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, stack) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str):
            name = None
            for prefix in _REF_PREFIXES:
                if ref.startswith(prefix):
                    name = ref[len(prefix) :]
                    break
            if name is not None and name in defs and name not in stack:
                merged = copy.deepcopy(defs[name])
                merged.update({key: value for key, value in node.items() if key != "$ref"})
                return resolve(merged, stack | {name})
            return node
        return {
            key: resolve(value, stack)
            for key, value in node.items()
            if key not in ("$defs", "definitions")
        }

    return resolve(out, frozenset())


def prepare_schema(
    schema: dict[str, Any], *, additional_properties_false: bool = True
) -> dict[str, Any]:
    """Normalize an ExtractBench JSON schema for structured-output calls."""
    prepared = (
        add_additional_properties_false(schema)
        if additional_properties_false
        else copy.deepcopy(schema)
    )
    return inline_json_schema_defs(prepared)


def as_extracted_dict(output: object) -> dict[str, Any]:
    """Return a plain dict from Extractor output without inventing fields."""
    if isinstance(output, ExtractedDocument):
        output = output.model_dump()
    elif isinstance(output, BaseModel):
        dumped = output.model_dump()
        output = dumped["root"] if list(dumped.keys()) == ["root"] else dumped
    if not isinstance(output, dict):
        raise TypeError(f"Expected dict extraction, got {type(output).__name__}")
    return output


def _output_type_for_schema(schema: dict[str, Any], model: str | object) -> object:
    """Build the structured output type for a schema, refusing to degrade.

    Running schema-less while the prompt references "the provided JSON schema"
    would produce a near-zero score misattributed to the model, so any failure
    here is a permanent, loudly-reported error rather than a silent fallback.
    """
    schema_name = schema.get("title") or "extraction"
    try:
        from pydantic_ai.output import NativeOutput, StructuredDict
    except ImportError as exc:
        message = (
            f"Cannot build structured output for schema {schema_name!r}: this "
            f"pydantic-ai version lacks StructuredDict ({exc}). Upgrade pydantic-ai."
        )
        print(f"warning: {message}", file=sys.stderr)
        raise SchemaValidationError(message) from exc

    try:
        output_type: object = StructuredDict(schema, name="extraction")
    except Exception as exc:
        message = (
            f"JSON schema {schema_name!r} could not be converted into a "
            f"structured output type: {exc}"
        )
        print(f"warning: {message}", file=sys.stderr)
        raise SchemaValidationError(message) from exc
    if isinstance(model, str) and model.startswith("ollama"):
        return NativeOutput(output_type)
    return output_type


def _build_schema_agent(
    model: str | object,
    schema: dict[str, Any],
    instructions: str,
    *,
    timeout: float | None = None,
):
    from pydantic_ai import Agent

    output_type = _output_type_for_schema(schema, model)
    routed = _route_model(model) if isinstance(model, str) else model
    settings = _usage_model_settings(model, None) if isinstance(model, str) else None
    if timeout is not None:
        merged = dict(settings) if settings is not None else {}
        merged["timeout"] = timeout
        settings = merged
    try:
        return Agent(
            routed,
            output_type=output_type,
            instructions=instructions,
            model_settings=settings,
            output_retries=WINDOW_OUTPUT_RETRIES,
        )
    except ImportError as exc:
        if isinstance(model, str):
            message = (
                f"Model {model!r} needs a provider SDK that is not installed. "
                f"Install it with: {_install_hint(model)} "
                f"(or 'pip install openextract[all]'). Original error: {exc}"
            )
        else:
            message = f"The configured model needs a provider SDK that is not installed: {exc}"
        raise ProviderNotInstalledError(message) from exc


def extract_document(
    source: Path | str | bytes,
    json_schema: dict[str, Any],
    model: str | object,
    *,
    instructions: str = DEFAULT_INSTRUCTIONS,
    media_type: str | None = None,
    max_retries: int = 2,
    max_input_bytes: int | None = DEFAULT_MAX_INPUT_BYTES,
    additional_properties_false: bool = True,
    cite: bool = False,
    timeout: float | None = DEFAULT_CALL_TIMEOUT,
    window_concurrency: int = DEFAULT_WINDOW_CONCURRENCY,
) -> tuple[dict[str, Any], Usage]:
    """Extract one document with openextract using an ExtractBench JSON schema."""
    data, usage, _citations = extract_document_with_citations(
        source,
        json_schema,
        model,
        instructions=instructions,
        media_type=media_type,
        max_retries=max_retries,
        max_input_bytes=max_input_bytes,
        additional_properties_false=additional_properties_false,
        cite=cite,
        timeout=timeout,
        window_concurrency=window_concurrency,
    )
    return data, usage


def extract_document_with_citations(
    source: Path | str | bytes,
    json_schema: dict[str, Any],
    model: str | object,
    *,
    instructions: str = DEFAULT_INSTRUCTIONS,
    media_type: str | None = None,
    max_retries: int = 2,
    max_input_bytes: int | None = DEFAULT_MAX_INPUT_BYTES,
    additional_properties_false: bool = True,
    cite: bool = True,
    timeout: float | None = DEFAULT_CALL_TIMEOUT,
    window_concurrency: int = DEFAULT_WINDOW_CONCURRENCY,
) -> tuple[dict[str, Any], Usage, tuple[Citation, ...]]:
    """Extract one document and return ``(data, usage, citations)``.

    ``cite=True`` (the ExtractBench runner default) wraps the JSON schema with
    an ``output`` / ``citations`` envelope so the model can return per-field
    page and quote evidence. PDFs are parsed locally first and sent as
    page-indexed windows so large documents do not blow the model context.
    Window citations are collected before reduce so a later window's cites
    are not dropped. Boxes come from parser spans, never from the model.
    """
    schema = prepare_schema(json_schema, additional_properties_false=additional_properties_false)
    run_instructions = with_citation_instructions(instructions) if cite else instructions
    if cite:
        schema = json_schema_with_citations(schema)
    run_source, run_media_type, parsed = _parse_then_extract_source(source, media_type)
    policy = RetryPolicy(max_retries=max_retries)
    windows = parse_windows(parsed) if parsed is not None and parsed.pages else ()
    scanned = parsed is not None and parsed.pages and not parsed.has_text()
    if len(windows) > 1 or scanned:
        payloads, usage = _extract_parse_windows(
            windows or (parsed,),
            model,
            schema,
            run_instructions,
            max_input_bytes=max_input_bytes,
            timeout=timeout,
            window_concurrency=window_concurrency,
        )
        data, citations = _merge_window_payloads(payloads, cite=cite)
    else:
        agent = _build_schema_agent(model, schema, run_instructions, timeout=timeout)
        with Extractor(
            ExtractedDocument,
            agent=agent,
            retry_policy=policy,
            max_input_bytes=max_input_bytes,
        ) as extractor:
            output, usage = extractor.extract_with_usage(run_source, media_type=run_media_type)
        data, citations = _unwrap_cited_document(as_extracted_dict(output), cite=cite)
    if cite:
        citations = ground_citations(citations, parsed, data)
    return data, usage, citations


def window_pool_timeout(n_windows: int, concurrency: int, timeout: float | None) -> float | None:
    """Wall budget for a windowed extract: per-call timeout times batches, plus slack.

    ``--timeout`` is per window. Collecting every future with that same value
    kills a 10-page doc at 240s even when each page would finish. Scale by
    ``ceil(windows / concurrency) + 2`` so pueblo's 11-window run is not
    killed at 960s when GLM variance exceeds one extra batch.
    """
    if timeout is None:
        return None
    if n_windows <= 0:
        return timeout
    workers = max(1, min(int(concurrency), n_windows))
    batches = math.ceil(n_windows / workers)
    return timeout * (batches + WINDOW_POOL_SLACK_BATCHES)


def extractbench_file_timeout(
    n_windows: int,
    concurrency: int,
    timeout: float | None,
    *,
    floor: float = EXTRACTBENCH_FILE_TIMEOUT_FLOOR,
) -> float | None:
    """Per-file ExtractBench cap: at least the window-pool wait, never below ``floor``.

    The runner's 1800s wrap is independent of ``window_pool_timeout``. A 1-page
    call can still take longer than ``--timeout`` (W14 ~298s at 240s), so keep
    the 1800s floor; raise it when the pool budget is larger (Veralto / long).
    """
    pool = window_pool_timeout(n_windows, concurrency, timeout)
    if pool is None:
        return None
    return max(pool, floor)


def bind_inference_timeouts(
    run: Callable[..., Any],
    per_file_timeout: float | None,
    timeout_retries: int = 0,
) -> Callable[..., Any]:
    """Force ExtractBench inference to use our file timeout and no timeout retries."""

    def bound(self: object, *args: Any, **kwargs: Any) -> Any:
        if per_file_timeout is not None:
            kwargs["per_file_timeout"] = per_file_timeout
        kwargs["timeout_retries"] = timeout_retries
        return run(self, *args, **kwargs)

    bound.__wrapped__ = run  # type: ignore[attr-defined]
    return bound


def _bind_extractbench_timeouts(per_file_timeout: float | None, timeout_retries: int = 0) -> None:
    """Patch ExtractBench's inference CLI so ``cli.run`` cannot 3×1800s a file."""
    from extract_bench.inference.cli import InferenceCLI

    original = getattr(InferenceCLI.run, "__wrapped__", InferenceCLI.run)
    InferenceCLI.run = bind_inference_timeouts(original, per_file_timeout, timeout_retries)


def _window_source(window: object) -> tuple[bytes, str]:
    """Encode one parse window as model input. Never send a raw PDF."""
    if getattr(window, "has_text", lambda: False)():
        return window.as_prompt_text().encode("utf-8"), "text/plain"
    image = next((page.image for page in getattr(window, "pages", ()) if page.image), None)
    if image is not None:
        return image, "image/png"
    return window.as_prompt_text().encode("utf-8"), "text/plain"


def _extract_parse_windows(
    windows: tuple,
    model: str | object,
    schema: dict[str, Any],
    instructions: str,
    *,
    max_input_bytes: int | None,
    timeout: float | None,
    window_concurrency: int,
) -> tuple[list[tuple[dict[str, Any], object]], Usage]:
    """Extract each parse window and sum usage. Citations are unwrapped later.

    Hung windows are not retried and must not burn the 1800s per-file budget
    three times (pueblo / long). Empty-tool-call / output-retry failures are
    retried at the window, not the file. The pool wait is the per-window
    timeout times concurrent batches, plus two batches of slack. On timeout,
    finished windows (and any in-flight that complete quickly) are merged
    instead of discarded. Do not ``shutdown(wait=True)`` after a pool timeout.
    """
    sources = [_window_source(window) for window in windows]
    workers = max(1, min(window_concurrency, len(sources)))
    once = RetryPolicy(max_retries=0)

    def _run(source: bytes, media_type: str) -> tuple[ExtractedDocument, Usage]:
        def _once() -> tuple[ExtractedDocument, Usage]:
            agent = _build_schema_agent(model, schema, instructions, timeout=timeout)
            with Extractor(
                ExtractedDocument,
                agent=agent,
                retry_policy=once,
                max_input_bytes=max_input_bytes,
            ) as extractor:
                return extractor.extract_with_usage(source, media_type=media_type)

        return _run_window_with_output_retries(_once)

    budget = window_pool_timeout(len(sources), workers, timeout)
    if budget is None:
        pairs = [
            (*_run(source, media_type), window)
            for (source, media_type), window in zip(sources, windows, strict=True)
        ]
    else:
        pairs = _run_window_pool(_run, sources, workers, budget, windows)
    return [(as_extracted_dict(part), window) for part, _usage, window in pairs], _sum_usage(
        usage for _part, usage, _window in pairs
    )


def window_pool_leftover_grace(budget: float) -> float:
    """Seconds to wait for in-flight windows after the pool budget expires."""
    if budget <= 0:
        return 0.0
    return min(WINDOW_POOL_LEFTOVER_GRACE, budget * 0.05)


def _run_window_with_output_retries(
    run: Callable[[], tuple[ExtractedDocument, Usage]],
    *,
    attempts: int = WINDOW_OUTPUT_ATTEMPTS,
) -> tuple[ExtractedDocument, Usage]:
    """Retry a window when pydantic-ai exhausts its output-retry budget."""
    last: ExtractionError | None = None
    for attempt in range(attempts):
        try:
            return run()
        except ExtractionError as exc:
            last = exc
            if attempt + 1 >= attempts or not _is_output_retry_error(exc):
                raise
    raise last  # pragma: no cover


def _finished_window_pairs(
    futures: list,
    windows: tuple,
) -> list[tuple[ExtractedDocument, Usage, object]]:
    """Collect successful window results; skip cancelled and failed futures."""
    pairs: list[tuple[ExtractedDocument, Usage, object]] = []
    for future, window in zip(futures, windows, strict=True):
        if not future.done() or future.cancelled():
            continue
        try:
            part, usage = future.result()
        except Exception:
            continue
        pairs.append((part, usage, window))
    return pairs


def _run_window_pool(
    run: Callable[..., tuple[ExtractedDocument, Usage]],
    sources: list[tuple[bytes, str]],
    workers: int,
    budget: float,
    windows: tuple,
) -> list[tuple[ExtractedDocument, Usage, object]]:
    """Run windows concurrently; merge leftovers when the budget expires."""
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(run, source, media_type) for source, media_type in sources]
        _done, pending = wait(futures, timeout=budget)
        if pending:
            _extra, pending = wait(pending, timeout=window_pool_leftover_grace(budget))
        pairs = _finished_window_pairs(futures, windows)
        if pending:
            for future in pending:
                future.cancel()
        if pairs:
            return pairs
        if pending:
            raise ModelError(
                f"Parse window timed out after {budget}s",
                retryable=False,
            )
        for future in futures:
            future.result()
        raise ModelError(
            f"Parse window timed out after {budget}s",
            retryable=False,
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _extractbench_retryable(exc: ModelError) -> bool:
    """True when ExtractBench should retry the whole file.

    Token-limit, request timeouts, and window output-retry failures are
    permanent at file level. Retrying them is how pueblo/long spent 1800s
    three times, and how Veralto would re-run 41 pages after one empty call.
    """
    if not exc.retryable or _is_output_retry_error(exc):
        return False
    return "timeout" not in str(exc).lower()


def _merge_window_payloads(
    payloads: list[dict[str, Any] | tuple[dict[str, Any], object]], *, cite: bool
) -> tuple[dict[str, Any], tuple[Citation, ...]]:
    """Reduce window extractions and keep every window's citations."""
    data_parts: list[ExtractedDocument] = []
    citations: list[Citation] = []
    for payload in payloads:
        window = None
        if isinstance(payload, tuple):
            payload, window = payload
        data, cites = _unwrap_cited_document(payload, cite=cite)
        if cite and window is not None:
            cites = align_citations_to_window(cites, window)
        data_parts.append(ExtractedDocument.model_validate(data))
        citations.extend(cites)
    merged = as_extracted_dict(reduce_outputs(data_parts)) if data_parts else {}
    return merged, tuple(citations)


def _parse_then_extract_source(
    source: Path | str | bytes, media_type: str | None
) -> tuple[Path | str | bytes, str | None, object]:
    """Load bytes, parse locally, and feed text or page images — never a PDF upload."""
    data, resolved_type = _source_bytes(source, media_type)
    parsed_inputs, parsed = maybe_parsed_inputs(data, resolved_type or "", parse=True)
    if parsed is not None and parsed.has_text() and parsed_inputs is not None:
        text = parsed_inputs[1]
        if isinstance(text, str):
            return text.encode("utf-8"), "text/plain", parsed
    if parsed is not None and parsed.has_images():
        image = next(page.image for page in parsed.pages if page.image)
        return image, "image/png", parsed
    if parsed is not None and parsed.pages and not parsed.has_text():
        return parsed.as_prompt_text().encode("utf-8"), "text/plain", parsed
    return source, media_type or resolved_type, parsed


def _source_bytes(source: Path | str | bytes, media_type: str | None) -> tuple[bytes, str | None]:
    if isinstance(source, bytes):
        return source, media_type
    path = Path(source)
    return path.read_bytes(), media_type or _get_media_type(str(path))


def _unwrap_cited_document(
    dumped: dict[str, Any], *, cite: bool
) -> tuple[dict[str, Any], tuple[Citation, ...]]:
    """Split a cited envelope, or treat a flat object as the extraction."""
    if not cite:
        return dumped, ()
    nested = dumped.get("output")
    citations = citations_from_payload(dumped.get("citations") or [])
    if isinstance(nested, dict):
        return nested, citations
    return {key: value for key, value in dumped.items() if key != "citations"}, citations


def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def extract_bench_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("extract_bench") is not None


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _inside_benchmark_venv() -> bool:
    """True when this interpreter runs from ``.extractbench/venv``.

    Compares ``sys.prefix`` to the venv directory instead of resolving
    executables: under uv both the project and benchmark interpreters are
    symlinks to the same base Python, so resolved executable paths are equal
    even when the venvs differ.
    """
    return Path(sys.prefix).resolve() == VENV_DIR.resolve()


def _venv_has_extract_bench(python: Path) -> bool:
    """Probe the benchmark venv for an importable ``extract_bench``."""
    if not python.exists():
        return False
    probe = subprocess.run(
        [str(python), "-c", "import extract_bench"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def ensure_extract_bench(*, reexec: bool = True) -> None:
    """Install ExtractBench into ``.extractbench/venv`` when it is not importable."""
    if extract_bench_available():
        return
    if _inside_benchmark_venv():
        raise RuntimeError(
            "extract_bench is not importable from the benchmark environment. "
            f"Delete {VENV_DIR} and re-run with network access so the script "
            f"can reinstall {EXTRACTBENCH_GIT}."
        )
    python = _venv_python()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        print(f"Creating ExtractBench environment at {VENV_DIR}", file=sys.stderr)
        _run(["uv", "venv", str(VENV_DIR)])
    if not _venv_has_extract_bench(python):
        print("Installing ExtractBench and openextract (first run only)...", file=sys.stderr)
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-config",
                EXTRACTBENCH_GIT,
                "-e",
                f"{REPO_ROOT}[all]",
            ]
        )
    if not reexec:
        return
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


def register_openextract_pipeline(
    model: str,
    *,
    pipeline_name: str | None = None,
    instructions: str = DEFAULT_INSTRUCTIONS,
    max_retries: int = 2,
    max_input_bytes: int | None = DEFAULT_MAX_INPUT_BYTES,
    additional_properties_false: bool = True,
    input_price_per_1m: float = 0.0,
    output_price_per_1m: float = 0.0,
    cite: bool = True,
    timeout: float | None = DEFAULT_CALL_TIMEOUT,
    window_concurrency: int = DEFAULT_WINDOW_CONCURRENCY,
) -> str:
    """Register the openextract ExtractBench provider and pipeline; return its name."""
    from extract_bench.inference.pipelines import get_pipeline, register_pipeline
    from extract_bench.inference.providers.base import (
        Provider,
        ProviderConfigError,
        ProviderPermanentError,
        ProviderTransientError,
    )
    from extract_bench.inference.providers.registry import (
        _PROVIDER_REGISTRY,
        register_provider,
    )
    from extract_bench.schemas.extract_output import ExtractOutput
    from extract_bench.schemas.pipeline import PipelineSpec
    from extract_bench.schemas.pipeline_io import (
        InferenceRequest,
        InferenceResult,
        RawInferenceResult,
    )
    from extract_bench.schemas.product import ProductType

    name = pipeline_name_for_model(model, pipeline_name)
    file_timeout = extractbench_file_timeout(
        DEFAULT_FILE_TIMEOUT_WINDOWS, window_concurrency, timeout
    )
    _bind_extractbench_timeouts(file_timeout, timeout_retries=0)
    config = {
        "model": model,
        "instructions": instructions,
        "max_retries": max_retries,
        "max_input_bytes": max_input_bytes,
        "additional_properties_false": additional_properties_false,
        "input_price_per_1m": input_price_per_1m,
        "output_price_per_1m": output_price_per_1m,
        "cite": cite,
        "timeout": timeout,
        "window_concurrency": window_concurrency,
    }

    if "openextract" not in _PROVIDER_REGISTRY:

        class OpenExtractProvider(Provider):
            def run_inference(
                self, pipeline: PipelineSpec, request: InferenceRequest
            ) -> RawInferenceResult:
                from datetime import datetime

                if request.product_type != ProductType.EXTRACT:
                    raise ProviderPermanentError(
                        f"openextract only supports EXTRACT, got {request.product_type}"
                    )
                schema = request.schema_override
                if not schema:
                    raise ProviderPermanentError("schema_override is required for EXTRACT")
                source = Path(request.source_file_path)
                if not source.exists():
                    raise ProviderPermanentError(f"File not found: {source}")

                cfg = self.base_config
                started_at = datetime.now()
                try:
                    extracted, usage, citations = extract_document_with_citations(
                        source,
                        schema,
                        cfg["model"],
                        instructions=cfg.get("instructions", DEFAULT_INSTRUCTIONS),
                        max_retries=int(cfg.get("max_retries", 2)),
                        max_input_bytes=cfg.get("max_input_bytes"),
                        additional_properties_false=bool(
                            cfg.get("additional_properties_false", True)
                        ),
                        cite=bool(cfg.get("cite", True)),
                        timeout=cfg.get("timeout", DEFAULT_CALL_TIMEOUT),
                        window_concurrency=int(
                            cfg.get("window_concurrency", DEFAULT_WINDOW_CONCURRENCY)
                        ),
                    )
                except ProviderNotInstalledError as exc:
                    raise ProviderConfigError(str(exc)) from exc
                except ModelError as exc:
                    if _extractbench_retryable(exc):
                        raise ProviderTransientError(str(exc)) from exc
                    raise ProviderPermanentError(str(exc)) from exc
                except (SchemaValidationError, InputTooLargeError, TypeError) as exc:
                    raise ProviderPermanentError(str(exc)) from exc
                except ExtractionError as exc:
                    raise ProviderPermanentError(str(exc)) from exc

                completed_at = datetime.now()
                in_tok = usage.input_tokens
                out_tok = usage.output_tokens
                cost_usd = in_tok / 1_000_000 * float(
                    cfg.get("input_price_per_1m", 0.0)
                ) + out_tok / 1_000_000 * float(cfg.get("output_price_per_1m", 0.0))
                raw_output = {
                    "data": extracted,
                    "citations": [citation.as_dict() for citation in citations],
                    "field_citations": field_citations_for_extractbench(citations),
                    "model": cfg["model"],
                    "usage": {
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "total_tokens": usage.total_tokens,
                    },
                    "cost_usd": cost_usd,
                }
                return RawInferenceResult(
                    request=request,
                    pipeline=pipeline,
                    pipeline_name=pipeline.pipeline_name,
                    product_type=request.product_type,
                    raw_output=raw_output,
                    started_at=started_at,
                    completed_at=completed_at,
                    latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
                )

            def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
                extracted = raw_result.raw_output.get("data") or {}
                field_citations = raw_result.raw_output.get("field_citations") or []
                return InferenceResult(
                    request=raw_result.request,
                    pipeline_name=raw_result.pipeline_name,
                    product_type=raw_result.product_type,
                    raw_output=raw_result.raw_output,
                    output=ExtractOutput(
                        task_type="extract",
                        example_id=raw_result.request.example_id,
                        pipeline_name=raw_result.pipeline_name,
                        extracted_data=extracted,
                        field_citations=field_citations,
                    ),
                    started_at=raw_result.started_at,
                    completed_at=raw_result.completed_at,
                    latency_in_ms=raw_result.latency_in_ms,
                )

        register_provider("openextract")(OpenExtractProvider)

    try:
        spec = get_pipeline(name)
        spec.config.update(config)
        spec.per_file_timeout = file_timeout
    except ValueError:
        register_pipeline(
            PipelineSpec(
                pipeline_name=name,
                provider_name="openextract",
                product_type=ProductType.EXTRACT,
                config=config,
                per_file_timeout=file_timeout,
            )
        )
    return name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ExtractBench through openextract with any pydantic-ai model.",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="pydantic-ai model id, e.g. openai:gpt-5, xai:grok-4.3, ollama:llama3",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="6-document test split (recommended first run)",
    )
    parser.add_argument("--group", choices=("short", "medium", "long"), help="run one length split")
    parser.add_argument("--file", help="run a single PDF/image instead of the dataset")
    parser.add_argument(
        "--max-concurrent", type=int, default=4, help="parallel documents (default: 4)"
    )
    parser.add_argument("--max-retries", type=int, default=2, help="transient ModelError retries")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_CALL_TIMEOUT,
        help=(
            f"per-window model timeout in seconds (default: {int(DEFAULT_CALL_TIMEOUT)}); "
            "document wall is timeout × (ceil(windows / concurrency) + 2); "
            "ExtractBench per-file timeout is max(1800, that budget for a long doc)"
        ),
    )
    parser.add_argument(
        "--window-concurrency",
        type=int,
        default=DEFAULT_WINDOW_CONCURRENCY,
        help="parallel parse windows per document (default: 4)",
    )
    parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=DEFAULT_MAX_INPUT_BYTES,
        help="per-document byte cap (default: 500 MiB)",
    )
    parser.add_argument("--pipeline-name", help="override the auto-generated pipeline name")
    parser.add_argument(
        "--data-dir", type=Path, help="dataset directory (default: .extractbench/data)"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="results directory (default: .extractbench/output)"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-run inference even if results exist"
    )
    parser.add_argument(
        "--skip-inference", action="store_true", help="re-evaluate existing results only"
    )
    parser.add_argument(
        "--open-report", action="store_true", help="open the HTML report when finished"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-additional-properties-false",
        action="store_true",
        help="do not close object schemas with additionalProperties: false",
    )
    parser.add_argument(
        "--cite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ask the model for per-field citations (default: on; needed for grounding)",
    )
    parser.add_argument(
        "--input-price-per-1m", type=float, default=0.0, help="USD per million input tokens"
    )
    parser.add_argument(
        "--output-price-per-1m", type=float, default=0.0, help="USD per million output tokens"
    )
    parser.add_argument(
        "--install", action="store_true", help="only bootstrap the ExtractBench environment"
    )
    parser.add_argument(
        "--download-only", action="store_true", help="download the dataset and exit"
    )
    parser.add_argument(
        "--serve",
        nargs="?",
        const="",
        metavar="PIPELINE",
        help="serve HTML reports (optionally for a pipeline name)",
    )
    parser.add_argument("--status", action="store_true", help="print dataset download status")
    return parser.parse_args(argv)


def _require_model(args: argparse.Namespace) -> str:
    model = args.model or os.environ.get("OPENEXTRACT_MODEL")
    if not model:
        print(
            "error: --model is required (or set OPENEXTRACT_MODEL). Example: --model openai:gpt-5",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(REPO_ROOT / ".env", override=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ensure_extract_bench()
    if args.install:
        print(f"ExtractBench ready ({CACHE_DIR})")
        return 0

    from extract_bench.cli import BenchCLI

    if args.data_dir is not None:
        args.data_dir = args.data_dir.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.file is not None:
        args.file = str(Path(args.file).expanduser().resolve())

    cli = BenchCLI()
    os.chdir(CACHE_DIR)
    if args.download_only:
        return int(cli.download(data_dir=args.data_dir, test=args.test) or 0)
    if args.status:
        return int(cli.status(data_dir=args.data_dir, test=args.test) or 0)
    if args.serve is not None:
        return int(cli.serve(pipeline=args.serve or None) or 0)

    model = _require_model(args)
    name = register_openextract_pipeline(
        model,
        pipeline_name=args.pipeline_name,
        max_retries=args.max_retries,
        max_input_bytes=args.max_input_bytes,
        additional_properties_false=not args.no_additional_properties_false,
        input_price_per_1m=args.input_price_per_1m,
        output_price_per_1m=args.output_price_per_1m,
        cite=args.cite,
        timeout=args.timeout,
        window_concurrency=args.window_concurrency,
    )
    print(f"Pipeline: {name}")
    print(f"Model:    {model}")
    if args.test:
        print("Split:    test (6 documents)")
    elif args.group:
        print(f"Split:    {args.group}")
    else:
        print("Split:    full ExtractBench (370 documents — this is metered API usage)")

    return int(
        cli.run(
            pipeline=name,
            input_dir=args.data_dir,
            file=args.file,
            output_dir=args.output_dir,
            max_concurrent=args.max_concurrent,
            force=args.force,
            verbose=args.verbose,
            group=args.group,
            open_report=args.open_report,
            skip_inference=args.skip_inference,
            test=args.test,
        )
        or 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
