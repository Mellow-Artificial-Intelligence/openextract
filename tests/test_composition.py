"""Offline behavior of application-owned recipes, not provider extraction quality."""

import asyncio
import importlib
import runpy
import sys
import threading
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import BinaryContent, models
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from openextract import ModelError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recipes(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    return importlib.import_module("composition"), importlib.import_module("async_composition")


def local_model(seen, settings=None):
    def respond(messages, info):
        content = next(
            item for item in messages[0].parts[-1].content if isinstance(item, BinaryContent)
        )
        seen.append((content, info.instructions, info.model_settings))
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"vendor": "Acme Paper Co.", "total": "120.00", "currency": "USD"},
                )
            ]
        )

    return FunctionModel(respond, settings=settings)


def test_preprocessing_and_sync_reuse(recipes):
    sync, _ = recipes
    assert sync.prepare_text("Café\r\n$120.00\n") == "Café\n$120.00\n".encode()
    seen = []
    model = local_model(seen, {"max_tokens": 1024})
    for text in ("Seller: Acme Paper Co.\r\nTotal: 120.00 USD", "Another invoice"):
        result = sync.extract_text(text, model)
        assert type(result) is sync.Invoice
        assert result.total == Decimal("120.00")
        content, instructions, settings = seen[-1]
        assert content.data == sync.prepare_text(text)
        assert content.media_type == "text/plain"
        assert instructions == sync.INSTRUCTIONS
        assert settings == {"max_tokens": 1024}
    assert len(seen) == 2


async def test_async_wiring_and_offloaded_read(recipes, monkeypatch, tmp_path):
    sync, asynchronous = recipes
    path = tmp_path / "invoice.txt"
    path.write_bytes(b"Seller: Acme Paper Co.\r\nTotal: 120.00 USD")
    read_threads = []
    read_source = asynchronous.read_source

    def record_read(path):
        read_threads.append(threading.get_ident())
        return read_source(path)

    monkeypatch.setattr(asynchronous, "read_source", record_read)
    seen = []
    results = await asynchronous.extract_files([path], local_model(seen))
    assert type(results[0]) is sync.Invoice
    assert results[0].total == Decimal("120.00")
    assert read_threads and read_threads[0] != threading.get_ident()
    assert seen[0][0].data == sync.prepare_text(path.read_text())
    assert seen[0][0].media_type == "text/plain"
    assert seen[0][1] == sync.INSTRUCTIONS


async def test_bounded_work_and_input_order(recipes, monkeypatch, tmp_path):
    sync, asynchronous = recipes
    paths = [tmp_path / str(i) for i in range(4)]
    for i, path in enumerate(paths):
        path.write_text(str(i))
    gates = [asyncio.Event() for _ in paths]
    started = [asyncio.Event() for _ in range(4)]
    reads = []
    active = peak = count = 0
    read_source = asynchronous.read_source

    def record_read(path):
        reads.append(path)
        return read_source(path)

    async def extract(schema, model, source, **kwargs):
        nonlocal active, peak, count
        index = int(source)
        active += 1
        peak = max(peak, active)
        started[count].set()
        count += 1
        try:
            await gates[index].wait()
            return sync.Invoice(vendor=str(index), total=index, currency=None)
        finally:
            active -= 1

    monkeypatch.setattr(asynchronous, "read_source", record_read)
    monkeypatch.setattr(asynchronous, "extract_async", extract)
    task = asyncio.create_task(asynchronous.extract_files(paths, local_model([]), concurrency=2))
    try:
        await asyncio.wait_for(started[1].wait(), 5)
        assert len(reads) == 2  # The semaphore also bounds reads, not just requests.
        gates[1].set()
        await asyncio.wait_for(started[2].wait(), 5)
        gates[2].set()
        await asyncio.wait_for(started[3].wait(), 5)
        gates[3].set()
        gates[0].set()
        results = await asyncio.wait_for(task, 5)
        assert [result.vendor for result in results] == ["0", "1", "2", "3"]
        assert peak == 2
        assert active == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_expected_failures_are_per_item(recipes, monkeypatch, tmp_path):
    sync, asynchronous = recipes
    paths = [tmp_path / name for name in ("missing", "invalid", "provider", "good")]
    paths[1].write_bytes(b"\xff")
    paths[2].write_text("provider")
    paths[3].write_text("good")

    async def extract(schema, model, source, **kwargs):
        if source == b"provider":
            raise ModelError("offline provider failure")
        return sync.Invoice(vendor="Acme", total=120, currency=None)

    monkeypatch.setattr(asynchronous, "extract_async", extract)
    results = await asynchronous.extract_files(paths, local_model([]))
    assert [type(result) for result in results] == [
        FileNotFoundError,
        UnicodeDecodeError,
        ModelError,
        sync.Invoice,
    ]


@pytest.mark.parametrize("unexpected_error", [False, True], ids=["cancel", "programming-error"])
async def test_cancellation_and_unexpected_error_cleanup(
    recipes, monkeypatch, tmp_path, unexpected_error
):
    _, asynchronous = recipes
    paths = [tmp_path / str(i) for i in range(3)]
    for i, path in enumerate(paths):
        path.write_text(str(i))
    two_started = asyncio.Event()
    fail = asyncio.Event()
    active = started = cancelled = 0

    async def extract(schema, model, source, **kwargs):
        nonlocal active, started, cancelled
        active += 1
        started += 1
        if started == 2:
            two_started.set()
        try:
            if source == b"0" and unexpected_error:
                await fail.wait()
                raise RuntimeError("application bug")
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            active -= 1

    monkeypatch.setattr(asynchronous, "extract_async", extract)
    task = asyncio.create_task(asynchronous.extract_files(paths, local_model([]), concurrency=2))
    try:
        await asyncio.wait_for(two_started.wait(), 5)
        if unexpected_error:
            fail.set()
            with pytest.raises(ExceptionGroup) as exc:
                await asyncio.wait_for(task, 5)
            assert isinstance(exc.value.exceptions[0], RuntimeError)
            assert cancelled >= 1
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert started == cancelled == 2
        assert active == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_empty_inputs_and_invalid_limit(recipes):
    _, asynchronous = recipes
    assert await asynchronous.extract_files([], local_model([])) == []
    for limit in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            await asynchronous.extract_files([], local_model([]), concurrency=limit)


@pytest.mark.parametrize("script", ["composition.py", "async_composition.py"])
def test_script_entry_points(recipes, monkeypatch, tmp_path, capsys, script):
    sync, _ = recipes
    seen = []
    close = AsyncMock()

    def make_model(name, *, settings):
        assert name == "claude-sonnet-4-5"
        assert settings == {"max_tokens": 1024}
        model = local_model(seen, settings)
        model.client = SimpleNamespace(close=close)
        return model

    monkeypatch.setattr(sync, "AnthropicModel", make_model)
    monkeypatch.setattr("pydantic_ai.models.anthropic.AnthropicModel", make_model)
    path = str(ROOT / "examples" / script)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [path])
    runpy.run_path(path, run_name="__main__")
    assert len(seen) == 1
    assert seen[0][1:] == (sync.INSTRUCTIONS, {"max_tokens": 1024})
    assert '"vendor":"Acme Paper Co."' in capsys.readouterr().out
    if script == "async_composition.py":
        assert seen[0][0].data == (ROOT / "examples/fixtures/invoice.txt").read_bytes()
        close.assert_awaited_once()
