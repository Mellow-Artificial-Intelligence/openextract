"""Bound file reading and extraction in the caller's application.

After the setup in examples/composition.md:
    uv run --no-sync python examples/async_composition.py
    uv run --no-sync python examples/async_composition.py invoice1.txt invoice2.txt
"""

import asyncio
import sys
from pathlib import Path

from composition import INSTRUCTIONS, Invoice, make_model, prepare_text
from pydantic_ai.models import Model

from openextract import ExtractionError, extract_async


def read_source(path: Path) -> bytes:
    return prepare_text(path.read_text(encoding="utf-8"))


async def extract_files(
    paths: list[Path], model: Model, *, concurrency: int = 2
) -> list[Invoice | Exception]:
    """Keep input order; return expected per-file failures alongside schema instances."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(path: Path) -> Invoice | Exception:
        async with semaphore:
            try:
                # extract_async reads local paths synchronously. Preload bytes off-loop.
                source = await asyncio.to_thread(read_source, path)
                return await extract_async(
                    Invoice, model, source, instructions=INSTRUCTIONS, media_type="text/plain"
                )
            except (OSError, UnicodeError, ExtractionError) as exc:
                return exc

    # Unexpected errors cancel sibling tasks; caller cancellation propagates.
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(path)) for path in paths]
    return [task.result() for task in tasks]


async def main(paths: list[Path]) -> None:
    model = make_model()
    try:
        results = await extract_files(paths, model)
        for path, result in zip(paths, results, strict=True):
            if isinstance(result, Exception):
                print(f"{path.name}: {type(result).__name__}: {result}")
            else:
                print(f"{path.name}: {result.model_dump_json()}")
    finally:
        await model.client.close()


if __name__ == "__main__":
    paths = [Path(arg) for arg in sys.argv[1:]] or [Path(__file__).parent / "fixtures/invoice.txt"]
    asyncio.run(main(paths))
