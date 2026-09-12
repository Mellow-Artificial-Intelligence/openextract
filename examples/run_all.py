#!/usr/bin/env python3
"""Run all examples that work without extra user-provided files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EXAMPLES_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLES_DIR.parent

# Scripts that run with bundled fixtures only (no API key needed for error_handling).
API_EXAMPLES: list[tuple[str, list[str]]] = [
    ("examples.basic.local_file", ["--fixture"]),
    ("examples.basic.bytes_input", []),
    ("examples.basic.url_extract", []),
    ("examples.images.document_summary", []),
    ("examples.images.receipt_extraction", ["--fixture"]),
    ("examples.documents.invoice_extraction", ["--fixture"]),
    ("examples.batch.batch_extract", []),
    ("examples.async.async_extract", []),
    ("examples.advanced.extract_with_usage", ["--fixture"]),
    ("examples.advanced.retry_extract", []),
]

NO_API_EXAMPLES: list[str] = [
    "examples.advanced.error_handling",
    "examples.advanced.reusable_sessions",
    "examples.advanced.extraction_styles",
    "examples.advanced.extract_with_citations",
    "examples.batch.stream_batch_extract",
]


def run_module(module: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module, *(extra_args or [])]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    print("Running examples that do not require a model API key...")
    for module in NO_API_EXAMPLES:
        run_module(module)

    print(
        "\nRunning examples that call a model "
        "(OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY — or OPENEXTRACT_MODEL to override all)..."
    )
    for module, args in API_EXAMPLES:
        run_module(module, args)

    print("\nAll runnable examples completed successfully.")


if __name__ == "__main__":
    main()
