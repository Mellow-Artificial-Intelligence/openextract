"""Shared helpers for runnable examples."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DOCUMENT_PAGE = FIXTURES_DIR / "document_page.png"
ACME_SNIPPET = FIXTURES_DIR / "acme_snippet.pdf"

OPENAI_MODEL = "openai:gpt-5.5"
ANTHROPIC_MODEL = "anthropic:claude-opus-4-8"
XAI_MODEL = "xai:grok-4.3"


def _resolve_model(default: str) -> str:
    """Use OPENEXTRACT_MODEL when set; otherwise the example's default provider."""
    return os.environ.get("OPENEXTRACT_MODEL", default)


def openai_model() -> str:
    return _resolve_model(OPENAI_MODEL)


def anthropic_model() -> str:
    return _resolve_model(ANTHROPIC_MODEL)


def xai_model() -> str:
    return _resolve_model(XAI_MODEL)


def require_input(argv: list[str], usage: str) -> str:
    if len(argv) < 2:
        print(usage)
        sys.exit(1)
    return argv[1]


def fixture_path(name: str) -> Path:
    path = FIXTURES_DIR / name
    if not path.is_file():
        print(f"Missing fixture: {path}")
        sys.exit(1)
    return path
