"""Fail when documented public call signatures drift from the package."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import openextract

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "api-reference.md"
PUBLIC_FUNCTIONS = (
    "extract",
    "extract_async",
    "extract_with_usage",
    "extract_with_usage_async",
    "extract_with_result",
    "extract_with_result_async",
    "extract_many",
    "extract_many_async",
    "iter_extract_many_async",
    "extract_many_with_results",
    "extract_many_with_results_async",
    "total_usage",
    "normalize_reduce",
    "reduce_outputs",
    "extract_swarm",
    "extract_swarm_async",
    "extract_swarm_with_results",
    "extract_swarm_with_results_async",
    "resolve_swarm_members",
    "define_agent",
    "define_remote_agent",
    "flatten_agent",
    "resolve_output_schema",
    "load_agent",
    "load_agents",
    "load_agent_directory",
)
SIGNATURE_HEADING = re.compile(r"^### `([a-z_]+)(\(.*\))`$", re.MULTILINE)


def _call_signature(function) -> str:
    """Return a readable call signature without implementation annotations."""
    signature = inspect.signature(function)
    parameters = [
        parameter.replace(annotation=inspect.Parameter.empty)
        for parameter in signature.parameters.values()
    ]
    return str(
        signature.replace(
            parameters=parameters,
            return_annotation=inspect.Signature.empty,
        )
    )


def main() -> int:
    documented = dict(SIGNATURE_HEADING.findall(REFERENCE.read_text()))
    expected_names = set(PUBLIC_FUNCTIONS)
    if set(documented) != expected_names:
        missing = sorted(expected_names - set(documented))
        extra = sorted(set(documented) - expected_names)
        print(f"API reference function set drifted; missing={missing}, extra={extra}")
        return 1

    mismatches = []
    for name in PUBLIC_FUNCTIONS:
        actual = _call_signature(getattr(openextract, name))
        if documented[name] != actual:
            mismatches.append(f"{name}: documented {documented[name]}, actual {actual}")

    if mismatches:
        print("API reference signatures are stale:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1

    print("API reference signatures match the installed package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
