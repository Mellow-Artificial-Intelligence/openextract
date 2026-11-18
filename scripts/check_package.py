"""CI check of an installed distribution: footprint limits and an offline first extraction."""

import asyncio
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

# Bytes, about 25% above the 2.1.0 baseline on CPython 3.12 for Linux x86_64.
LIMITS = {"wheel_bytes": 40_000, "package_bytes": 110_000, "runtime_bytes": 11_500_000}
# Exact names, so an added or replaced dependency needs an explicit review.
DEPENDENCIES = {
    "direct_dependencies": ["httpx", "pydantic"],
    "transitive_dependencies": [
        "annotated-types",
        "anyio",
        "certifi",
        "h11",
        "httpcore",
        "idna",
        "pydantic-core",
        "typing-extensions",
        "typing-inspection",
    ],
}
# Installer bookkeeping varies with paths and tools; bytecode varies with Python versions.
INSTALLER_FILES = {
    "RECORD",
    "direct_url.json",
    "INSTALLER",
    "REQUESTED",
    "uv_cache.json",
    "uv_build.json",
}


def installed_bytes(dist: importlib.metadata.Distribution) -> int:
    total = 0
    for file in dist.files or []:
        if file.suffix == ".pyc" or file.name in INSTALLER_FILES:
            continue
        located = Path(dist.locate_file(file))
        if located.is_file():
            total += located.stat().st_size
    return total


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def measure(wheel: Path) -> dict:
    distributions = {
        normalize(dist.metadata["Name"]): dist for dist in importlib.metadata.distributions()
    }
    package = distributions.pop("openextract")
    # Count every declared requirement, including optional ones.
    direct = sorted({normalize(re.match(r"[\w.-]+", req)[0]) for req in package.requires or []})
    package_bytes = installed_bytes(package)
    return {
        "wheel_bytes": wheel.stat().st_size,
        "package_bytes": package_bytes,
        "runtime_bytes": package_bytes + sum(map(installed_bytes, distributions.values())),
        "direct_dependencies": direct,
        "transitive_dependencies": sorted(set(distributions) - set(direct)),
        "installed_versions": {name: dist.version for name, dist in sorted(distributions.items())},
    }


def violations(report: dict) -> list[str]:
    sizes = [
        f"{metric}: {report[metric]:,} exceeds {limit:,} bytes"
        for metric, limit in LIMITS.items()
        if report[metric] > limit
    ]
    names = [
        f"{kind}: expected {expected}, got {report[kind]}"
        for kind, expected in DEPENDENCIES.items()
        if report[kind] != expected
    ]
    return sizes + names


def offline_smoke() -> None:
    """Import the package and extract once in each mode with networking blocked and no keys."""

    def deny_network(event: str, args: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            raise RuntimeError(f"Network access forbidden during smoke test: {event}")

    sys.addaudithook(deny_network)
    import httpx
    from pydantic import BaseModel

    from openextract import Model, extract, extract_async

    class Invoice(BaseModel):
        total: float

    text = {"type": "output_text", "text": '{"total": 42}'}
    payload = {"status": "completed", "output": [{"type": "message", "content": [text]}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    model = Model("openai:test", api_key="test")

    async def extract_asynchronously() -> Invoice:
        async with httpx.AsyncClient(transport=transport) as client:
            return await extract_async(Invoice, model, b"text", client=client)

    with httpx.Client(transport=transport) as client:
        assert extract(Invoice, model, b"text", client=client) == Invoice(total=42)
    assert asyncio.run(extract_asynchronously()) == Invoice(total=42)


def main() -> None:
    wheel, distribution = sys.argv[1:3]
    report = measure(Path(wheel))
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as output:
            output.write(f"\n### Package footprint ({distribution})\n```json\n{rendered}\n```\n")
    if errors := violations(report):
        raise SystemExit("\n".join(errors))
    offline_smoke()
    print("Offline sync/async first-use smoke passed")


if __name__ == "__main__":
    main()
