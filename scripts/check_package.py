"""CI-only installed-distribution footprint and offline first-use check (stdlib only)."""

import asyncio
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

# Installer bookkeeping varies with paths/tools; bytecode varies with Python versions.
INSTALLER_FILES = {
    "RECORD",
    "direct_url.json",
    "INSTALLER",
    "REQUESTED",
    "uv_cache.json",
    "uv_build.json",
}


def installed_bytes(dist: importlib.metadata.Distribution) -> int:
    return sum(
        dist.locate_file(file).stat().st_size
        for file in dist.files or []
        if file.suffix != ".pyc"
        and file.name not in INSTALLER_FILES
        and dist.locate_file(file).is_file()
    )


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def measure(wheel: Path) -> dict:
    distributions = {
        normalize(dist.metadata["Name"]): dist for dist in importlib.metadata.distributions()
    }
    package = distributions.pop("openextract")
    # Count all declared requirements, including optional ones, conservatively.
    direct = sorted({normalize(re.match(r"[\w.-]+", req)[0]) for req in package.requires or []})
    return {
        "wheel_bytes": wheel.stat().st_size,
        "package_bytes": installed_bytes(package),
        "runtime_bytes": installed_bytes(package)
        + sum(installed_bytes(dist) for dist in distributions.values()),
        "direct_dependencies": direct,
        "transitive_dependencies": sorted(set(distributions) - set(direct)),
        "installed_versions": {name: dist.version for name, dist in sorted(distributions.items())},
    }


def violations(report: dict, budget: dict) -> list[str]:
    errors = []
    for key, maximum in budget["limits"].items():
        if report[key] > maximum:
            errors.append(f"{key}: {report[key]:,} exceeds {maximum:,} bytes")
    for key in ("direct_dependencies", "transitive_dependencies"):
        if report[key] != budget["baseline"][key]:
            errors.append(f"{key}: expected {budget['baseline'][key]}, got {report[key]}")
    return errors


def offline_smoke() -> None:
    def deny_network(event: str, args: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            raise RuntimeError(f"Network access forbidden during smoke test: {event}")

    sys.addaudithook(deny_network)
    # Import and first extraction both run with networking blocked and no provider keys.
    import httpx
    from pydantic import BaseModel

    from openextract import Model, extract, extract_async

    class Invoice(BaseModel):
        total: float

    payload = {
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": '{"total": 42}'}]}
        ],
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    model = Model("openai:test", api_key="test")
    with httpx.Client(transport=transport) as client:
        result = extract(Invoice, model, b"text", client=client)
    assert isinstance(result, Invoice) and result.total == 42

    async def check_async() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            result = await extract_async(Invoice, model, b"text", client=client)
        assert isinstance(result, Invoice) and result.total == 42

    asyncio.run(check_async())


def main() -> None:
    wheel = Path(sys.argv[1])
    budget = json.loads(Path(__file__).with_name("package_budget.json").read_text())
    report = measure(wheel)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as output:
            output.write(f"\n### Package footprint ({sys.argv[2]})\n```json\n{rendered}\n```\n")
    errors = violations(report, budget)
    if errors:
        raise SystemExit("\n".join(errors))
    offline_smoke()
    print("Offline sync/async first-use smoke passed")


if __name__ == "__main__":
    main()
