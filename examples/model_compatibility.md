# Configured-model compatibility record

Investigation for [#273](https://github.com/Mellow-Artificial-Intelligence/openextract/issues/273),
2026-11-09. The [configured-model guide](configured_models.md) uses the repository's
locked Pydantic AI / Anthropic SDK pair. Neither package constraints nor `uv.lock`
were changed by this investigation.

## What was checked

`tests/test_configured_models.py` has four offline cases: sync/async requests with
a local `FunctionModel` recording model-default settings, and both variants of
the published Anthropic example. The latter constructs the real provider and
model, substitutes the SDK's async message-creation method, and records
`max_tokens=1024` at that boundary. HTTP sends are blocked. This verifies adapter
propagation up to the SDK call, not wire serialization, service acceptance, model
availability, or extraction accuracy.

All runs used Python 3.12.14. Temporary environments resolved dependencies with
the repository's `2026-10-14T00:00:00Z` embargo; project dependencies were untouched.
The temporary runs used Pydantic 2.14.1 and pytest 9.1.1.

| Pydantic AI | Anthropic SDK | Local defaults (sync/async) | Published provider example (sync/async) |
| --- | --- | --- | --- |
| 1.37.0 (declared minimum) | 0.75.0 (its extra's minimum) | Pass | Pass |
| 1.37.0 | 1.13.0 | Pass | Fail: adapter import |
| 2.0.0 (major-version boundary) | 0.108.0 (its extra's minimum) | Pass | Pass |
| 2.0.0 | 1.13.0 | Pass | Fail: provider construction |
| 2.56.0 (repository lock) | 1.13.0 (repository lock) | Pass | Pass |

The two failing combinations satisfy the packages' declared version constraints:

- With 1.37.0 / 1.13.0, the Anthropic adapter imports `UserLocation` from
  `anthropic.types.beta.beta_web_search_tool_20250305_param`, where the SDK no
  longer exports it. The adapter wraps this in an installation-hint `ImportError`,
  even though the SDK is installed.
- With 2.0.0 / 1.13.0, provider construction passes an `httpx.AsyncClient` to the
  SDK, which requires `httpx2.AsyncClient` and raises `TypeError`.

These are provider/SDK incompatibilities before extraction, not failures to pass
model defaults. No new OpenExtract settings argument would repair them. The
base-model settings hook exists at 1.37.0 and worked at all three sampled Pydantic
AI versions. This does not establish every intermediate version or every other
provider's behavior; the full suite runs against the repository lock in CI.

## Reproduce without changing the project environment

From the repository root, use a disposable environment (choose a fresh path):

```bash
uv venv /tmp/openextract-settings-check
uv pip install --python /tmp/openextract-settings-check/bin/python \
  --exclude-newer 2026-10-14T00:00:00Z \
  'pydantic-ai-slim[anthropic]==1.37.0' 'anthropic==0.75.0' \
  'pytest==9.1.1' 'pytest-asyncio==1.4.0'
PYTHONPATH=src /tmp/openextract-settings-check/bin/python -m pytest \
  tests/test_configured_models.py -q
```

Repeat in a fresh environment for each table row, substituting both version pins.
Passing rows produce four passing tests; failing rows produce two passes and two
failures. Transitive packages may resolve differently if this is rerun later;
retain the resolved versions when investigating a new result. These commands do
not modify the project's dependency declarations or lockfile.

## Separate compatibility remedy proposal — not applied

The current `pydantic-ai-slim>=1.37.0,<3` range does not guarantee the documented
Anthropic recipe works with every permitted SDK. The example therefore pins a
verified pair. This is a documentation constraint, not a package-wide fix.

Propose a separate dependency-compatibility PR that establishes the earliest
Pydantic AI release compatible with the locked Anthropic SDK, then narrows the
supported Pydantic AI floor consistently across the core and provider extras.
Use 2.56.0 / 1.13.0 as a known-working upper bound for that investigation, not an
unverified claim about the minimum fix version. Before adopting a new floor,
check all existing provider imports, run the full suite and package smoke tests,
and add a small oldest-supported/locked compatibility check for this recipe.
Keep the embargo and review the resulting lockfile changes explicitly.

That proposal would remove support for some currently allowed dependency
versions and needs a separate compatibility decision. A blanket Anthropic SDK
cap is not justified by these samples and could conflict with newer adapters.
No upgrade, new public option, provider shim, or dependency-policy change is
included here. Other providers and the exact minimum fix version remain to be
verified before applying a remedy.
