---
layout: page
title: Live provider smoke tests
---

# Live provider smoke tests

Optional harness for maintainers to verify real provider integration paths.
Default `pytest` / CI runs never make network or model calls.

## Opt-in

```bash
# representative live smoke (requires credentials for the selected model)
OPENEXTRACT_LIVE_SMOKE=1 uv run pytest -m integration tests/test_live_smoke.py -v

# existing example-based live checks
OPENEXTRACT_RUN_EXAMPLES=1 uv run pytest -m integration tests/test_examples.py -v
```

Without these env vars, integration tests skip.

## What is covered initially

| Test | Model (default) | Media | Credential |
| ---- | --------------- | ----- | ---------- |
| `test_live_openai_image_smoke` | `openai:gpt-5` | bundled PNG fixture | `OPENAI_API_KEY` |
| `test_live_openrouter_jev_text_smoke` | `openrouter:~typesafe/jev-latest` | sample memo text | `OPENROUTER_API_KEY` |

Override the model with `OPENEXTRACT_LIVE_MODEL` when needed.

Cookbook live run (same model, no pytest)::

    OPENROUTER_API_KEY=... uv run python -m examples.advanced.openrouter_jev --live

## Design rules

- Marked `@pytest.mark.integration`.
- Explicit env opt-in only.
- Fixtures are small and non-sensitive (`examples/fixtures/`).
- Not enabled in default CI.
- Expand one provider path at a time; promote matrix cells in
  [providers.md](providers.md) from expected → verified when a path is stable.

## Related

- [Provider capability matrix](providers.md)
- [Examples](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/examples/README.md)
