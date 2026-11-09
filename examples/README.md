# First extraction example

The [README quickstart](../README.md#first-extraction) includes the whole fictional
invoice as text bytes. The [landing page](../docs/index.html) uses the same source,
schema, model, and typed field access. Neither requires downloading a document.

For the checked-in file example, from the repository root:

```bash
export OPENAI_API_KEY='your-api-key'  # macOS / Linux
uv run --extra openai python examples/basic.py
```

On PowerShell, set `$env:OPENAI_API_KEY = 'your-api-key'` instead. This requires
Python 3.12+, uv, an OpenAI API key with access to `gpt-5`, and makes a paid request.
The script locates `fixtures/invoice.txt` relative to itself. You can also supply
a path or URL as the first argument; supported formats depend on the model:

```bash
uv run --extra openai python examples/basic.py examples/fixtures/invoice.txt
```

All names and amounts in the fixture are fictional. The expected fields are
`vendor="Acme Paper Co."` (seller, not the buyer Northwind Studio),
`total=Decimal("120.00")` (final amount including tax, not the 100.00 subtotal),
and `currency="USD"`. The schema permits `None` for unclear currency.
Continue with [fine-detail controls](fine_details.md) to tailor these field
meanings, validate currency, guide interpretation, and override media detection.

For generation behavior, see [configured models](configured_models.md) and the
[dependency compatibility record](model_compatibility.md).

## Offline checks

```bash
uv sync --dev --frozen
uv run --no-sync pytest tests/test_quickstart.py
```

These tests execute the README and landing-page snippets and the file example,
including its optional path argument. They use the real extraction function,
source loading, media inference, and Pydantic validation with a local
`FunctionModel` returning canned expected fields. They check the selected model
identifier, fixture bytes, typed fields, and printed output. Provider requests
are disabled, and no API key is needed. These checks run in ordinary CI.

This verifies example wiring, not the provider's ability to extract accurate
values. Schema validation does not guarantee factual accuracy.

## Optional live quality check

No live quality check was run for this change (2026-11-06). OpenAI / `openai:gpt-5`
is the documented target, not a recorded successful provider run. The expected
values above were read from the fictional fixture, not observed model output.

To check quality manually, run the file example with your credentials and compare
all three fields with the fixture. Record the date, provider, exact model, and
observed fields when reporting a result. Keep live requests out of ordinary CI.
