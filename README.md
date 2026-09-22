<div align="center">

# openextract

**Extract structured data from documents, images, audio, and video using LLMs.**

[![PyPI version](https://img.shields.io/pypi/v/openextract.svg?logo=pypi&logoColor=white&color=4B8BBE)](https://pypi.org/project/openextract/)
[![Python versions](https://img.shields.io/pypi/pyversions/openextract.svg?logo=python&logoColor=white)](https://pypi.org/project/openextract/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)

[Website](https://mellow-artificial-intelligence.github.io/openextract/) &middot; [PyPI](https://pypi.org/project/openextract/) &middot; [Changelog](CHANGELOG.md)

</div>

One function: give it a Pydantic schema, a model, and a file. You get back a validated instance.

```python
from pydantic import BaseModel
from openextract import extract


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str


invoice = extract(Invoice, "openai:gpt-5", source="./invoice.pdf")  # path to a file on disk
print(invoice.total)
```

## Install

```bash
pip install 'openextract[all]'   # or just your provider: [openai], [anthropic], [google], ...
```

Requires Python 3.12+. Set the API key your provider expects, such as `OPENAI_API_KEY`.

## Usage

```python
extract(schema, model, source, instructions=None, *, media_type=None)
await extract_async(schema, model, source, instructions=None, *, media_type=None)
```

- **`source`** is where the file is: a path on disk, a `pathlib.Path`, or an `http(s)://` URL. openextract opens it for you. You can also pass the file's raw `bytes`. A string is always treated as a path or URL, never as the document's text; to extract from text you already have, pass `text.encode()`.
- **`model`** is any [pydantic-ai](https://ai.pydantic.dev/models/) model string (`"anthropic:claude-sonnet-5"`, `"google-gla:gemini-2.5-pro"`, `"ollama:llama3"`, ...) or a configured pydantic-ai `Model`.
- **`instructions`** are optional extra guidance for the model.
- **`media_type`** is inferred for you. openextract checks the file extension, then the URL's `Content-Type`, then the file's leading bytes (PDF, PNG, JPEG, GIF, WebP, MP3, WAV, FLAC, OGG, MP4, UTF-8 text). Pass it only to override the result.

```python
from pathlib import Path

extract(Invoice, "openai:gpt-5", source="./invoices/2291.pdf")                    # file path
extract(Invoice, "openai:gpt-5", source=Path("invoices") / "2291.pdf")            # pathlib.Path
extract(Invoice, "anthropic:claude-sonnet-5", source="https://example.com/2291.pdf")  # URL
extract(Invoice, "openai:gpt-5", source=pdf_bytes, instructions="Amounts are in EUR.")  # bytes
```

## Errors

Every error subclasses `ExtractionError` and keeps the original exception as `__cause__`.

| Exception | Raised when |
| --- | --- |
| `ModelError` | The provider returns an error. |
| `SchemaValidationError` | The model's output doesn't match the schema. |
| `ProviderNotInstalledError` | The provider's SDK isn't installed. |
| `ExtractionError` | A URL can't be fetched. |

If openextract can't infer the media type, it raises a `ValueError` and asks you to pass `media_type`.

## License

MIT
