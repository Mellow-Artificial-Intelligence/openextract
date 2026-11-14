# Scanned PDFs with Mistral document annotations

Install `openextract` as usual (`pip install openextract`) and set
`MISTRAL_API_KEY`. No Mistral SDK or local PDF/OCR dependencies are needed.
Select the OCR API explicitly through the existing `Model` configuration:

```python
from pathlib import Path

from pydantic import BaseModel, Field
from openextract import Model, extract, extract_async


class Invoice(BaseModel):
    vendor: str
    total: float = Field(description="Final amount due, including tax")


model = Model("mistral:mistral-ocr-latest", api="mistral-ocr")
invoice = extract(Invoice, model, Path("scanned-invoice.pdf"))
print(invoice.total)

# Inside an async function:
# invoice = await extract_async(Invoice, model, pdf_bytes, media_type="application/pdf")
```

Both calls return an `Invoice`. Paths, PDF bytes, and HTTP(S) URLs use the existing
source loader. URLs are downloaded locally before their bytes are sent as a PDF
data URI; they are not forwarded to Mistral. Use `media_type="application/pdf"`
when detection needs an override. This adapter accepts only PDFs, including scans.
Ordinary `mistral:...` model strings still select Chat Completions, even if their
model name contains `ocr`.

The adapter sends the Pydantic JSON Schema as `document_annotation_format` to
[`POST /v1/ocr`](https://docs.mistral.ai/api/endpoint/ocr) using direct HTTPX.
`instructions` become the document annotation prompt. The returned JSON string
is validated with the original schema, including its validators. Existing model
connection options, HTTP retries, and validation retries apply. Validation retries
resubmit the PDF and can incur another provider charge. Missing or invalid
annotations raise `SchemaValidationError`; provider failures raise `ModelError`.

Provider-native options such as `settings={"pages": [0, 1]}` can explicitly select
zero-based pages. The default submits the complete PDF without silently selecting
or truncating pages. Request fields, schema, prompt, bbox annotations, and image
return settings cannot be overridden. This interface returns the typed document
annotation only; it does not expose OCR markdown, images, or bbox captions.

## Limits and visual accuracy

Verified against Mistral's documentation on November 14, 2026:

- [`mistral-ocr-latest` currently points to OCR 4.1](https://docs.mistral.ai/resources/changelogs)
  (`mistral-ocr-4-1`). Use that explicit model ID if you need to pin the provider
  model. Model availability and service limits can change.
- The [Document AI limits](https://docs.mistral.ai/studio/document-processing/annotations)
  are 50 MB and 1,000 pages per document. Mistral enforces these limits;
  OpenExtract does not parse PDFs to count pages or split documents.
- Mistral removed the old eight-page annotation limit in its January 27, 2026
  [update](https://docs.mistral.ai/resources/changelogs). Older cookbooks still
  show that limit; it is not imposed here.
- Document annotations consume OCR markdown plus only the **first eight extracted
  image bounding boxes**, according to the
  [annotation documentation](https://docs.mistral.ai/studio/document-processing/annotations).
  They do not inspect every page as a full-page image. Prefer text-focused scans;
  charts, signatures, stamps, layout, faint text, and images beyond those eight
  boxes may be missed or misinterpreted. Pydantic validation checks structure and
  constraints, not factual accuracy. Review important results against the source.

No local rendering, PDF parsing, bbox captioning, document storage, or provider
SDK is included. Offline tests verify HTTP and validation contracts, not live OCR
quality.
