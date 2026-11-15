# Typed extraction from Office files with Gemini

Use a supported Gemini model and set `GOOGLE_API_KEY` or `GEMINI_API_KEY`:

```python
from pathlib import Path

from pydantic import BaseModel
from openextract import extract


class Invoice(BaseModel):
    vendor: str
    total: float


invoice = extract(Invoice, "google:gemini-4-hydrogen", Path("invoice.xlsx"))
```

`extract_async` accepts the same inputs. Files and downloaded URLs are sent as
inline base64 bytes to `generateContent`; no SDK or Office parser is installed.
For raw bytes, pass `media_type` explicitly (for XLSX,
`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`).

Google's [November 5, 2026 release notes](https://aistudio.google.com/docs/changelog#11-05-2026)
expanded Office support. The [GenerateContent document guide](https://aistudio.google.com/docs/document-processing?generate-content=true#office-documents),
checked November 15, lists DOCX, DOC, DOTX, ODT, PPTX, PPT, ODP, XLSX, XLS, and ODS.

Spreadsheets expose only the first sheet: XLSX/XLS provide cell values without
formulas or formatting; ODS includes formulas. Export the intended sheet separately
when needed. DOC/PPT/ODP are rendered as images; other document formats provide
text, with embedded images for DOCX/PPTX. Layout and charts may be lost. Provider
file-size, page, and context limits apply; OpenExtract does not split files.

The original Pydantic schema validates the result. Offline tests verify request
encoding, not live extraction accuracy. Review important values against the source.
Gemini Decisions retains its separate media restrictions.
