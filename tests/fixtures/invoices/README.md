# Invoice fixtures

These frozen synthetic inputs exercise document and image extraction without
using customer data. Every input has an expected invoice and a SHA-256 checksum
in `manifest.json`. Paths in the manifest are relative to this directory.

| Test ID | File | Pages | Content |
| --- | --- | --- | --- |
| `digital_pdf` | `invoice.pdf` | 1 | Text-based invoice |
| `png` | `invoice.png` | 1 | 150-DPI rendering of the single-page invoice |
| `jpeg` | `invoice.jpg` | 1 | JPEG version of the same rendering |
| `multipage_pdf` | `invoice-multipage.pdf` | 3 | Nine line items split across pages |
| `scanned_pdf` | `invoice-single-scanned.pdf` | 1 | Image-only version of the single-page invoice |
| `scanned_multipage_pdf` | `invoice-multipage-scanned.pdf` | 3 | Image-only, 150-DPI version of the multi-page invoice |
| `degraded_scanned_pdf` | `invoice-multipage-scanned-degraded.pdf` | 3 | Lower-resolution grayscale scan with skew and compression |

`invoice.json` describes the single-page invoice: two line items, totaling USD
42.50. `invoice-multipage.json` describes the three-page invoice: nine line items,
totaling USD 108.50. Only the final page of the multi-page invoice contains the
grand total and paid status; each page has its own subtotal.

The PDFs were created with ReportLab and rendered with Poppler. Raster-only PDFs
were assembled from page images, with no OCR layer. Before saving these snapshots,
pypdf verified their page counts, zero extracted text, and one embedded image per
scanned page. The manifest records those properties. The degraded scan uses
110 DPI, grayscale, rotations of 0.8/-0.6/0.7 degrees, a 0.25-pixel blur, and JPEG
quality 65. The single-page JPEG uses quality 92.

Running tests requires no rendering, OCR, or PDF-generation dependencies. Offline
tests check the frozen bytes against the manifest and validate expected invoice
data and totals. Live tests submit these files to providers and compare every
returned field, including line-item order, with the expected invoice.

When replacing a fixture, visually review every page and verify its structural
properties again, then update its checksum, metadata, and expected invoice as
needed. Keep the scanned fixtures image-only and preserve the cross-page checks.
