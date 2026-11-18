"""Sources: text, paths, URLs, and bytes become content with a detected media type."""

import mimetypes
from pathlib import Path

import httpx
import pytest

from openextract import ExtractionError, _content, extract
from tests.helpers import Invoice, configured, output

PDF_BYTES = b"%PDF-1.7 fake"
PDF_FILE = {
    "type": "input_file",
    "filename": "document.pdf",
    "file_data": "data:application/pdf;base64,JVBERi0xLjcgZmFrZQ==",
}


def document(provider):
    """The source as the Responses API received it."""
    return provider.bodies[-1]["input"][0]["content"][-1]


@pytest.mark.parametrize(
    "source",
    [
        "Total due: 120.00",
        "Vendor: Café 東京\nTotal due: 120.00\n",
        "%PDF is literal text here",
        '{"total": 120.00}',
        "x" * 10000,
        "",
    ],
    ids=["plain", "unicode", "signature", "json", "long", "empty"],
)
async def test_strings_are_text(run, provider, source):
    assert await run(Invoice, configured(), source) == Invoice(total=42)
    assert document(provider) == {"type": "input_text", "text": source}


def test_text_defaults_to_plain_text():
    assert _content.load("Total due: 120.00", None).media_type == "text/plain"


async def test_string_never_reads_a_matching_local_file(run, provider, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("invoice.txt").write_text("Private local document")
    await run(Invoice, configured(), "invoice.txt")
    assert document(provider) == {"type": "input_text", "text": "invoice.txt"}


async def test_path_is_read_and_typed_by_extension(run, provider, tmp_path):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(PDF_BYTES)
    await run(Invoice, configured(), path)
    assert document(provider) == PDF_FILE


async def test_missing_path_raises_file_not_found(run):
    with pytest.raises(FileNotFoundError):
        await run(Invoice, configured(), Path("/missing/invoice.pdf"))


def test_office_extensions_do_not_depend_on_the_system_media_type_table(monkeypatch, tmp_path):
    monkeypatch.setattr(mimetypes, "guess_type", lambda name: (None, None))
    for media_type, extension in _content.DOCUMENT_EXTENSIONS.items():
        path = tmp_path / f"File{extension.upper()}"
        path.write_bytes(b"PK\x03\x04\xff")
        assert _content.load(path, None).media_type == media_type


async def test_bytes_are_typed_by_their_leading_bytes(run, provider):
    await run(Invoice, configured(), b"\x89PNG....")
    assert document(provider) == {
        "type": "input_image",
        "image_url": "data:image/png;base64,iVBORy4uLi4=",
    }


@pytest.mark.parametrize(
    "data,media_type",
    [
        (b"%PDF-1.4", "application/pdf"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"GIF89a", "image/gif"),
        (b"ID3\x04", "audio/mpeg"),
        (b"fLaC", "audio/flac"),
        (b"OggS", "audio/ogg"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav"),
        (b"RIFF\x00\x00\x00\x00AVI LIST", None),
        (b"\x00\x00\x00\x18ftypmp42", "video/mp4"),
        (b"plain text, caf\xc3\xa9", "text/plain"),
        (b"\xff\xfe\x00\x01", None),
    ],
)
def test_sniffs_media_type(data, media_type):
    assert _content._sniff(data) == media_type


def test_uninferable_media_type_raises():
    with pytest.raises(ValueError, match="media_type"):
        extract(Invoice, configured(), b"\xff\xfe\x00\x01")


@pytest.mark.parametrize(
    "source,media_type,expected",
    [
        (PDF_BYTES, "text/plain", {"type": "input_text", "text": PDF_BYTES.decode()}),
        (PDF_BYTES.decode(), "application/pdf", PDF_FILE),
    ],
    ids=["bytes", "text"],
)
async def test_explicit_media_type_wins(run, provider, source, media_type, expected):
    await run(Invoice, configured(), source, media_type=media_type)
    assert document(provider) == expected


async def test_url_is_typed_by_content_type_and_fetched_without_credentials(run, provider):
    def page(request):
        assert "authorization" not in request.headers
        headers = {"content-type": "text/html; charset=utf-8"}
        return httpx.Response(200, content=b"<p>hi</p>", headers=headers)

    provider.reply(page, output())
    await run(Invoice, configured(), "https://example.com/page")
    assert document(provider)["text"] == "<p>hi</p>"


@pytest.mark.parametrize(
    "path,content_type,expected",
    [
        ("/invoice.pdf", "text/html", "application/pdf"),
        ("/download", "image/png; charset=binary", "image/png"),
        ("/download", "application/octet-stream", "application/pdf"),
        ("/download", "", "application/pdf"),
    ],
    ids=["extension", "header", "generic-header", "no-header"],
)
async def test_url_is_typed_by_extension_then_header_then_content(
    provider, path, content_type, expected
):
    provider.reply(
        lambda request: httpx.Response(
            200, content=PDF_BYTES, headers={"content-type": content_type}
        )
    )
    url = "https://example.com" + path
    assert _content.load(url, None).media_type == expected
    assert (await _content.load_async(url, None)).media_type == expected


async def test_url_redirects_are_followed(run, provider):
    def redirect(request):
        return httpx.Response(302, headers={"location": "https://example.com/final.txt"})

    def final(request):
        return httpx.Response(200, content=b"Total due: 120.00")

    provider.reply(redirect, final, output())
    await run(Invoice, configured(), "https://example.com/start")
    assert document(provider) == {"type": "input_text", "text": "Total due: 120.00"}


async def test_url_http_error_raises_extraction_error(run, provider):
    provider.reply(None, status=404)
    with pytest.raises(ExtractionError, match="Could not fetch") as error:
        await run(Invoice, configured(), "https://example.com/missing.pdf")
    assert isinstance(error.value.__cause__, httpx.HTTPStatusError)


async def test_url_connection_error_suggests_custom_fetching_without_echoing_the_url(run, provider):
    def unreachable(request):
        raise httpx.ConnectError("private URL: " + str(request.url), request=request)

    provider.reply(unreachable)
    with pytest.raises(ExtractionError) as error:
        await run(Invoice, configured(), "https://example.com/?token=private-token")
    message = str(error.value)
    assert "Could not fetch the source URL (ConnectError)" in message
    assert "Next step:" in message and "pass its bytes" in message
    assert "private-token" not in message
    assert isinstance(error.value.__cause__, httpx.ConnectError)
