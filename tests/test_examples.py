"""Run the published examples offline; canned responses do not measure model quality."""

import html
import re
import runpy
import sys
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from tests.helpers import output

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/fixtures/invoice.txt"
EXPECTED = {"vendor": "Acme Paper Co.", "total": "120.00", "currency": "USD"}
README_EXAMPLES = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
LANDING_SNIPPET = re.search(
    r'aria-label="Python extraction example"><code>(.*?)</code>',
    (ROOT / "docs/index.html").read_text(),
    re.S,
)[1]
# The landing page wraps tokens in highlighting spans; strip them to get runnable code.
LANDING_EXAMPLE = html.unescape(re.sub(r"<[^>]+>", "", LANDING_SNIPPET))


@pytest.mark.parametrize("entry", ["readme", "landing", "script", "script_path", "script_url"])
def test_first_extraction(provider, entry, monkeypatch, tmp_path, capsys):
    def download(request):
        assert request.url == "https://example.com/invoice.txt"
        assert "authorization" not in request.headers
        return httpx.Response(200, content=FIXTURE.read_bytes())

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    # The script must find its fixture from any working directory.
    monkeypatch.chdir(tmp_path)
    reply = output(value={"total": EXPECTED["total"]} if entry == "landing" else EXPECTED)
    provider.reply(*([download, reply] if entry == "script_url" else [reply]))
    if entry.startswith("script"):
        argv = [str(ROOT / "examples/basic.py")]
        if entry == "script_path":
            (tmp_path / "invoice.txt").write_bytes(FIXTURE.read_bytes())
            argv.append(str(tmp_path / "invoice.txt"))
        elif entry == "script_url":
            argv.append("https://example.com/invoice.txt")
        monkeypatch.setattr(sys, "argv", argv)
        namespace = runpy.run_path(argv[0], run_name="__main__")
    else:
        namespace = {}
        code = README_EXAMPLES[0] if entry == "readme" else LANDING_EXAMPLE
        exec(compile(code, entry, "exec"), namespace)

    invoice = namespace["invoice"]
    assert isinstance(invoice, namespace["Invoice"])
    assert isinstance(invoice.total, Decimal) and invoice.total == Decimal(EXPECTED["total"])
    if entry == "landing":
        source, printed = "Total due: 120.00", "120.00\n"
    else:
        source, printed = FIXTURE.read_text(), "Acme Paper Co.\n120.00\nUSD\n"
        assert (invoice.vendor, invoice.currency) == (EXPECTED["vendor"], EXPECTED["currency"])
    assert provider.requests[-1].url == "https://api.openai.com/v1/responses"
    assert len(provider.bodies) == 1
    assert provider.bodies[0]["model"] == "gpt-6.1-sol"
    assert provider.bodies[0]["input"][0]["content"][-1] == {"type": "input_text", "text": source}
    assert capsys.readouterr().out == printed


@pytest.mark.parametrize("code", README_EXAMPLES)
def test_readme_examples_are_valid_python(code):
    compile(code, "README.md", "exec")
