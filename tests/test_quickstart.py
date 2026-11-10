"""Execute published quickstarts offline; canned responses do not measure model quality."""

import html
import json
import re
import runpy
import sys
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from tests.helpers import output, serve

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/fixtures/invoice.txt"
LANDING_SOURCE = "Seller: Acme Paper Co.\nTotal due: 120.00 USD\n"
EXPECTED = {"vendor": "Acme Paper Co.", "total": "120.00", "currency": "USD"}


@pytest.mark.parametrize("entry", ["readme", "landing", "script", "script_path"])
def test_first_extraction(entry, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    seen = []

    def respond(request):
        assert request.url == "https://api.openai.com/v1/responses"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=output(value=EXPECTED))

    serve(monkeypatch, respond)
    # The default script must find its fixture even outside the repository root.
    monkeypatch.chdir(tmp_path)
    if entry.startswith("script"):
        argv = [str(ROOT / "examples/basic.py")]
        if entry == "script_path":
            source = tmp_path / "invoice.txt"
            source.write_bytes(FIXTURE.read_bytes())
            argv.append(str(source))
        monkeypatch.setattr(sys, "argv", argv)
        namespace = runpy.run_path(argv[0], run_name="__main__")
    else:
        if entry == "readme":
            code = re.search(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S)[1]
        else:
            page = (ROOT / "docs/index.html").read_text()
            code = html.unescape(
                re.search(r'aria-label="Python extraction example"><code>(.*?)</code>', page, re.S)[
                    1
                ]
            )
        namespace = {}
        exec(compile(code, entry, "exec"), namespace)

    invoice = namespace["invoice"]
    assert isinstance(invoice, namespace["Invoice"])
    assert invoice.vendor == EXPECTED["vendor"]
    assert isinstance(invoice.total, Decimal)
    assert invoice.total == Decimal(EXPECTED["total"])
    assert invoice.currency == EXPECTED["currency"]
    assert len(seen) == 1
    assert seen[0]["model"] == "gpt-5"
    expected_source = LANDING_SOURCE if entry == "landing" else FIXTURE.read_text()
    assert seen[0]["input"][0]["content"][-1] == {"type": "input_text", "text": expected_source}
    assert capsys.readouterr().out == "Acme Paper Co.\n120.00\nUSD\n"
