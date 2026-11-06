"""Execute published quickstarts offline; canned responses do not measure model quality."""

import html
import re
import runpy
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic_ai import BinaryContent, models
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

import openextract._extract as extract_module

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/fixtures/invoice.txt"
EXPECTED = {"vendor": "Acme Paper Co.", "total": "120.00", "currency": "USD"}


@pytest.mark.parametrize("entry", ["readme", "landing", "script", "script_path"])
def test_first_extraction(entry, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = []

    def respond(messages, info):
        content = messages[0].parts[-1].content
        seen.extend(item for item in content if isinstance(item, BinaryContent))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, EXPECTED)])

    build_agent = extract_module._build_agent

    def offline_agent(schema, model, instructions):
        assert model == "openai:gpt-5"
        assert instructions is None
        return build_agent(schema, FunctionModel(respond), instructions)

    monkeypatch.setattr(extract_module, "_build_agent", offline_agent)
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
    assert seen[0].data == FIXTURE.read_bytes()
    assert seen[0].media_type == "text/plain"
    assert capsys.readouterr().out == "Acme Paper Co.\n120.00\nUSD\n"
