"""Window progress callbacks on extract, sessions, batch, and CLI."""

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from openextract import (
    AsyncExtractor,
    ExtractProgress,
    Extractor,
    extract,
    extract_async,
    extract_many,
    extract_with_usage,
)
from openextract._cli import _window_progress, main
from tests.pdf_fixture import synthetic_pdf


class Person(BaseModel):
    name: str
    age: int


def _model() -> TestModel:
    return TestModel(
        custom_output_args={
            "output": {"name": "Ada", "age": 36},
            "citations": [{"field": "name", "quote": "Ada", "page": 1}],
        }
    )


def _long_pdf() -> bytes:
    return synthetic_pdf(pages=["AAAA " * 30, "Ada Lovelace " + "BBBB " * 30])


def test_extract_emits_one_event_for_unparsed_input():
    events: list[ExtractProgress] = []
    result = extract(
        Person,
        TestModel(custom_output_args={"name": "Ada", "age": 36}),
        b"Ada is 36",
        media_type="text/plain",
        on_progress=events.append,
    )
    assert result == Person(name="Ada", age=36)
    assert events == [ExtractProgress(current=1, total=1, page=None, pages=())]


def test_extract_emits_per_window_events_for_cited_pdf(monkeypatch):
    monkeypatch.setattr("openextract._parse.DEFAULT_PARSE_WINDOW_CHARS", 40)
    events: list[ExtractProgress] = []
    result = extract(
        Person,
        _model(),
        _long_pdf(),
        media_type="application/pdf",
        cite=True,
        on_progress=events.append,
    )
    assert result == Person(name="Ada", age=36)
    assert len(events) >= 2
    assert [event.current for event in events] == list(range(1, len(events) + 1))
    assert {event.total for event in events} == {len(events)}
    assert all(event.page is not None and event.pages for event in events)
    assert [event.page for event in events] == [event.pages[0] for event in events]


def test_default_path_is_silent(monkeypatch):
    monkeypatch.setattr("openextract._parse.DEFAULT_PARSE_WINDOW_CHARS", 40)
    assert extract(Person, _model(), _long_pdf(), media_type="application/pdf", cite=True) == Person(
        name="Ada", age=36
    )


def test_extract_with_usage_and_batch_forward_progress(monkeypatch):
    monkeypatch.setattr("openextract._parse.DEFAULT_PARSE_WINDOW_CHARS", 40)
    pdf = _long_pdf()
    usage_events: list[ExtractProgress] = []
    output, usage = extract_with_usage(
        Person,
        _model(),
        pdf,
        media_type="application/pdf",
        cite=True,
        on_progress=usage_events.append,
    )
    assert output == Person(name="Ada", age=36)
    assert usage.total_tokens >= 0
    assert len(usage_events) >= 2

    batch_events: list[ExtractProgress] = []
    results = extract_many(
        Person,
        _model(),
        [pdf],
        media_type="application/pdf",
        cite=True,
        on_progress=batch_events.append,
    )
    assert results == [Person(name="Ada", age=36)]
    assert len(batch_events) >= 2


def test_session_constructor_and_per_call_override():
    session_events: list[ExtractProgress] = []
    call_events: list[ExtractProgress] = []
    model = TestModel(custom_output_args={"name": "Ada", "age": 36})
    with Extractor(Person, model, on_progress=session_events.append) as extractor:
        first = extractor.extract(b"one", media_type="text/plain")
        second = extractor.extract(b"two", media_type="text/plain", on_progress=call_events.append)
    assert first == second == Person(name="Ada", age=36)
    assert session_events == [ExtractProgress(current=1, total=1)]
    assert call_events == [ExtractProgress(current=1, total=1)]


async def test_async_extract_and_session_emit_progress():
    events: list[ExtractProgress] = []
    model = TestModel(custom_output_args={"name": "Ada", "age": 36})
    result = await extract_async(
        Person, model, b"Ada is 36", media_type="text/plain", on_progress=events.append
    )
    assert result == Person(name="Ada", age=36)
    assert events == [ExtractProgress(current=1, total=1)]

    session_events: list[ExtractProgress] = []
    async with AsyncExtractor(Person, model) as extractor:
        output = await extractor.extract(
            b"again", media_type="text/plain", on_progress=session_events.append
        )
    assert output == Person(name="Ada", age=36)
    assert session_events == [ExtractProgress(current=1, total=1)]


def test_cli_window_progress_format(capsys):
    _window_progress(ExtractProgress(current=1, total=1))
    _window_progress(ExtractProgress(current=2, total=20, page=2, pages=(2,)))
    _window_progress(ExtractProgress(current=3, total=4, page=5, pages=(5, 6)))
    err = capsys.readouterr().err
    assert "progress: window 1/1\n" in err
    assert "progress: window 2/20 (page 2)\n" in err
    assert "progress: window 3/4 (pages 5-6)\n" in err


def test_cli_single_progress_forwards_callback(mocker, capsys):
    fake = Person(name="Ada", age=36)

    def _extract(**kwargs):
        callback = kwargs["on_progress"]
        callback(ExtractProgress(current=1, total=2, page=1, pages=(1,)))
        callback(ExtractProgress(current=2, total=2, page=2, pages=(2,)))
        return fake

    mocker.patch("openextract._cli.extract", side_effect=lambda **kwargs: _extract(**kwargs))
    exit_code = main(
        [
            "doc.pdf",
            "--schema",
            "tests.test_progress:Person",
            "--model",
            "xai:grok-4.3",
            "--progress",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "progress: window 1/2 (page 1)" in captured.err
    assert "progress: window 2/2 (page 2)" in captured.err
    assert "progress" not in captured.out
