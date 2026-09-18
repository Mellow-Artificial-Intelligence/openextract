"""Tests for SchemaValidationError field-path messages."""

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from openextract import SchemaValidationError
from openextract._errors import (
    _format_validation_issue,
    _map_exception,
    _schema_validation_error,
    _validation_field_path,
)


class _Address(BaseModel):
    zip: int


class _Line(BaseModel):
    qty: int


class _Invoice(BaseModel):
    vendor: str
    lines: list[_Line]
    address: _Address
    note: str = Field(min_length=1)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


def _validate(model: type[BaseModel], payload: dict) -> ValidationError:
    try:
        model.model_validate(payload)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")


def _adapter_error(adapter: TypeAdapter[object], value: object) -> ValidationError:
    try:
        adapter.validate_python(value)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")


def test_nested_field_and_list_index_are_in_the_message():
    exc = _validate(
        _Invoice,
        {"vendor": "Acme", "lines": [{"qty": "x"}], "address": {"zip": "nope"}, "note": "ok"},
    )
    mapped = _map_exception(exc)
    assert isinstance(mapped, SchemaValidationError)
    assert str(mapped) == (
        "Model output did not match schema (2 errors): "
        "lines[0].qty: expected int, got str; address.zip: expected int, got str"
    )
    assert mapped.errors == (
        {"field": "lines[0].qty", "expected": "int", "received": "str"},
        {"field": "address.zip", "expected": "int", "received": "str"},
    )


def test_root_list_index_and_scalar_root():
    mapped_list = _schema_validation_error(_adapter_error(TypeAdapter(list[int]), ["x"]))
    assert str(mapped_list) == "Model output did not match schema: [0]: expected int, got str"
    assert mapped_list.errors == ({"field": "[0]", "expected": "int", "received": "str"},)

    mapped_root = _schema_validation_error(_adapter_error(TypeAdapter(int), "x"))
    assert str(mapped_root) == "Model output did not match schema: <root>: expected int, got str"


def test_missing_and_unexpected_fields():
    missing = _schema_validation_error(_validate(_Invoice, {}))
    assert "vendor: missing required field" in str(missing)
    vendor = {"field": "vendor", "expected": "required field", "received": "missing"}
    assert vendor in missing.errors

    extra = _schema_validation_error(_validate(_Strict, {"name": "Ada", "nope": 1}))
    assert str(extra) == "Model output did not match schema: nope: unexpected field"
    assert extra.errors == (
        {"field": "nope", "expected": "absent", "received": "unexpected field"},
    )


def test_constraint_failures_keep_the_pydantic_message():
    mapped = _schema_validation_error(
        _validate(
            _Invoice,
            {"vendor": "A", "lines": [{"qty": 1}], "address": {"zip": 1}, "note": ""},
        )
    )
    assert str(mapped) == (
        "Model output did not match schema: note: String should have at least 1 character"
    )


def test_empty_validation_error_keeps_the_prefix():
    mapped = _schema_validation_error(ValidationError.from_exception_data("Model", []))
    assert str(mapped) == "Model output did not match schema: 0 validation errors for Model"
    assert mapped.errors == ()


def test_manual_schema_validation_error_has_empty_details():
    exc = SchemaValidationError("already mapped")
    assert str(exc) == "already mapped"
    assert exc.errors == ()


def test_field_path_rendering():
    assert _validation_field_path(()) == "<root>"
    assert _validation_field_path((0,)) == "[0]"
    assert _validation_field_path((0, "name")) == "[0].name"
    assert _validation_field_path(("lines", 0, "qty")) == "lines[0].qty"
    assert _validation_field_path(("address", "zip")) == "address.zip"


def test_issue_formatting_edge_cases():
    missing_input, missing_line = _format_validation_issue({"type": "int_type", "loc": ("age",)})
    assert missing_input == {"field": "age", "expected": "int", "received": "missing"}
    assert missing_line == "age: expected int, got missing"

    none_detail, none_line = _format_validation_issue(
        {"type": "int_type", "loc": ("age",), "input": None}
    )
    assert none_detail == {"field": "age", "expected": "int", "received": "None"}
    assert none_line == "age: expected int, got None"

    fallback_detail, fallback_line = _format_validation_issue({"loc": ("note",)})
    assert fallback_detail == {"field": "note", "expected": "invalid value", "received": "missing"}
    assert fallback_line == "note: invalid value"
