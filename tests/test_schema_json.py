"""Tests for public ``schema_from_json``."""

import json

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from openextract import extract, schema_from_json

_PERSON = {
    "title": "Person",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}


def _write_schema(tmp_path, data=_PERSON, name="person.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestSchemaFromJson:
    def test_path_builds_model(self, tmp_path):
        path = _write_schema(tmp_path)
        cls = schema_from_json(path)
        assert issubclass(cls, BaseModel)
        assert cls.__name__ == "Person"
        assert cls(name="Ada", age=36).name == "Ada"

    def test_str_path_builds_model(self, tmp_path):
        path = _write_schema(tmp_path)
        cls = schema_from_json(str(path))
        assert cls(name="Ada", age=36).age == 36

    def test_dict_builds_model(self):
        cls = schema_from_json(_PERSON)
        assert cls.__name__ == "Person"
        assert cls(name="Ada", age=36).model_dump() == {"name": "Ada", "age": 36}

    def test_json_text_builds_model(self):
        cls = schema_from_json(json.dumps(_PERSON))
        assert cls(name="Ada", age=36).name == "Ada"

    def test_existing_non_json_suffix_file(self, tmp_path):
        path = _write_schema(tmp_path, name="person.schema")
        cls = schema_from_json(str(path))
        assert cls(name="Ada", age=36).name == "Ada"

    def test_title_fallback_is_json_schema(self):
        cls = schema_from_json({"type": "object", "title": "My Schema"})
        assert cls.__name__ == "JsonSchema"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            schema_from_json(tmp_path / "missing.json")

    def test_invalid_json_file_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            schema_from_json(path)

    def test_invalid_json_text_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            schema_from_json("{")

    def test_non_object_dict_raises(self):
        with pytest.raises(ValueError, match="JSON Schema object"):
            schema_from_json({"type": "string"})

    def test_non_object_text_raises(self):
        with pytest.raises(ValueError, match="JSON Schema object"):
            schema_from_json("[]")

    def test_non_object_file_raises(self, tmp_path):
        path = _write_schema(tmp_path, ["not", "an", "object"])
        with pytest.raises(ValueError, match="JSON Schema object"):
            schema_from_json(path)

    def test_unreadable_file_raises(self, tmp_path, mocker):
        path = _write_schema(tmp_path)
        mocker.patch(
            "pathlib.Path.read_text",
            side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad"),
        )
        with pytest.raises(ValueError, match="cannot read schema file"):
            schema_from_json(path)

    def test_invalid_source_type_raises(self):
        with pytest.raises(TypeError, match="expected a path"):
            schema_from_json(42)  # type: ignore[arg-type]

    def test_extract_round_trip(self):
        schema = schema_from_json(_PERSON)
        result = extract(
            schema=schema,
            model=TestModel(custom_output_args={"name": "Ada", "age": 36}),
            input_file=b"Ada Lovelace, 36",
            media_type="text/plain",
        )
        assert result.name == "Ada"
        assert result.age == 36
