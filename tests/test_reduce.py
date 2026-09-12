"""Tests for openextract._reduce."""

import pytest
from pydantic import BaseModel

from openextract import (
    Citation,
    SchemaValidationError,
    SwarmReduce,
    normalize_reduce,
    reduce_outputs,
)
from openextract._reduce import merge_values, reduce_citations, vote_values


class Person(BaseModel):
    name: str
    age: int | None = None


class Invoices(BaseModel):
    vendor: str | None = None
    lines: list[str] = []


class TestNormalizeReduce:
    def test_defaults_to_merge(self):
        assert normalize_reduce() is SwarmReduce.MERGE

    @pytest.mark.parametrize("value", ["merge", "vote", "first"])
    def test_accepts_every_strategy_name(self, value):
        assert normalize_reduce(value) is SwarmReduce(value)

    def test_passes_through_enum_members(self):
        assert normalize_reduce(SwarmReduce.VOTE) is SwarmReduce.VOTE

    def test_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="reduce must be one of"):
            normalize_reduce("average")

    def test_rejects_non_string_values(self):
        with pytest.raises(ValueError, match="reduce must be one of"):
            normalize_reduce(3)


class TestMergeValues:
    def test_empty_input_is_none(self):
        assert merge_values([]) is None

    def test_single_value_passes_through(self):
        assert merge_values([{"a": 1}]) == {"a": 1}

    def test_lists_union_and_drop_duplicates(self):
        assert merge_values([[1, 2], [2, 3]]) == [1, 2, 3]

    def test_list_duplicates_compare_structurally(self):
        left = [{"b": 2, "a": 1}]
        right = [{"a": 1, "b": 2}, {"a": 9}]
        assert merge_values([left, right]) == [{"b": 2, "a": 1}, {"a": 9}]

    def test_dicts_merge_per_key_and_keep_first_seen_order(self):
        merged = merge_values([{"a": 1, "b": None}, {"b": 2, "c": 3}])
        assert merged == {"a": 1, "b": 2, "c": 3}
        assert list(merged) == ["a", "b", "c"]

    def test_nested_dicts_recurse(self):
        assert merge_values([{"a": {"x": None}}, {"a": {"x": 5, "y": 6}}]) == {
            "a": {"x": 5, "y": 6}
        }

    def test_scalars_take_the_first_non_empty_value(self):
        assert merge_values([None, "", "second", "third"]) == "second"

    def test_all_empty_scalars_fall_back_to_the_first(self):
        assert merge_values([None, None]) is None

    def test_mixed_shapes_fall_back_to_scalar_rules(self):
        assert merge_values([None, [1, 2]]) == [1, 2]

    def test_unserializable_list_items_still_dedupe(self):
        marker = object()
        assert merge_values([[marker], [marker]]) == [marker]


class TestVoteValues:
    def test_empty_input_is_none(self):
        assert vote_values([]) is None

    def test_single_value_passes_through(self):
        assert vote_values(["only"]) == "only"

    def test_lists_fall_back_to_merge(self):
        assert vote_values([[1], [1, 2]]) == [1, 2]

    def test_dicts_vote_per_key(self):
        assert vote_values([{"a": 1}, {"a": 2}, {"a": 2}]) == {"a": 2}

    def test_nested_dicts_recurse(self):
        values = [{"a": {"x": 1}}, {"a": {"x": 2}}, {"a": {"x": 2}}]
        assert vote_values(values) == {"a": {"x": 2}}

    def test_scalar_ties_break_toward_the_earliest_agent(self):
        assert vote_values(["a", "b"]) == "a"

    def test_empty_scalars_never_win(self):
        assert vote_values([None, None, "only"]) == "only"

    def test_all_empty_scalars_fall_back_to_the_first(self):
        assert vote_values(["", ""]) == ""

    def test_mixed_shapes_fall_back_to_scalar_rules(self):
        assert vote_values([{"a": 1}, "text", "text"]) == "text"


class TestReduceOutputs:
    def test_requires_at_least_one_value(self):
        with pytest.raises(ValueError, match="at least one value"):
            reduce_outputs([])

    def test_rejects_unknown_strategy_before_touching_values(self):
        with pytest.raises(ValueError, match="reduce must be one of"):
            reduce_outputs([Person(name="Ada")], "average")

    def test_single_value_skips_reduction(self):
        only = Person(name="Ada", age=36)
        assert (
            reduce_outputs(
                [
                    only,
                ],
                SwarmReduce.VOTE,
            )
            is only
        )

    def test_first_returns_the_first_agent_output(self):
        first = Person(name="Ada")
        assert reduce_outputs([first, Person(name="Grace")], "first") is first

    def test_merge_fills_missing_fields_and_unions_lists(self):
        merged = reduce_outputs(
            [
                Invoices(vendor="Acme", lines=["a"]),
                Invoices(vendor=None, lines=["a", "b"]),
            ],
            "merge",
        )
        assert merged == Invoices(vendor="Acme", lines=["a", "b"])

    def test_vote_keeps_the_majority_field_value(self):
        voted = reduce_outputs(
            [Person(name="Ada", age=36), Person(name="Ada", age=41), Person(name="Ada", age=41)],
            SwarmReduce.VOTE,
        )
        assert voted == Person(name="Ada", age=41)

    def test_merge_defaults_when_no_strategy_is_given(self):
        merged = reduce_outputs([Invoices(lines=["a"]), Invoices(lines=["b"])])
        assert merged.lines == ["a", "b"]

    def test_combined_payload_is_revalidated(self):
        class Strict(BaseModel):
            values: list[int] = []

            def model_dump(self, *args, **kwargs):
                return {"values": ["not-an-int"]}

        with pytest.raises(SchemaValidationError):
            reduce_outputs([Strict(), Strict()], "merge")


class TestReduceCitations:
    def test_empty_items_are_empty(self):
        assert reduce_citations([], Person(name="Ada")) == ()

    def test_single_item_keeps_its_citations(self):
        cites = (Citation("name", "Ada", 1),)
        assert reduce_citations([(Person(name="Ada"), cites)], Person(name="Ada")) is cites

    def test_first_keeps_the_leading_agent(self):
        first = (Citation("name", "Ada", 1),)
        later = (Citation("name", "Grace", 2),)
        assert (
            reduce_citations(
                [(Person(name="Ada"), first), (Person(name="Grace"), later)],
                Person(name="Ada"),
                "first",
            )
            is first
        )

    def test_merge_keeps_cites_for_fields_that_survived(self):
        merged = reduce_citations(
            [
                (Person(name="Ada"), (Citation("name", "Ada", 1),)),
                (Person(name="Ada", age=36), (Citation("age", "36", 1),)),
            ],
            Person(name="Ada", age=36),
            "merge",
        )
        assert merged == (Citation("name", "Ada", 1), Citation("age", "36", 1))

    def test_vote_drops_the_minority_citation(self):
        voted = reduce_citations(
            [
                (Person(name="Ada"), (Citation("name", "Ada", 1),)),
                (Person(name="Grace"), (Citation("name", "Grace", 2),)),
                (Person(name="Grace"), (Citation("name", "Grace Hopper", 3),)),
            ],
            Person(name="Grace"),
            SwarmReduce.VOTE,
        )
        assert voted == (Citation("name", "Grace", 2),)

    def test_nested_paths_compare_the_indexed_value(self):
        left = Invoices(vendor="Acme", lines=["a"])
        right = Invoices(lines=["b"])
        merged = reduce_outputs([left, right], "merge")
        cites = reduce_citations(
            [
                (left, (Citation("lines[0]", "a", 1), Citation("vendor", "Acme", 1))),
                (right, (Citation("lines[0]", "b", 2),)),
            ],
            merged,
            "merge",
        )
        assert cites == (Citation("lines[0]", "a", 1), Citation("vendor", "Acme", 1))

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="reduce must be one of"):
            reduce_citations([(Person(name="Ada"), ())], Person(name="Ada"), "average")
