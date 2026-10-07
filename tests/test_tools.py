import pytest

from mcp_toolserver.server import TOOL_SCHEMAS, dispatch
from mcp_toolserver.tools import (
    ToolError,
    calculate,
    date_difference,
    fetch_document,
    search_documents,
)


class TestSearch:
    def test_finds_a_relevant_document(self):
        result = search_documents("refund window")
        assert result["count"] >= 1
        assert result["results"][0]["id"] == "policy-refunds"

    def test_results_are_ranked_by_relevance(self):
        result = search_documents("deploy rollback", limit=5)
        scores = [r["relevance"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_prefix_matching_finds_word_variants(self):
        # "deploy" should still reach "deployment".
        result = search_documents("deploy")
        assert any(r["id"] == "runbook-deploy" for r in result["results"])

    def test_no_match_returns_empty_with_grounding_note(self):
        result = search_documents("quarterly helicopter budget")
        assert result["count"] == 0
        assert "rather than answering from memory" in result["note"]

    def test_limit_is_respected(self):
        assert len(search_documents("the", limit=2)["results"]) <= 2

    def test_team_filter_narrows_results(self):
        result = search_documents("access", team="people")
        assert all(r["team"] == "people" for r in result["results"])

    @pytest.mark.parametrize("bad_query", ["", "   ", None, 42])
    def test_rejects_empty_or_non_string_query(self, bad_query):
        with pytest.raises(ToolError, match="non-empty string"):
            search_documents(bad_query)

    def test_rejects_overlong_query(self):
        with pytest.raises(ToolError, match="at most"):
            search_documents("x" * 500)

    @pytest.mark.parametrize("bad_limit", [0, -1, 999])
    def test_rejects_out_of_range_limit(self, bad_limit):
        with pytest.raises(ToolError, match="between 1 and"):
            search_documents("refund", limit=bad_limit)

    def test_rejects_boolean_limit(self):
        # bool is a subclass of int; it must not slip through.
        with pytest.raises(ToolError, match="must be an integer"):
            search_documents("refund", limit=True)

    def test_unknown_team_lists_the_valid_ones(self):
        with pytest.raises(ToolError, match="known teams are"):
            search_documents("refund", team="marketing")


class TestFetchDocument:
    def test_returns_the_full_body(self):
        doc = fetch_document("policy-pricing")
        assert doc["title"] == "Plan pricing"
        assert "Enterprise" in doc["body"]

    def test_unknown_id_error_lists_available_ids(self):
        with pytest.raises(ToolError, match="Available ids"):
            fetch_document("does-not-exist")

    def test_rejects_empty_id(self):
        with pytest.raises(ToolError, match="non-empty string"):
            fetch_document("")

    def test_returns_a_copy_so_callers_cannot_mutate_the_corpus(self):
        fetch_document("policy-refunds")["body"] = "tampered"
        assert "tampered" not in fetch_document("policy-refunds")["body"]


class TestCalculate:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("2 + 2", 4),
            ("(29 * 12) * 0.85", 295.79999999999995),
            ("sqrt(144)", 12.0),
            ("round(10 / 3, 2)", 3.33),
            ("max(4, 9, 2)", 9),
        ],
    )
    def test_evaluates_valid_expressions(self, expression, expected):
        assert calculate(expression)["result"] == pytest.approx(expected)

    @pytest.mark.parametrize(
        "attack",
        [
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd').read()",
            "().__class__.__bases__[0].__subclasses__()",
            "exec('x=1')",
            "eval('2+2')",
        ],
    )
    def test_blocks_code_execution_attempts(self, attack):
        # The headline security property: a model-generated string must never
        # become arbitrary code.
        with pytest.raises(ToolError):
            calculate(attack)

    def test_rejects_unapproved_function_names(self):
        # Caught by the allow-list of names, after the character filter lets
        # a bare identifier through.
        with pytest.raises(ToolError, match="not allowed"):
            calculate("getattr(1, 2)")

    def test_division_by_zero_is_a_clean_error(self):
        with pytest.raises(ToolError, match="division by zero"):
            calculate("1 / 0")

    def test_syntax_error_is_a_clean_error(self):
        with pytest.raises(ToolError, match="not valid"):
            calculate("2 +* 3")

    def test_rejects_overlong_expression(self):
        with pytest.raises(ToolError, match="too long"):
            calculate("1+" * 200 + "1")

    def test_rejects_non_finite_result(self):
        # Float overflow in plain arithmetic yields inf, which must not escape.
        with pytest.raises(ToolError, match="non-finite"):
            calculate("1e308 * 10")

    def test_math_domain_error_is_a_clean_error(self):
        # math.exp raises OverflowError before a value is ever produced.
        with pytest.raises(ToolError, match="could not evaluate"):
            calculate("exp(10000)")


class TestDateDifference:
    def test_counts_days_forward(self):
        result = date_difference("2026-01-01", "2026-01-15")
        assert result["days"] == 14
        assert result["direction"] == "forward"

    def test_counts_days_backward(self):
        result = date_difference("2026-03-01", "2026-02-01")
        assert result["days"] == -28
        assert result["direction"] == "backward"

    def test_handles_leap_day(self):
        assert date_difference("2024-02-28", "2024-03-01")["days"] == 2

    @pytest.mark.parametrize("bad", ["01/01/2026", "2026-13-01", "not a date", ""])
    def test_rejects_malformed_dates(self, bad):
        with pytest.raises(ToolError):
            date_difference(bad, "2026-01-01")


class TestDispatch:
    def test_routes_to_the_named_tool(self):
        assert dispatch("calculate", {"expression": "6 * 7"})["result"] == 42

    def test_unknown_tool_lists_the_available_ones(self):
        with pytest.raises(ToolError, match="Available tools"):
            dispatch("nope", {})

    def test_wrong_argument_name_is_a_clean_error(self):
        with pytest.raises(ToolError, match="invalid arguments"):
            dispatch("calculate", {"expr": "2+2"})

    def test_missing_required_argument_is_a_clean_error(self):
        with pytest.raises(ToolError, match="invalid arguments"):
            dispatch("date_difference", {"start": "2026-01-01"})

    def test_non_dict_arguments_rejected(self):
        with pytest.raises(ToolError, match="must be an object"):
            dispatch("calculate", ["2+2"])


class TestSchemas:
    def test_every_handler_has_a_schema_and_vice_versa(self):
        from mcp_toolserver.server import HANDLERS

        schema_names = {s["name"] for s in TOOL_SCHEMAS}
        assert schema_names == set(HANDLERS)

    def test_schemas_are_strict_and_documented(self):
        for spec in TOOL_SCHEMAS:
            schema = spec["inputSchema"]
            assert spec["description"].strip(), f"{spec['name']} has no description"
            # additionalProperties:False stops the model inventing parameters.
            assert schema["additionalProperties"] is False, spec["name"]
            assert schema.get("required"), f"{spec['name']} declares no required fields"
            for prop, definition in schema["properties"].items():
                assert definition.get("description"), f"{spec['name']}.{prop} undocumented"
