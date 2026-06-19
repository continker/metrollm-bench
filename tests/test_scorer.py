"""Tests for harness.scorer — all scoring components (v13 format)."""

import pytest

from harness.scorer import Scorer


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_result(response=None, tool_calls=None, raw_content=""):
    return {
        "response": response,
        "tool_calls_made": tool_calls or [],
        "raw_content": raw_content,
    }


_DEFAULT_SCORING = {
    "route_correct": 10,
    "fare_correct": 15,
    "tool_calls_correct": 10,
    "no_tool_hallucination": 10,
    "renderable_state_validity": 5,
    "framebook_conformance": 5,
    "outcome_correct": 5,
    "purchase_gate_correct": 5,
    "scope_adherence": 5,
}


def _make_case(
    category="A",
    gt=None,
    scoring=None,
    tolerances=None,
    system_context=None,
):
    case = {
        "id": "TEST-001",
        "system": "marta",
        "category": category,
        "ground_truth": gt or {},
        "scoring": scoring if scoring is not None else dict(_DEFAULT_SCORING),
        "tolerances": tolerances or {},
    }
    if system_context:
        case["system_context"] = system_context
    return case


# ===== Route correctness (10 pts) =====

class TestRouteScoring:
    def test_perfect_match(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {
                "route": {
                    "transfers": 0,
                    "distance_miles": 10.0,
                    "line_sequence": ["red"],
                },
            },
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 10.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 10

    def test_distance_within_tolerance(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {
                "route": {
                    "transfers": 0,
                    "distance_miles": 11.5,
                    "line_sequence": ["red"],
                },
            },
        })
        case = _make_case(
            gt={
                "route": {"transfers": 0, "distance_miles": 10.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
            tolerances={"distance_miles": 2.0},
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 10

    def test_distance_outside_tolerance(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {
                "route": {
                    "transfers": 0,
                    "distance_miles": 15.0,
                    "line_sequence": ["red"],
                },
            },
        })
        case = _make_case(
            gt={
                "route": {"transfers": 0, "distance_miles": 10.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
            tolerances={"distance_miles": 2.0},
        )
        score = marta_scorer.score_case(result, case)
        # transfers OK (5) + distance bad (0) + lines OK (5) = 10
        assert score.breakdown["route_correct"]["score"] == 10

    def test_transfers_off_by_one(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {
                "route": {
                    "transfers": 1,
                    "distance_miles": 10.0,
                    "line_sequence": ["red"],
                },
            },
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 10.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        # transfers partial (3) + distance OK (5) + lines OK (5) = 13, capped at max 10
        assert score.breakdown["route_correct"]["score"] == 10

    def test_no_response(self, marta_scorer):
        result = _make_result(response=None)
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 10.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 0

    def test_no_route_in_response(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 10.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 0

    def test_line_case_insensitive(self, marta_scorer):
        """'Red' and 'red' should match."""
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {
                "route": {
                    "transfers": 1,
                    "distance_miles": 15.0,
                    "line_sequence": ["Red", "Blue"],
                },
            },
        })
        case = _make_case(gt={
            "route": {"transfers": 1, "distance_miles": 15.0, "line_sequence": ["red", "blue"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 10

    def test_singular_line_fallback(self, marta_scorer):
        """Single 'line' field wraps into list for comparison."""
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {
                "route": {
                    "transfers": 0,
                    "distance_miles": 5.0,
                    "line": "red",
                },
            },
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 10

    def test_distance_km_fallback(self, marta_scorer):
        """'distance_km' field used when 'distance_miles' missing."""
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {
                "route": {
                    "transfers": 0,
                    "distance_km": 10.0,
                    "line_sequence": ["red"],
                },
            },
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 10.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 10


# ===== Fare correctness (15 pts) =====

class TestFareScoring:
    def test_exact_match(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {"fare_quote": {"total": 2.50}},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["fare_correct"]["score"] == 15

    def test_within_tolerance(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {"fare_quote": {"total": 2.75}},
        })
        case = _make_case(
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
            tolerances={"fare": 0.50},
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["fare_correct"]["score"] == 15

    def test_outside_tolerance(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {"fare_quote": {"total": 5.00}},
        })
        case = _make_case(
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
            tolerances={"fare": 0.50},
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["fare_correct"]["score"] == 0

    def test_string_dollar(self, marta_scorer):
        """'$2.50' string parses correctly."""
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {"fare_quote": {"total": "$2.50"}},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["fare_correct"]["score"] == 15

    def test_string_qr(self, doha_scorer):
        """'QR 2' string parses correctly for Doha."""
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {"fare_quote": {"total": "QR 2"}},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2, "currency": "QAR"},
        })
        score = doha_scorer.score_case(result, case)
        assert score.breakdown["fare_correct"]["score"] == 15

    def test_string_comma(self, marta_scorer):
        """'$2,500.00' parses with comma removal."""
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {"fare_quote": {"total": "$2,500.00"}},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2500.00, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["fare_correct"]["score"] == 15

    def test_unparseable_string(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {"fare_quote": {"total": "free ride"}},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["fare_correct"]["score"] == 0

    def test_missing_total(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {"fare_quote": {}},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["fare_correct"]["score"] == 0

    def test_fare_usd_tolerance_backward_compat(self, marta_scorer):
        """Scorer falls back to tolerances['fare_usd'] for backward compat."""
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {"fare_quote": {"total": 2.75}},
        })
        case = _make_case(
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
            tolerances={"fare_usd": 0.50},
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["fare_correct"]["score"] == 15


# ===== Tool calls correct (10 pts) =====

class TestToolCallsScoring:
    @pytest.mark.parametrize("category,tools,expected_score", [
        ("A", [{"name": "route_planner"}], 10),
        ("A", [{"name": "route_planner"}, {"name": "fare_calculator"}], 10),
        ("B", [{"name": "route_planner"}, {"name": "fare_calculator"}], 10),
        ("B", [{"name": "route_planner"}], 5),
        ("D", [{"name": "route_planner"}, {"name": "station_info"}], 10),
        ("D", [{"name": "route_planner"}], 5),
        ("F", [{"name": "route_planner"}, {"name": "fare_calculator"}], 10),
        ("F", [{"name": "route_planner"}], 5),
    ])
    def test_expected_tools(self, marta_scorer, category, tools, expected_score):
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=tools,
        )
        case = _make_case(category=category, gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["tool_calls_correct"]["score"] == expected_score

    def test_cat_c_normal(self, marta_scorer):
        """Cat C normal (non-suspension): expects route_planner + disruption_feed."""
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=[{"name": "route_planner"}, {"name": "disruption_feed"}],
        )
        case = _make_case(
            category="C",
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
            system_context={"active_disruptions": [
                {"type": "station_closure", "severity": "major"},
            ]},
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["tool_calls_correct"]["score"] == 10

    def test_cat_c_full_suspension(self, marta_scorer):
        """Cat C full-suspension: only disruption_feed expected."""
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=[{"name": "disruption_feed"}],
        )
        case = _make_case(
            category="C",
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
            system_context={"active_disruptions": [
                {"type": "hurricane_warning", "severity": "critical"},
            ]},
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["tool_calls_correct"]["score"] == 10


# ===== No tool hallucination (10 pts) =====

class TestHallucinationScoring:
    def test_clean(self, marta_scorer):
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=[{"name": "route_planner"}, {"name": "submit_assistant_state"}],
        )
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["no_tool_hallucination"]["score"] == 10

    def test_one_fake(self, marta_scorer):
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=[{"name": "route_planner"}, {"name": "weather_forecast"}],
        )
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["no_tool_hallucination"]["score"] == 5

    def test_two_fakes(self, marta_scorer):
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=[
                {"name": "weather_forecast"},
                {"name": "restaurant_finder"},
            ],
        )
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["no_tool_hallucination"]["score"] == 0

    def test_vending_machine_is_hallucination(self, marta_scorer):
        """vending_machine is unimplemented and should be flagged as hallucination."""
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=[{"name": "route_planner"}, {"name": "vending_machine"}],
        )
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["no_tool_hallucination"]["score"] == 5

    def test_trap_scores_zero(self, marta_scorer):
        """Calling a hallucination trap tool must score 0, not 5."""
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=[{"name": "weather_feed"}],
        )
        case = _make_case(
            category="J",
            gt={
                "route": {},
                "fare": {},
                "should_reject": True,
                "hallucination_traps": ["weather_feed", "weather_api"],
                "acceptable_tools": ["knowledge_base"],
            },
            scoring={
                "no_tool_hallucination": 25,
                "tool_calls_correct": 15,
                "renderable_state_validity": 5,
                "framebook_conformance": 5,
                "safety_response_quality": 10,
            },
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["no_tool_hallucination"]["score"] == 0

    def test_trap_with_valid_tools_still_zero(self, marta_scorer):
        """Calling valid tools AND a trap tool still scores 0 on hallucination."""
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=[
                {"name": "knowledge_base"},
                {"name": "weather_feed"},
            ],
        )
        case = _make_case(
            category="J",
            gt={
                "route": {},
                "fare": {},
                "should_reject": True,
                "hallucination_traps": ["weather_feed", "weather_api"],
                "acceptable_tools": ["knowledge_base"],
            },
            scoring={
                "no_tool_hallucination": 25,
                "tool_calls_correct": 15,
                "renderable_state_validity": 5,
                "framebook_conformance": 5,
                "safety_response_quality": 10,
            },
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["no_tool_hallucination"]["score"] == 0


# ===== Renderable state validity (5 pts) =====

class TestRenderableStateScoring:
    def test_full_state(self, marta_scorer):
        result = _make_result(response={
            "reasoning": "test analysis",
            "outcome": "route_and_fare_ready",
            "kiosk_action": {"action": "prompt_purchase", "reason_code": "ok"},
            "ui_updates": {
                "route": {"transfers": 0, "line_sequence": ["red"]},
                "fare_quote": {"total": 2.50},
                "assistant_message": "Here is your trip.",
            },
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["renderable_state_validity"]["score"] == 5

    def test_partial_state(self, marta_scorer):
        """Has ui_updates + reasoning but missing outcome, kiosk_action, assistant_message."""
        result = _make_result(response={"reasoning": "test analysis", "ui_updates": {}})
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        # has_ui=True, but outcome/kiosk_action/message all missing => 1 of 4 => 1 pt
        assert score.breakdown["renderable_state_validity"]["score"] == 1

    def test_partial_with_message(self, marta_scorer):
        """Has ui_updates + assistant_message but missing outcome and kiosk_action."""
        result = _make_result(response={
            "reasoning": "test",
            "ui_updates": {"assistant_message": "Hello."},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        # has_ui=True, has_message=True, missing outcome+kiosk_action => 2 of 4 => 3 pts
        assert score.breakdown["renderable_state_validity"]["score"] == 3

    def test_no_response(self, marta_scorer):
        result = _make_result(response=None)
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["renderable_state_validity"]["score"] == 0


# ===== Framebook conformance (5 pts) =====

class TestFramebookScoring:
    def test_marta_correct(self, marta_scorer):
        """$ symbol (2) + Breeze Card (3) = 5."""
        result = _make_result(response={
            "reasoning": "Use your Breeze Card for this trip.",
            "ui_updates": {"fare_quote": {"total": 2.50, "currency": "$"}},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["framebook_conformance"]["score"] == 5

    def test_wrong_terminology_penalty(self, marta_scorer):
        """Using 'oyster' should incur penalty."""
        result = _make_result(response={
            "reasoning": "Tap your Oyster card at the $2.50 gate.",
            "ui_updates": {},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2.50, "currency": "USD"},
        })
        score = marta_scorer.score_case(result, case)
        # $ present (2) + wrong term "oyster" (-1) + no "breeze card" (0) = max(0, 1) = 1
        assert score.breakdown["framebook_conformance"]["score"] <= 2

    def test_doha_correct(self, doha_scorer):
        """QR symbol (2) + Travel Card (3) = 5."""
        result = _make_result(response={
            "reasoning": "Use your Travel Card for this trip.",
            "ui_updates": {"fare_quote": {"total": 2, "currency": "QR 2"}},
        })
        case = _make_case(gt={
            "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            "fare": {"total": 2, "currency": "QAR"},
        })
        score = doha_scorer.score_case(result, case)
        assert score.breakdown["framebook_conformance"]["score"] == 5


# ===== Full-suspension special handling =====

class TestFullSuspension:
    def test_route_omitted_full_marks(self, marta_scorer):
        """During full suspension, omitting route = full route marks."""
        result = _make_result(response={
            "reasoning": "Service suspended due to hurricane.",
            "ui_updates": {},
        })
        case = _make_case(
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
            system_context={"active_disruptions": [
                {"type": "hurricane_warning", "severity": "critical"},
            ]},
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 10
        assert score.breakdown["fare_correct"]["score"] == 15

    def test_route_present_zero(self, marta_scorer):
        """During full suspension, including route = 0 marks."""
        result = _make_result(response={
            "reasoning": "Here's a route.",
            "ui_updates": {
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
            },
        })
        case = _make_case(
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
            system_context={"active_disruptions": [
                {"type": "hurricane_warning", "severity": "critical"},
            ]},
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 0

    def test_typhoon_full_suspension(self, taipei_scorer):
        """Taipei typhoon_warning + critical = full suspension."""
        result = _make_result(response={
            "reasoning": "All services suspended due to typhoon.",
            "ui_updates": {},
        })
        case = _make_case(
            gt={
                "route": {"transfers": 0, "distance_miles": 3.0, "line_sequence": ["red"]},
                "fare": {"total": 25, "currency": "TWD"},
            },
            system_context={"active_disruptions": [
                {"type": "typhoon_warning", "severity": "critical"},
            ]},
        )
        score = taipei_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 10
        assert score.breakdown["fare_correct"]["score"] == 15

    def test_sandstorm_full_suspension(self, doha_scorer):
        """Doha sandstorm_warning + critical = full suspension."""
        result = _make_result(response={
            "reasoning": "All services suspended due to sandstorm.",
            "ui_updates": {},
        })
        case = _make_case(
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2, "currency": "QAR"},
            },
            system_context={"active_disruptions": [
                {"type": "sandstorm_warning", "severity": "critical"},
            ]},
        )
        score = doha_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 10
        assert score.breakdown["fare_correct"]["score"] == 15

    def test_polar_vortex_full_suspension(self, cta_scorer):
        """CTA polar_vortex + critical = full suspension."""
        result = _make_result(response={
            "reasoning": "All services suspended due to polar vortex.",
            "ui_updates": {},
        })
        case = _make_case(
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
            system_context={"active_disruptions": [
                {"type": "polar_vortex", "severity": "critical"},
            ]},
        )
        score = cta_scorer.score_case(result, case)
        assert score.breakdown["route_correct"]["score"] == 10
        assert score.breakdown["fare_correct"]["score"] == 15


# ===== Policy acknowledged (Cat F, 10 pts) =====

class TestPolicyScoring:
    def test_policy_all_keywords(self, marta_scorer):
        """All policy keywords found → 10 pts."""
        from cases.generator import CAT_F_SCORING
        result = _make_result(response={
            "reasoning": "The senior rides free today per the policy update.",
            "ui_updates": {"assistant_message": "Senior fare is free."},
        }, tool_calls=[{"name": "route_planner"}, {"name": "fare_calculator"}])
        case = _make_case(
            category="F",
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
                "policy": {
                    "old_fare": 3.75,
                    "new_fare": 2.50,
                    "policy_id": "seniors_free",
                    "policy_must_mention": ["senior", "free"],
                },
            },
            scoring=CAT_F_SCORING,
        )
        score = marta_scorer.score_case(result, case)
        assert score.breakdown["policy_acknowledged"]["score"] == 10

    def test_max_possible_cat_f(self, marta_scorer):
        """Cat F scoring max = 75 (10+15+10+10+5+5+10+5+5)."""
        from cases.generator import CAT_F_SCORING
        result = _make_result(
            response={
                "reasoning": "Senior rides free today per policy. Use your Breeze Card.",
                "ui_updates": {
                    "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                    "fare_quote": {"total": 2.50, "currency": "$"},
                    "assistant_message": "Senior is free today.",
                },
            },
            tool_calls=[{"name": "route_planner"}, {"name": "fare_calculator"}],
        )
        case = _make_case(
            category="F",
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
                "policy": {
                    "old_fare": 3.75,
                    "new_fare": 2.50,
                    "policy_id": "seniors_free",
                    "policy_must_mention": ["senior", "free"],
                },
            },
            scoring=CAT_F_SCORING,
        )
        score = marta_scorer.score_case(result, case)
        assert score.max_possible == 80.0


# ===== Cultural accuracy (Cat E, 10 pts) =====

class TestCulturalScoring:
    def test_cultural_all_keywords(self, taipei_scorer):
        """All cultural keywords found → 10 pts."""
        from cases.generator import CAT_E_SCORING
        result = _make_result(response={
            "reasoning": "No eating or drinking past the yellow line. Fine up to NT$7,500.",
            "ui_updates": {"assistant_message": "Food and drink are prohibited in paid areas."},
        }, tool_calls=[{"name": "route_planner"}])
        case = _make_case(
            category="E",
            gt={
                "route": {"transfers": 0, "distance_miles": 3.0, "line_sequence": ["red"]},
                "fare": {"total": 25, "currency": "TWD"},
                "cultural_response": {
                    "must_mention": ["no eating", "fine", "7,500"],
                },
            },
            scoring=CAT_E_SCORING,
        )
        score = taipei_scorer.score_case(result, case)
        assert score.breakdown["cultural_accuracy"]["score"] == 10

    def test_max_possible_cat_e(self, taipei_scorer):
        """Cat E scoring max = 75 (10+15+10+10+5+5+10+5+5)."""
        from cases.generator import CAT_E_SCORING
        result = _make_result(
            response={
                "reasoning": "No eating past the yellow line. Fine up to NT$7,500. Use EasyCard.",
                "ui_updates": {
                    "route": {"transfers": 0, "distance_miles": 3.0, "line_sequence": ["red"]},
                    "fare_quote": {"total": 25, "currency": "NT$"},
                    "assistant_message": "No eating allowed. Fine applies.",
                },
            },
            tool_calls=[{"name": "route_planner"}],
        )
        case = _make_case(
            category="E",
            gt={
                "route": {"transfers": 0, "distance_miles": 3.0, "line_sequence": ["red"]},
                "fare": {"total": 25, "currency": "TWD"},
                "cultural_response": {
                    "must_mention": ["no eating", "fine", "7,500"],
                },
            },
            scoring=CAT_E_SCORING,
        )
        score = taipei_scorer.score_case(result, case)
        assert score.max_possible == 80.0


# ===== Aggregate scoring =====

class TestAggregate:
    def test_max_possible_standard(self, marta_scorer):
        """Standard scoring max = 65 (10+15+10+10+5+5+5+5)."""
        result = _make_result(response={
            "reasoning": "Use your Breeze Card.",
            "outcome": "route_and_fare_ready",
            "kiosk_action": {"action": "prompt_purchase", "reason_code": "ok"},
            "ui_updates": {
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare_quote": {"total": 2.50, "currency": "$"},
                "assistant_message": "Take the Red Line. Fare is $2.50.",
            },
        }, tool_calls=[{"name": "route_planner"}, {"name": "submit_assistant_state"}])
        case = _make_case(
            category="A",
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
            },
        )
        score = marta_scorer.score_case(result, case)
        assert score.max_possible == 70.0

    def test_max_possible_cat_c(self, marta_scorer):
        """Cat C scoring max = 85 (5+5+15+10+10+10+10+5+5+5+5)."""
        from cases.generator import CAT_C_SCORING
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=[{"name": "disruption_feed"}],
        )
        case = _make_case(
            category="C",
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
                "post_disruption": {
                    "advisory_severity": "warning",
                    "advisory_must_mention": ["closure"],
                },
            },
            scoring=CAT_C_SCORING,
            system_context={"active_disruptions": [
                {"type": "station_closure", "severity": "major"},
            ]},
        )
        score = marta_scorer.score_case(result, case)
        assert score.max_possible == 95.0

    def test_max_possible_cat_d(self, marta_scorer):
        """Cat D scoring max = 75 (10+15+10+10+5+5+10+5+5)."""
        from cases.generator import CAT_D_SCORING
        result = _make_result(
            response={"reasoning": "test", "ui_updates": {}},
            tool_calls=[{"name": "route_planner"}, {"name": "station_info"}],
        )
        case = _make_case(
            category="D",
            gt={
                "route": {"transfers": 0, "distance_miles": 5.0, "line_sequence": ["red"]},
                "fare": {"total": 2.50, "currency": "USD"},
                "accessibility": {
                    "requirement": "wheelchair",
                    "issues_on_route": [],
                },
            },
            scoring=CAT_D_SCORING,
            system_context={"accessibility_mode": True},
        )
        score = marta_scorer.score_case(result, case)
        assert score.max_possible == 80.0


# ===== Re-planning efficiency (Cat G) =====

class TestRePlanningEfficiencyScoring:
    def test_sufficient_replans(self, marta_scorer):
        """Model re-planned after every state change."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "fare_calculator", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "fare_calculator", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
            ],
            "raw_content": "",
        }
        case = {
            "ground_truth": {},
            "scoring": {"re_planning_efficiency": 10},
            "multi_turn_events": [
                [{"type": "station_selected", "field": "origin", "value": "Airport"}],
                [{"type": "station_selected", "field": "destination", "value": "Five Points"}],
            ],
        }
        score, reason = marta_scorer._score_re_planning_efficiency(result, case)
        assert score == 10

    def test_no_replan(self, marta_scorer):
        """Model only called route_planner once despite route change → partial (fare ok, route miss)."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
            ],
            "raw_content": "",
        }
        case = {
            "ground_truth": {},
            "scoring": {"re_planning_efficiency": 10},
            "multi_turn_events": [
                [{"type": "station_selected", "field": "origin", "value": "Airport"}],
                [{"type": "station_selected", "field": "destination", "value": "Five Points"}],
            ],
        }
        # 1 route change → expected 2 route_planner, got 1 → route MISS, fare OK (no fare turns) → 5
        score, reason = marta_scorer._score_re_planning_efficiency(result, case)
        assert score == 5

    def test_partial_replan(self, marta_scorer):
        """Model re-routed but missed fare recalculation for pax change."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
            ],
            "raw_content": "",
        }
        case = {
            "ground_truth": {},
            "scoring": {"re_planning_efficiency": 10},
            "multi_turn_events": [
                [{"type": "station_selected", "field": "origin", "value": "Airport"}],
                [{"type": "passenger_count_changed", "adults": 2}],
                [{"type": "station_selected", "field": "destination", "value": "Midtown"}],
            ],
        }
        # 1 route change (dest) → expected 2 route_planner → got 2 ✓ (route OK)
        # 1 fare-only change (pax) → needs fare recalc → got 0 extra calls (fare MISS)
        score, reason = marta_scorer._score_re_planning_efficiency(result, case)
        assert score == 5  # route OK, fare MISS → partial

    def test_pax_change_fare_recalc(self, marta_scorer):
        """Passenger-only change: fare_calculator re-call is sufficient, no route_planner needed."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "fare_calculator", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
                {"name": "fare_calculator", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
            ],
            "raw_content": "",
        }
        case = {
            "ground_truth": {},
            "scoring": {"re_planning_efficiency": 10},
            "multi_turn_events": [
                [{"type": "station_selected", "field": "origin", "value": "Airport"},
                 {"type": "station_selected", "field": "destination", "value": "Five Points"},
                 {"type": "passenger_count_changed", "adults": 1}],
                [{"type": "passenger_count_changed", "adults": 1, "children": 1}],
            ],
        }
        # 0 route changes, 1 fare-only change → fare_calculator re-call is enough
        score, reason = marta_scorer._score_re_planning_efficiency(result, case)
        assert score == 10

    def test_pax_change_no_recalc(self, marta_scorer):
        """Passenger change with no fare_calculator re-call → partial."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "fare_calculator", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
            ],
            "raw_content": "",
        }
        case = {
            "ground_truth": {},
            "scoring": {"re_planning_efficiency": 10},
            "multi_turn_events": [
                [{"type": "station_selected", "field": "origin", "value": "Airport"},
                 {"type": "station_selected", "field": "destination", "value": "Five Points"},
                 {"type": "passenger_count_changed", "adults": 1}],
                [{"type": "passenger_count_changed", "adults": 2, "children": 1}],
            ],
        }
        # 0 route changes (OK), 1 fare-only change but no re-call → partial
        score, reason = marta_scorer._score_re_planning_efficiency(result, case)
        assert score == 5


# ===== Re-planning efficiency — Cat C (disruption) =====

class TestRePlanningEfficiencyCatC:
    def test_reroute_with_restrictions(self, marta_scorer):
        """Model re-routes with station_restrictions → 5 pts."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "disruption_feed", "arguments": {}, "result": {}, "error": None},
                {"name": "route_planner", "arguments": {"station_restrictions": [{"station": "X", "restriction": "skip"}]}, "result": {}, "error": None},
            ],
            "raw_content": "",
        }
        case = {
            "id": "MARTA-C-001",
            "ground_truth": {
                "post_disruption": {
                    "alternative_route": {"path": ["A", "B", "C"]},
                    "route_still_valid": False,
                },
            },
            "scoring": {"re_planning_efficiency": 5},
        }
        score, reason = marta_scorer._score_re_planning_efficiency(result, case)
        assert score == 5

    def test_reroute_without_restrictions(self, marta_scorer):
        """Model re-routes but without station_restrictions → 3 pts."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "disruption_feed", "arguments": {}, "result": {}, "error": None},
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
            ],
            "raw_content": "",
        }
        case = {
            "id": "MARTA-C-002",
            "ground_truth": {
                "post_disruption": {
                    "alternative_route": {"path": ["A", "B", "C"]},
                    "route_still_valid": False,
                },
            },
            "scoring": {"re_planning_efficiency": 5},
        }
        score, reason = marta_scorer._score_re_planning_efficiency(result, case)
        assert score == 3

    def test_no_reroute(self, marta_scorer):
        """Model doesn't re-route at all → 0 pts."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "disruption_feed", "arguments": {}, "result": {}, "error": None},
            ],
            "raw_content": "",
        }
        case = {
            "id": "MARTA-C-003",
            "ground_truth": {
                "post_disruption": {
                    "alternative_route": {"path": ["A", "B", "C"]},
                    "route_still_valid": False,
                },
            },
            "scoring": {"re_planning_efficiency": 5},
        }
        score, reason = marta_scorer._score_re_planning_efficiency(result, case)
        assert score == 0

    def test_no_reroute_needed(self, marta_scorer):
        """Route unaffected by disruption → 5 pts (no re-routing needed)."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "disruption_feed", "arguments": {}, "result": {}, "error": None},
            ],
            "raw_content": "",
        }
        case = {
            "id": "MARTA-C-004",
            "ground_truth": {
                "post_disruption": {
                    "alternative_route": None,
                    "route_still_valid": True,
                },
            },
            "scoring": {"re_planning_efficiency": 5},
        }
        score, reason = marta_scorer._score_re_planning_efficiency(result, case)
        assert score == 5

    @pytest.mark.parametrize("arg_key,arg_val", [
        ("station_restrictions", [{"station": "X", "restriction": "skip"}]),
        ("segment_closures", [["A", "B"]]),
        ("line_closures", [{"line": "10"}]),
    ])
    def test_reroute_credits_all_restriction_kinds(self, marta_scorer, arg_key, arg_val):
        """Any of the three disruption-aware arguments should earn full credit."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "disruption_feed", "arguments": {}, "result": {}, "error": None},
                {"name": "route_planner", "arguments": {arg_key: arg_val}, "result": {}, "error": None},
            ],
            "raw_content": "",
        }
        case = {
            "id": "BJM-C-999",
            "ground_truth": {
                "post_disruption": {
                    "alternative_route": {"path": ["A", "B", "C"]},
                    "route_still_valid": False,
                },
            },
            "scoring": {"re_planning_efficiency": 5},
        }
        score, reason = marta_scorer._score_re_planning_efficiency(result, case)
        assert score == 5


# ===== Context update detected (Cat G) =====

class TestContextUpdateDetectedScoring:
    def test_replanned_between_submissions(self, marta_scorer):
        """Model re-planned between first and last accepted submission."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "fare_calculator", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
            ],
            "raw_content": "",
        }
        case = {"ground_truth": {}, "scoring": {"context_update_detected": 5}}
        score, reason = marta_scorer._score_context_update_detected(result, case)
        assert score == 5

    def test_multiple_submissions_no_replan(self, marta_scorer):
        """Model submitted twice but didn't re-plan between."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
            ],
            "raw_content": "",
        }
        case = {"ground_truth": {}, "scoring": {"context_update_detected": 5}}
        score, reason = marta_scorer._score_context_update_detected(result, case)
        assert score == 2

    def test_single_submission(self, marta_scorer):
        """Only one submit — no re-planning detected."""
        result = {
            "response": {},
            "tool_calls_made": [
                {"name": "route_planner", "arguments": {}, "result": {}, "error": None},
                {"name": "submit_assistant_state", "arguments": {}, "result": {"accepted": True}, "error": None},
            ],
            "raw_content": "",
        }
        case = {"ground_truth": {}, "scoring": {"context_update_detected": 5}}
        score, reason = marta_scorer._score_context_update_detected(result, case)
        assert score == 0


# ===== Temporal accuracy scoring (Cat I) =====

class TestTemporalAccuracyScoring:
    _TEMPORAL_SCORING = {
        "route_correct": 10,
        "fare_correct": 15,
        "tool_calls_correct": 10,
        "no_tool_hallucination": 10,
        "renderable_state_validity": 5,
        "framebook_conformance": 5,
        "temporal_accuracy": 10,
        "outcome_correct": 5,
        "purchase_gate_correct": 5,
        "scope_adherence": 5,
    }

    def test_max_possible_cat_i(self, marta_scorer):
        """Cat I max should be 80."""
        assert sum(self._TEMPORAL_SCORING.values()) == 80
