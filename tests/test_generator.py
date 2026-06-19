"""Tests for cases.generator — ground truth consistency and case structure."""

import pytest
from pathlib import Path

from harness.graph import MetroGraph
from harness.fares import FareCalculator
from cases.generator import (
    generate_category_a,
    generate_category_b,
    generate_category_c,
    generate_category_d,
    generate_category_e,
    generate_category_f,
    generate_category_g,
    generate_category_h,
    generate_category_j,
    generate_category_i,
    generate_category_k,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["marta", "doha", "bart", "taipei", "cta", "beijing"])
def system_setup(request):
    """Return (graph, fare_calc, system_name, system_dir) for each system."""
    name = request.param
    system_dir = PROJECT_ROOT / "data" / "systems" / name
    graph = MetroGraph(system_dir)
    fare_calc = FareCalculator(system_dir)
    return graph, fare_calc, name, system_dir


@pytest.fixture
def cat_a_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_a(graph, fare_calc, name, system_dir)


@pytest.fixture
def cat_b_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_b(graph, fare_calc, name, system_dir)


@pytest.fixture
def cat_c_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_c(graph, fare_calc, name, system_dir)


@pytest.fixture
def cat_d_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_d(graph, fare_calc, name, system_dir)


@pytest.fixture
def cat_e_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_e(graph, fare_calc, name, system_dir)


@pytest.fixture
def cat_f_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_f(graph, fare_calc, name, system_dir)


@pytest.fixture
def cat_g_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_g(graph, fare_calc, name, system_dir)


@pytest.fixture
def cat_h_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_h(graph, fare_calc, name, system_dir)


@pytest.fixture
def cat_j_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_j(graph, fare_calc, name, system_dir)


@pytest.fixture
def cat_i_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_i(graph, fare_calc, name, system_dir)


@pytest.fixture
def cat_k_cases(system_setup):
    graph, fare_calc, name, system_dir = system_setup
    return generate_category_k(graph, fare_calc, name, system_dir)


@pytest.fixture
def all_cases(cat_a_cases, cat_b_cases, cat_c_cases, cat_d_cases, cat_e_cases, cat_f_cases, cat_g_cases, cat_h_cases, cat_j_cases, cat_i_cases, cat_k_cases):
    return cat_a_cases + cat_b_cases + cat_c_cases + cat_d_cases + cat_e_cases + cat_f_cases + cat_g_cases + cat_h_cases + cat_j_cases + cat_i_cases + cat_k_cases


# ===== Case counts =====

class TestCaseCounts:
    def test_cat_a_count(self, cat_a_cases):
        # v23: Beijing adds 1 direction variant → 21; others remain 20
        assert 20 <= len(cat_a_cases) <= 21

    def test_cat_b_count(self, cat_b_cases):
        # v23: freetext_balance (BART, Taipei) and advisory_extra (Beijing) push max to 16
        assert 15 <= len(cat_b_cases) <= 16

    def test_cat_c_count(self, cat_c_cases):
        # v23: +1 for Beijing SIC, CTA Loop, Taipei weather, Doha bus-bridge respectively
        assert 15 <= len(cat_c_cases) <= 25, \
            f"Cat C count {len(cat_c_cases)} not in [15, 25]"

    def test_cat_d_count(self, cat_d_cases):
        # v23: MARTA adds 1 with_disruption → 16; others remain 15
        assert 15 <= len(cat_d_cases) <= 16

    def test_cat_e_count(self, cat_e_cases, system_setup):
        _, _, name, _ = system_setup
        if name == "taipei":
            expected_min, expected_max = 15, 15
        elif name == "doha":
            expected_min, expected_max = 7, 7  # v23 +2 cultural variants
        else:
            expected_min, expected_max = 5, 5
        assert expected_min <= len(cat_e_cases) <= expected_max

    def test_cat_f_count(self, cat_f_cases):
        # v23: BART/CTA +1 routing-impact policy; MARTA +2 (Green + holiday Sunday)
        assert 15 <= len(cat_f_cases) <= 17

    def test_cat_g_count(self, cat_g_cases):
        assert len(cat_g_cases) == 15

    def test_cat_h_count(self, cat_h_cases):
        assert len(cat_h_cases) == 15

    def test_cat_j_count(self, cat_j_cases):
        assert len(cat_j_cases) == 15

    def test_cat_i_count(self, cat_i_cases):
        assert len(cat_i_cases) == 15

    def test_cat_k_count(self, cat_k_cases):
        assert len(cat_k_cases) == 5


# ===== Required fields =====

class TestRequiredFields:
    REQUIRED_KEYS = {"id", "system", "category", "difficulty", "events",
                     "system_context", "ground_truth", "scoring", "tolerances"}

    def test_all_cases_have_required_fields(self, all_cases):
        for case in all_cases:
            missing = self.REQUIRED_KEYS - set(case.keys())
            assert not missing, f"{case['id']} missing fields: {missing}"

    def test_ground_truth_has_route_and_fare(self, all_cases):
        # Cat F routing-impact and Cat D disruption-combo cases may resolve
        # to service_unavailable / advisory_only, with route/fare set to None.
        for case in all_cases:
            gt = case["ground_truth"]
            assert "route" in gt, f"{case['id']} missing GT route"
            assert "fare" in gt, f"{case['id']} missing GT fare"

    def test_events_not_empty(self, all_cases):
        for case in all_cases:
            # Cat H/J/I/K can have a single event (e.g. freetext_input)
            min_events = 1 if case["category"] in ("H", "J", "I", "K") else 2
            assert len(case["events"]) >= min_events, f"{case['id']} has < {min_events} events"

    def test_gold_class_has_payment_event(self, cat_b_cases):
        for case in cat_b_cases:
            label = case.get("passenger_composition", "")
            if "gold" in label.lower():
                has_pm = any(
                    e["type"] == "payment_method_selected" for e in case["events"]
                )
                assert has_pm, f"{case['id']} gold class case missing payment_method_selected event"


# ===== Ground truth route validity =====

class TestGroundTruthRoutes:
    def test_routes_are_valid_paths(self, all_cases, system_setup):
        graph, _, _, _ = system_setup
        for case in all_cases:
            route = case["ground_truth"]["route"]
            if route is None:
                # Cat H/J rejection cases may have no route
                continue
            path = route["path"]
            assert graph.is_valid_path(path), \
                f"{case['id']} GT route is not a valid path: {path}"

    def test_routes_have_positive_distance(self, all_cases):
        for case in all_cases:
            route = case["ground_truth"]["route"]
            if route is None:
                # Cat H/J rejection cases may have no route
                continue
            if len(route["path"]) > 1:
                assert route["distance_miles"] > 0, \
                    f"{case['id']} has zero distance for multi-station route"


# ===== Ground truth fare consistency =====

class TestGroundTruthFares:
    def test_fares_match_calculator(self, cat_b_cases, system_setup):
        """Cat B fares should exactly match FareCalculator output."""
        _, fare_calc, name, _ = system_setup
        for case in cat_b_cases:
            gt_total = case["ground_truth"]["fare"]["total"]
            # Reconstruct passengers from events
            pax_event = next(
                e for e in case["events"]
                if e["type"] == "passenger_count_changed"
            )
            passengers = {}
            for key in ("adults", "children", "seniors", "disabled"):
                if key in pax_event:
                    passengers[key] = pax_event[key]

            # Detect payment method from events
            pm_event = next(
                (e for e in case["events"]
                 if e["type"] == "payment_method_selected"),
                None,
            )
            payment = pm_event["method"] if pm_event else "smartcard"

            # Extract route info for distance-based fare models
            gt_route = case["ground_truth"].get("route", {})
            distance = gt_route.get("distance_miles")
            origin_events = [e for e in case["events"] if e.get("field") == "origin"]
            dest_events = [e for e in case["events"] if e.get("field") == "destination"]
            origin_id = origin_events[0]["station_id"] if origin_events else None
            dest_id = dest_events[0]["station_id"] if dest_events else None

            result = fare_calc.calculate(
                passengers=passengers, payment_method=payment,
                route_distance_miles=distance,
                origin_id=origin_id, destination_id=dest_id,
            )
            assert result.total == gt_total, \
                f"{case['id']} fare mismatch: calc={result.total} vs gt={gt_total}"


# ===== Category C structure =====

class TestCatCStructure:
    def test_has_disruption_scoring(self, cat_c_cases):
        for case in cat_c_cases:
            scoring = case["scoring"]
            assert "disruption_detected" in scoring, \
                f"{case['id']} missing disruption_detected scoring"
            assert "advisory_issued" in scoring
            assert "advisory_content_correct" in scoring

    def test_has_active_disruptions(self, cat_c_cases):
        for case in cat_c_cases:
            disruptions = case["system_context"].get("active_disruptions", [])
            assert len(disruptions) > 0, \
                f"{case['id']} has no active disruptions"

    def test_has_post_disruption_gt(self, cat_c_cases):
        for case in cat_c_cases:
            gt = case["ground_truth"]
            assert "post_disruption" in gt, \
                f"{case['id']} missing post_disruption ground truth"
            pd = gt["post_disruption"]
            assert "advisory_severity" in pd
            assert "advisory_must_mention" in pd

    def test_disruption_event_in_events(self, cat_c_cases):
        for case in cat_c_cases:
            has_disruption = any(
                e["type"] == "disruption_update" for e in case["events"]
            )
            assert has_disruption, f"{case['id']} missing disruption_update event"

    def test_has_restriction_type(self, cat_c_cases):
        """Station closure disruptions should produce skip restriction."""
        for case in cat_c_cases:
            pd = case["ground_truth"]["post_disruption"]
            assert "restriction_type" in pd, \
                f"{case['id']} missing restriction_type"
            assert pd["restriction_type"] in ("skip", "closed"), \
                f"{case['id']} unexpected restriction_type: {pd['restriction_type']}"

    def test_has_expected_restrictions(self, cat_c_cases):
        for case in cat_c_cases:
            pd = case["ground_truth"]["post_disruption"]
            assert "expected_restrictions" in pd, \
                f"{case['id']} missing expected_restrictions"
            assert isinstance(pd["expected_restrictions"], list)
            for r in pd["expected_restrictions"]:
                assert "station" in r and "restriction" in r

    def test_has_expected_segment_closures(self, cat_c_cases):
        for case in cat_c_cases:
            pd = case["ground_truth"]["post_disruption"]
            assert "expected_segment_closures" in pd, \
                f"{case['id']} missing expected_segment_closures"
            assert isinstance(pd["expected_segment_closures"], list)

    def test_has_original_route(self, cat_c_cases):
        for case in cat_c_cases:
            gt = case["ground_truth"]
            assert "original_route" in gt, \
                f"{case['id']} missing original_route"
            orig = gt["original_route"]
            assert "path" in orig
            assert "line_sequence" in orig
            assert "transfers" in orig

    def test_alternative_route_implies_route_and_fare_ready(self, cat_c_cases):
        """Cases with an alternative route should expect route_and_fare_ready."""
        for case in cat_c_cases:
            pd = case["ground_truth"]["post_disruption"]
            if pd.get("alternative_route") is not None:
                assert case["ground_truth"]["expected_outcome"] == "route_and_fare_ready", \
                    f"{case['id']} has alt route but outcome={case['ground_truth'].get('expected_outcome')}"

    def test_has_expected_line_closures(self, cat_c_cases):
        """Every Cat C case exposes expected_line_closures (list, possibly empty)."""
        for case in cat_c_cases:
            pd = case["ground_truth"]["post_disruption"]
            assert "expected_line_closures" in pd, f"{case['id']} missing expected_line_closures"
            assert isinstance(pd["expected_line_closures"], list)

    def test_line_suspension_cases_carry_line_closures(self, cat_c_cases):
        """Cases generated from line_suspension template must have non-empty expected_line_closures."""
        ls_cases = [c for c in cat_c_cases
                    if c["ground_truth"]["post_disruption"].get("expected_line_closures")]
        if not ls_cases:
            pytest.skip("no line_suspension template instantiated for this system")
        for case in ls_cases:
            pd = case["ground_truth"]["post_disruption"]
            # Each closure entry names a line
            for lc in pd["expected_line_closures"]:
                assert "line" in lc, f"{case['id']} line_closure missing 'line'"


# ===== Category D structure =====

class TestCatDStructure:
    def test_has_accessibility_gt(self, cat_d_cases):
        for case in cat_d_cases:
            gt = case["ground_truth"]
            assert "accessibility" in gt, \
                f"{case['id']} missing accessibility ground truth"
            acc = gt["accessibility"]
            assert "requirement" in acc
            assert "issues_on_route" in acc

    def test_has_accessibility_scoring(self, cat_d_cases):
        for case in cat_d_cases:
            assert "accessibility_accuracy" in case["scoring"], \
                f"{case['id']} missing accessibility_accuracy scoring"

    def test_has_accessibility_mode(self, cat_d_cases):
        for case in cat_d_cases:
            assert case["system_context"].get("accessibility_mode") is True, \
                f"{case['id']} missing accessibility_mode in system_context"

    def test_has_freetext_event(self, cat_d_cases):
        for case in cat_d_cases:
            has_freetext = any(
                e["type"] == "freetext_input" for e in case["events"]
            )
            assert has_freetext, f"{case['id']} missing freetext_input event"


# ===== Category F structure =====

class TestCatFStructure:
    def test_has_policy_gt(self, cat_f_cases):
        for case in cat_f_cases:
            gt = case["ground_truth"]
            assert "policy" in gt, \
                f"{case['id']} missing policy ground truth"
            policy = gt["policy"]
            assert "old_fare" in policy
            assert "new_fare" in policy
            assert "policy_id" in policy
            assert "policy_must_mention" in policy

    def test_has_policy_in_system_context(self, cat_f_cases):
        for case in cat_f_cases:
            sc = case["system_context"]
            assert "policy_change" in sc, \
                f"{case['id']} missing policy_change in system_context"
            pc = sc["policy_change"]
            assert "text" in pc
            assert "policy_id" in pc

    def test_has_policy_scoring(self, cat_f_cases):
        for case in cat_f_cases:
            assert "policy_acknowledged" in case["scoring"], \
                f"{case['id']} missing policy_acknowledged scoring"

    def test_easy_policies_fare_unchanged(self, cat_f_cases):
        """Tier 1 (easy) policies should not change the fare."""
        for case in cat_f_cases:
            if case["difficulty"] == "easy":
                gt = case["ground_truth"]
                assert gt["policy"]["old_fare"] == gt["policy"]["new_fare"], \
                    f"{case['id']} easy policy changed fare"
                assert gt["fare"]["total"] == gt["policy"]["old_fare"]

    def test_medium_hard_policies_category(self, cat_f_cases):
        for case in cat_f_cases:
            assert case["category"] == "F"


# ===== Cross-cutting =====

class TestCrossCutting:
    def test_all_ids_unique(self, all_cases):
        ids = [c["id"] for c in all_cases]
        assert len(ids) == len(set(ids)), \
            f"Duplicate case IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_station_ids_in_events_exist(self, all_cases, system_setup):
        graph, _, _, _ = system_setup
        for case in all_cases:
            for event in case["events"]:
                if "station_id" in event:
                    sid = event["station_id"]
                    assert sid in graph.stations, \
                        f"{case['id']} references unknown station {sid}"


# ===== Category E structure =====

class TestCatEStructure:
    def test_has_cultural_gt(self, cat_e_cases):
        for case in cat_e_cases:
            gt = case["ground_truth"]
            assert "cultural_response" in gt, \
                f"{case['id']} missing cultural_response ground truth"
            cr = gt["cultural_response"]
            assert "must_mention" in cr
            assert len(cr["must_mention"]) > 0

    def test_has_cultural_scoring(self, cat_e_cases):
        for case in cat_e_cases:
            assert "cultural_accuracy" in case["scoring"], \
                f"{case['id']} missing cultural_accuracy scoring"

    def test_has_freetext_event(self, cat_e_cases):
        for case in cat_e_cases:
            has_freetext = any(
                e["type"] == "freetext_input" for e in case["events"]
            )
            assert has_freetext, f"{case['id']} missing freetext_input event"

    def test_category_is_e(self, cat_e_cases):
        for case in cat_e_cases:
            assert case["category"] == "E"


# ===== Category G structure =====

class TestCatGStructure:
    def test_has_multi_turn_events(self, cat_g_cases):
        for case in cat_g_cases:
            assert "multi_turn_events" in case, f"{case['id']} missing multi_turn_events"
            mte = case["multi_turn_events"]
            assert len(mte) >= 2, f"{case['id']} has < 2 turn groups"

    def test_each_turn_has_events(self, cat_g_cases):
        for case in cat_g_cases:
            for i, turn in enumerate(case["multi_turn_events"]):
                assert len(turn) >= 1, f"{case['id']} turn {i} has no events"

    def test_has_context_update_scoring(self, cat_g_cases):
        for case in cat_g_cases:
            assert "context_update_detected" in case["scoring"], \
                f"{case['id']} missing context_update_detected scoring"

    def test_has_replanning_scoring(self, cat_g_cases):
        for case in cat_g_cases:
            assert "re_planning_efficiency" in case["scoring"], \
                f"{case['id']} missing re_planning_efficiency scoring"

    def test_category_is_g(self, cat_g_cases):
        for case in cat_g_cases:
            assert case["category"] == "G"

    def test_events_flat_matches_multi_turn(self, cat_g_cases):
        """Flat events should equal all multi_turn_events concatenated."""
        for case in cat_g_cases:
            flat = []
            for turn in case["multi_turn_events"]:
                flat.extend(turn)
            assert len(case["events"]) == len(flat), \
                f"{case['id']} flat events count mismatch"


# ===== Category H structure =====

class TestCatHStructure:
    def test_has_should_reject(self, cat_h_cases):
        for case in cat_h_cases:
            gt = case["ground_truth"]
            assert "should_reject" in gt, f"{case['id']} missing should_reject"
            assert isinstance(gt["should_reject"], bool)

    def test_has_safety_scoring(self, cat_h_cases):
        for case in cat_h_cases:
            assert "safety_response_quality" in case["scoring"], \
                f"{case['id']} missing safety_response_quality scoring"
            assert "no_data_fabrication" in case["scoring"], \
                f"{case['id']} missing no_data_fabrication scoring"

    def test_has_acceptable_response_patterns(self, cat_h_cases):
        for case in cat_h_cases:
            gt = case["ground_truth"]
            assert "acceptable_response_patterns" in gt, \
                f"{case['id']} missing acceptable_response_patterns"
            assert len(gt["acceptable_response_patterns"]) > 0

    def test_has_freetext_event(self, cat_h_cases):
        for case in cat_h_cases:
            has_freetext = any(
                e["type"] == "freetext_input" for e in case["events"]
            )
            # Most H cases use freetext, but some use structured events (invalid pax)
            has_pax = any(
                e["type"] == "passenger_count_changed" for e in case["events"]
            )
            # Some contradictory cases use only station_selected events
            has_station = any(
                e["type"] == "station_selected" for e in case["events"]
            )
            assert has_freetext or has_pax or has_station, f"{case['id']} missing interaction event"

    def test_category_is_h(self, cat_h_cases):
        for case in cat_h_cases:
            assert case["category"] == "H"

    def test_uses_diverse_stations(self, cat_h_cases):
        """Cat H should rotate through memorizable pairs, not use one pair."""
        event_strs = set()
        for case in cat_h_cases:
            event_strs.add(str(case["events"]))
        assert len(event_strs) > 5, "Cat H cases should have diverse event content"


# ===== Category J structure =====

class TestCatJStructure:
    def test_has_hallucination_traps(self, cat_j_cases):
        for case in cat_j_cases:
            gt = case["ground_truth"]
            assert "hallucination_traps" in gt, \
                f"{case['id']} missing hallucination_traps"

    def test_has_acceptable_tools(self, cat_j_cases):
        for case in cat_j_cases:
            gt = case["ground_truth"]
            assert "acceptable_tools" in gt, \
                f"{case['id']} missing acceptable_tools"
            assert len(gt["acceptable_tools"]) > 0

    def test_hallucination_scoring_weight(self, cat_j_cases):
        for case in cat_j_cases:
            assert case["scoring"]["no_tool_hallucination"] == 25, \
                f"{case['id']} hallucination weight should be 25"

    def test_has_freetext_event(self, cat_j_cases):
        for case in cat_j_cases:
            has_freetext = any(
                e["type"] == "freetext_input" for e in case["events"]
            )
            assert has_freetext, f"{case['id']} missing freetext_input event"

    def test_has_should_reject(self, cat_j_cases):
        for case in cat_j_cases:
            gt = case["ground_truth"]
            assert "should_reject" in gt, f"{case['id']} missing should_reject"
            assert isinstance(gt["should_reject"], bool)

    def test_category_is_j(self, cat_j_cases):
        for case in cat_j_cases:
            assert case["category"] == "J"


# ===== Category I structure =====

class TestCatIStructure:
    def test_has_temporal_gt(self, cat_i_cases):
        for case in cat_i_cases:
            gt = case["ground_truth"]
            assert "temporal" in gt, f"{case['id']} missing temporal ground truth"
            t = gt["temporal"]
            assert "service_available" in t
            assert "should_warn_last_train" in t
            assert "temporal_keywords" in t
            assert len(t["temporal_keywords"]) > 0

    def test_has_temporal_scoring(self, cat_i_cases):
        for case in cat_i_cases:
            assert "temporal_accuracy" in case["scoring"], \
                f"{case['id']} missing temporal_accuracy scoring"
            assert case["scoring"]["temporal_accuracy"] == 10

    def test_has_temporal_context(self, cat_i_cases):
        for case in cat_i_cases:
            sc = case["system_context"]
            assert "temporal_context" in sc, \
                f"{case['id']} missing temporal_context in system_context"
            tc = sc["temporal_context"]
            assert "current_time" in tc
            assert "day_of_week" in tc

    def test_has_freetext_event(self, cat_i_cases):
        for case in cat_i_cases:
            has_freetext = any(
                e["type"] == "freetext_input" for e in case["events"]
            )
            assert has_freetext, f"{case['id']} missing freetext_input event"

    def test_category_is_i(self, cat_i_cases):
        for case in cat_i_cases:
            assert case["category"] == "I"

    def test_no_route_when_service_unavailable(self, cat_i_cases):
        for case in cat_i_cases:
            t = case["ground_truth"]["temporal"]
            if not t["service_available"]:
                assert case["ground_truth"]["route"] is None, \
                    f"{case['id']} has route but service is unavailable"
                assert case["ground_truth"]["fare"] is None, \
                    f"{case['id']} has fare but service is unavailable"

    def test_max_possible_75(self, cat_i_cases):
        for case in cat_i_cases:
            total = sum(case["scoring"].values())
            assert total == 80, f"{case['id']} scoring max {total} != 80"


# ===== Category K structure =====

class TestCatKStructure:
    def test_category_is_k(self, cat_k_cases):
        for case in cat_k_cases:
            assert case["category"] == "K"

    def test_has_compound_modes(self, cat_k_cases):
        for case in cat_k_cases:
            modes = case.get("compound_modes", [])
            assert len(modes) >= 2, f"{case['id']} has < 2 compound modes: {modes}"

    def test_has_multiple_scoring_dimensions(self, cat_k_cases):
        """Each case should score on at least 3 category-specific components beyond standard."""
        standard = {"route_correct", "fare_correct", "tool_calls_correct",
                     "no_tool_hallucination", "renderable_state_validity", "framebook_conformance",
                     "outcome_correct", "purchase_gate_correct"}
        for case in cat_k_cases:
            extra = set(case["scoring"].keys()) - standard
            assert len(extra) >= 2, \
                f"{case['id']} only has {len(extra)} extra components: {extra}"

    def test_modes_match_system_context(self, cat_k_cases):
        for case in cat_k_cases:
            modes = case["compound_modes"]
            sc = case["system_context"]
            if "disruption" in modes:
                assert sc.get("active_disruptions"), f"{case['id']} missing disruptions"
            if "accessibility" in modes:
                assert sc.get("accessibility_mode"), f"{case['id']} missing accessibility_mode"
            if "temporal" in modes:
                assert sc.get("temporal_context"), f"{case['id']} missing temporal_context"
            if "policy" in modes:
                assert sc.get("policy_change"), f"{case['id']} missing policy_change"

    def test_scoring_max_80_to_95(self, cat_k_cases):
        for case in cat_k_cases:
            total = sum(case["scoring"].values())
            assert 85 <= total <= 100, f"{case['id']} scoring max {total} not in [85, 100]"


# ===== Cross-cutting invariants =====
# These catch the bug classes that actually bit us (TRTC-H-009, scoring drift).

class TestOutcomeInvariants:
    """Outcome/kiosk_action must be consistent with ground truth semantics."""

    def test_info_only_cases_not_prompt_purchase(self, cat_h_cases, cat_j_cases):
        """Cases with only info tools should never prompt_purchase (TRTC-H-009 bug)."""
        _ROUTE_TOOLS = {"route_planner", "fare_calculator"}
        for case in cat_h_cases + cat_j_cases:
            gt = case["ground_truth"]
            tools = set(gt.get("acceptable_tools", []))
            if not gt.get("should_reject") and not (tools & _ROUTE_TOOLS) and tools:
                assert gt["expected_outcome"] == "policy_answer_only", \
                    f"{case['id']}: info-only tools {tools} but outcome={gt['expected_outcome']}"
                assert gt["expected_kiosk_action"] == "display_info", \
                    f"{case['id']}: info-only but kiosk_action={gt['expected_kiosk_action']}"

    def test_no_route_means_no_prompt_purchase(self, all_cases):
        """If ground truth route is None, kiosk should not prompt_purchase."""
        for case in all_cases:
            gt = case["ground_truth"]
            if gt.get("route") is None and gt.get("expected_kiosk_action") == "prompt_purchase":
                # Exception: Cat H/J cases where route is null but model does route planning
                cat = case["category"]
                if cat in ("H", "J"):
                    tools = set(gt.get("acceptable_tools", []))
                    if "route_planner" in tools:
                        continue
                assert False, \
                    f"{case['id']} (Cat {case['category']}): route=None but kiosk_action=prompt_purchase"


class TestScoringInvariants:
    """All scoring dicts must include required components."""

    def test_all_cases_have_scope_adherence(self, all_cases):
        """Every case must include scope_adherence scoring."""
        for case in all_cases:
            assert "scope_adherence" in case["scoring"], \
                f"{case['id']} missing scope_adherence in scoring"
            assert case["scoring"]["scope_adherence"] == 5

    def test_all_cases_have_framebook_conformance(self, all_cases):
        """Every case must include framebook_conformance scoring."""
        for case in all_cases:
            assert "framebook_conformance" in case["scoring"], \
                f"{case['id']} missing framebook_conformance in scoring"


# ===== v23 scenario-gap-audit additions =====

class TestV23ScenarioGaps:
    """Regression tests for the 15 gap-audit cases added in v23."""

    def test_cat_f_routing_impact_emits_expected_restrictions(self, cat_f_cases, system_setup):
        """CTA State/Lake routing-impact policy carries expected_station_restrictions.

        Note: BART Yellow and MARTA Green short-turns intentionally omit
        routing_impact because their segments are shared across lines
        (closing the physical edge would also block unaffected lines).
        The policy prose + advisory_must_mention carry the test signal.
        """
        _, _, name, _ = system_setup
        for case in cat_f_cases:
            pid = case.get("policy_id", "")
            if pid == "cta_state_lake_closed":
                assert name == "cta"
                policy = case["ground_truth"]["policy"]
                assert "expected_station_restrictions" in policy, "CTA State/Lake missing expected_station_restrictions"
                assert policy["expected_station_restrictions"][0]["station"] == "CTA-STL"

    def test_cat_f_routing_impact_has_advisory_must_mention(self, cat_f_cases, system_setup):
        """Routing-impact Cat F policies carry advisory_must_mention for judge."""
        _, _, name, _ = system_setup
        for case in cat_f_cases:
            policy_gt = case["ground_truth"].get("policy", {})
            pid = policy_gt.get("policy_id", "")
            if pid in {"bart_yellow_night_shuttle", "marta_green_kingmemorial_shortturn",
                       "cta_state_lake_closed", "marta_holiday_sunday_schedule"}:
                assert "advisory_must_mention" in policy_gt, \
                    f"{case['id']} ({pid}) missing advisory_must_mention"
                assert case["scoring"].get("advisory_content_correct") == 10, \
                    f"{case['id']} missing advisory_content_correct=10"

    def test_cat_c_station_entry_restriction_advisory_only(self, cat_c_cases, system_setup):
        """Beijing SIC case resolves to advisory_only with wait/entry keywords."""
        _, _, name, _ = system_setup
        if name != "beijing":
            return
        matches = [c for c in cat_c_cases if c.get("disruption_type", "").startswith("ser_guomao")]
        assert matches, "Beijing should have station_entry_restriction case"
        case = matches[0]
        assert case["ground_truth"]["expected_outcome"] == "advisory_only"
        must = case["ground_truth"]["post_disruption"].get("advisory_must_mention", [])
        assert "wait" in must and "flow control" in must

    def test_cat_c_weather_slow_zone_route_valid(self, cat_c_cases, system_setup):
        """Taipei weather slow zone is route_and_fare_ready with delay advisory."""
        _, _, name, _ = system_setup
        if name != "taipei":
            return
        matches = [c for c in cat_c_cases if c.get("disruption_type", "").startswith("wsz_")]
        assert matches, "Taipei should have weather_slow_zone case"
        case = matches[0]
        assert case["ground_truth"]["expected_outcome"] == "route_and_fare_ready"
        must = case["ground_truth"]["post_disruption"].get("advisory_must_mention", [])
        assert "delay" in must and "weather" in must

    def test_cat_d_with_disruption_service_unavailable(self, cat_d_cases, system_setup):
        """MARTA Five Points elevator + wheelchair resolves to service_unavailable."""
        _, _, name, _ = system_setup
        if name != "marta":
            return
        matches = [c for c in cat_d_cases if c.get("accessibility_tier") == "with_disruption"]
        assert matches, "MARTA should have with_disruption Cat D case"
        case = matches[0]
        assert case["ground_truth"]["expected_outcome"] == "service_unavailable"
        assert case["ground_truth"]["expected_kiosk_action"] == "refer_to_staff"
        assert case["ground_truth"]["route"] is None
        assert case["scoring"].get("advisory_content_correct") == 10

    def test_cat_b_freetext_balance_includes_top_up(self, cat_b_cases, system_setup):
        """BART and Taipei freetext_balance cases encode top-up amount in advisory_must_mention."""
        _, _, name, _ = system_setup
        if name not in {"bart", "taipei"}:
            return
        matches = [c for c in cat_b_cases if "balance" in c.get("passenger_composition", "")]
        assert matches, f"{name} should have freetext_balance case"
        case = matches[0]
        gt = case["ground_truth"]
        assert "balance_advisory" in gt
        adv = gt["advisory_must_mention"]
        sym = gt["balance_advisory"]["currency_symbol"]
        top_up = gt["balance_advisory"]["top_up_amount"]
        expected_str = f"{sym}{top_up:.2f}"
        assert expected_str in adv, f"advisory_must_mention should include {expected_str}; got {adv}"
        assert case["scoring"].get("advisory_content_correct") == 10

    def test_cat_a_direction_includes_direction_keyword(self, cat_a_cases, system_setup):
        """Beijing Line 10 circular direction case carries direction in advisory_must_mention."""
        _, _, name, _ = system_setup
        if name != "beijing":
            return
        matches = [c for c in cat_a_cases if c.get("route_type") == "direction"]
        assert matches, "Beijing should have direction Cat A case"
        case = matches[0]
        gt = case["ground_truth"]
        assert gt.get("expected_direction") in {"clockwise", "counterclockwise"}
        assert gt["expected_direction"] in gt.get("advisory_must_mention", [])
        assert case["scoring"].get("advisory_content_correct") == 10

    def test_cat_c_loop_shutdown_multi_line(self, cat_c_cases, system_setup):
        """CTA Loop shutdown encodes multiple line_closures."""
        _, _, name, _ = system_setup
        if name != "cta":
            return
        matches = [c for c in cat_c_cases if c.get("disruption_type", "") == "ls_loop-shutdown"]
        assert matches, "CTA should have loop-shutdown case"
        case = matches[0]
        closures = case["ground_truth"]["post_disruption"].get("expected_line_closures", [])
        lines_closed = {c["line"] for c in closures}
        assert {"brown", "orange", "pink"} <= lines_closed, \
            f"Loop shutdown should close brown/orange/pink; got {lines_closed}"

    def test_cat_c_doha_bus_bridge(self, cat_c_cases, system_setup):
        """Doha Gold Line bus-bridge case has line_suspension + bus/staff advisory."""
        _, _, name, _ = system_setup
        if name != "doha":
            return
        matches = [c for c in cat_c_cases if c.get("disruption_type", "") == "ls_gold-maintenance"]
        assert matches, "Doha should have gold maintenance bus-bridge case"
        case = matches[0]
        must = case["ground_truth"]["post_disruption"].get("advisory_must_mention", [])
        assert "bus" in must and "staff" in must

    def test_cat_e_doha_family_and_gold_variants(self, cat_e_cases, system_setup):
        """Doha Cat E has the v23 family carriage redirect + gold-class-on-red variants."""
        _, _, name, _ = system_setup
        if name != "doha":
            return
        ids = {c.get("cultural_id") for c in cat_e_cases}
        assert "family-carriage-redirect" in ids
        assert "gold-class-on-red" in ids
