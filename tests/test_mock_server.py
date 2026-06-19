"""Tests for harness.mock_server — multi-system routing, lazy loading, case isolation."""

import pytest
from pathlib import Path

from fastapi.testclient import TestClient

from harness.mock_server import (
    _SystemData,
    _load_system,
    _system_for_case,
    _systems,
    _case_system,
    _system_name,
    _DATA_ROOT,
    RoutePlannerRequest,
    FareCalculatorRequest,
    StationInfoRequest,
    KnowledgeBaseRequest,
    SubmitAssistantStateRequest,
    DisruptionFeedRequest,
    LineInfoRequest,
    LineClosure,
    app,
)

import harness.mock_server as _mod


# ---------------------------------------------------------------------------
# Fixture: clean module state between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_module_state():
    """Reset module-level caches so tests don't leak state."""
    saved_systems = dict(_mod._systems)
    saved_case_system = dict(_mod._case_system)
    saved_name = _mod._system_name
    yield
    _mod._systems.clear()
    _mod._systems.update(saved_systems)
    _mod._case_system.clear()
    _mod._case_system.update(saved_case_system)
    _mod._system_name = saved_name


# ---------------------------------------------------------------------------
# _SystemData
# ---------------------------------------------------------------------------

class TestSystemData:
    def test_dataclass_fields(self):
        """_SystemData has expected fields with route_cache defaulting to empty dict."""
        sd = _load_system("marta")
        assert isinstance(sd.metro, object)
        assert isinstance(sd.fares, object)
        assert isinstance(sd.policies, list)
        assert isinstance(sd.line_alias, dict)
        assert isinstance(sd.route_cache, dict)

    def test_route_cache_default_empty(self):
        """Fresh system has empty route_cache."""
        _mod._systems.pop("marta", None)  # force fresh load
        sd = _load_system("marta")
        assert sd.route_cache == {}


# ---------------------------------------------------------------------------
# _load_system — lazy loading + caching
# ---------------------------------------------------------------------------

class TestLoadSystem:
    def test_loads_known_system(self):
        sd = _load_system("marta")
        assert sd.metro is not None
        assert len(sd.policies) > 0

    def test_caches_result(self):
        sd1 = _load_system("marta")
        sd2 = _load_system("marta")
        assert sd1 is sd2

    def test_different_systems_different_data(self):
        marta = _load_system("marta")
        doha = _load_system("doha")
        assert marta is not doha
        assert marta.metro is not doha.metro

    def test_unknown_system_raises(self):
        with pytest.raises(ValueError, match="Unknown system"):
            _load_system("nonexistent_system_xyz")

    @pytest.mark.parametrize("system", ["marta", "doha", "bart", "taipei", "cta", "beijing"])
    def test_all_systems_loadable(self, system):
        sd = _load_system(system)
        assert sd.metro is not None
        assert isinstance(sd.policies, list)
        assert isinstance(sd.line_alias, dict)

    @pytest.mark.parametrize("system", ["marta", "doha", "bart", "taipei", "cta", "beijing"])
    def test_stations_populated(self, system):
        sd = _load_system(system)
        assert len(sd.metro.stations) > 10


# ---------------------------------------------------------------------------
# _system_for_case — case_id routing + fallback
# ---------------------------------------------------------------------------

class TestSystemForCase:
    def test_fallback_to_default(self):
        """When case_id is not registered, falls back to _system_name."""
        _mod._system_name = "marta"
        sd = _system_for_case(None)
        assert sd is _load_system("marta")

    def test_fallback_with_unknown_case_id(self):
        _mod._system_name = "doha"
        sd = _system_for_case("unknown-case-999")
        assert sd is _load_system("doha")

    def test_registered_case_routes_correctly(self):
        _mod._system_name = "marta"
        _mod._case_system["test-case-1"] = "doha"
        sd = _system_for_case("test-case-1")
        assert sd is _load_system("doha")

    def test_different_cases_different_systems(self):
        _mod._case_system["case-marta"] = "marta"
        _mod._case_system["case-doha"] = "doha"
        sd_m = _system_for_case("case-marta")
        sd_d = _system_for_case("case-doha")
        assert sd_m is not sd_d

    def test_empty_string_case_id_uses_default(self):
        _mod._system_name = "bart"
        sd = _system_for_case("")
        assert sd is _load_system("bart")

    def test_no_system_configured_raises(self):
        _mod._system_name = ""
        with pytest.raises(RuntimeError, match="No system configured"):
            _system_for_case(None)


# ---------------------------------------------------------------------------
# Request models — case_id field
# ---------------------------------------------------------------------------

class TestRequestModels:
    def test_route_planner_accepts_case_id(self):
        req = RoutePlannerRequest(origin="A", destination="B", case_id="test-1")
        assert req.case_id == "test-1"

    def test_route_planner_case_id_defaults_none(self):
        req = RoutePlannerRequest(origin="A", destination="B")
        assert req.case_id is None

    def test_fare_calculator_accepts_case_id(self):
        req = FareCalculatorRequest(
            route_id="r1",
            passengers={"adults": 1},
            case_id="test-2",
        )
        assert req.case_id == "test-2"

    def test_station_info_accepts_case_id(self):
        req = StationInfoRequest(station_id="S1", query_type="accessibility", case_id="test-3")
        assert req.case_id == "test-3"

    def test_knowledge_base_accepts_case_id(self):
        req = KnowledgeBaseRequest(query="fares", case_id="test-4")
        assert req.case_id == "test-4"

    def test_submit_state_accepts_case_id(self):
        req = SubmitAssistantStateRequest(
            outcome="route_and_fare_ready",
            kiosk_action={"action": "prompt_purchase", "reason_code": "ok"},
            assistant_message="Hello",
            case_id="test-5",
        )
        assert req.case_id == "test-5"

    def test_disruption_feed_already_has_case_id(self):
        req = DisruptionFeedRequest(case_id="test-6")
        assert req.case_id == "test-6"


# ---------------------------------------------------------------------------
# Integration: per-system data isolation
# ---------------------------------------------------------------------------

class TestSystemIsolation:
    def test_marta_stations_not_in_doha(self):
        marta = _load_system("marta")
        doha = _load_system("doha")
        # MARTA has station IDs like MARTA-AP; Doha doesn't
        marta_ids = set(marta.metro.stations.keys())
        doha_ids = set(doha.metro.stations.keys())
        assert marta_ids.isdisjoint(doha_ids)

    def test_route_caches_isolated(self):
        marta = _load_system("marta")
        doha = _load_system("doha")
        marta.route_cache["test_route"] = {"origin": "A", "destination": "B"}
        assert "test_route" not in doha.route_cache

    @pytest.mark.parametrize("system", ["marta", "doha", "bart", "taipei", "cta", "beijing"])
    def test_policies_are_lists(self, system):
        sd = _load_system(system)
        assert isinstance(sd.policies, list)
        assert all("policy_id" in p for p in sd.policies)


# ---------------------------------------------------------------------------
# /line_info endpoint
# ---------------------------------------------------------------------------

class TestLineInfoEndpoint:
    @pytest.fixture(autouse=True)
    def _beijing_default(self):
        _mod._system_name = "beijing"
        _load_system("beijing")
        yield

    def test_returns_thin_topology(self):
        client = TestClient(app)
        r = client.post("/line_info", json={"line": "10"})
        assert r.status_code == 200
        body = r.json()
        assert body["line_id"] == "10"
        assert body["is_loop"] is True
        assert body["station_count"] == 45
        assert len(body["stations"]) == 45
        # Each station entry has the expected shape
        first = body["stations"][0]
        assert "station_id" in first and "station_name" in first
        assert "position" in first and "connections" in first
        assert isinstance(first["connections"], list)

    def test_resolves_natural_language_alias(self):
        client = TestClient(app)
        r = client.post("/line_info", json={"line": "Line 10"})
        assert r.status_code == 200
        assert r.json()["line_id"] == "10"

    def test_unknown_line_returns_404(self):
        client = TestClient(app)
        r = client.post("/line_info", json={"line": "nonexistent-line"})
        assert r.status_code == 404

    def test_linear_line_has_terminals(self):
        client = TestClient(app)
        r = client.post("/line_info", json={"line": "yanfang"})
        body = r.json()
        assert body["is_loop"] is False
        assert len(body["terminals"]) == 2


# ---------------------------------------------------------------------------
# /route_planner with line_closures
# ---------------------------------------------------------------------------

class TestRoutePlannerLineClosures:
    @pytest.fixture(autouse=True)
    def _beijing_default(self):
        _mod._system_name = "beijing"
        _load_system("beijing")
        yield

    def test_whole_line_closure_reroutes(self):
        client = TestClient(app)
        r = client.post("/route_planner", json={
            "origin": "Yanshan",
            "destination": "Dongbabei",
            "line_closures": [{"line": "10"}],
        })
        assert r.status_code == 200
        body = r.json()
        assert "10" not in body["line_sequence"]
        assert body["transfers"] >= 1

    def test_line_closure_nl_alias_resolves(self):
        """Passing 'Line 10' instead of '10' should still work via the alias map."""
        client = TestClient(app)
        r = client.post("/route_planner", json={
            "origin": "Yanshan",
            "destination": "Dongbabei",
            "line_closures": [{"line": "Line 10"}],
        })
        assert r.status_code == 200
        assert "10" not in r.json()["line_sequence"]

    def test_partial_line_closure_on_loop_errors(self):
        client = TestClient(app)
        r = client.post("/route_planner", json={
            "origin": "Yanshan",
            "destination": "Dongbabei",
            "line_closures": [{"line": "10", "from_station": "Tuanjiehu", "to_station": "Guomao"}],
        })
        assert r.status_code == 404  # partial loop closure is rejected

    def test_line_closure_model_accepts_partial(self):
        """Pydantic model accepts from/to station fields."""
        lc = LineClosure(line="14", from_station="A", to_station="B")
        assert lc.line == "14"
        assert lc.from_station == "A"
        assert lc.to_station == "B"
