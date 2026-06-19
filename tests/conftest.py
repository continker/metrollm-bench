"""Shared fixtures for MetroLLM-Bench test suite."""

import pytest
from pathlib import Path

from harness.graph import MetroGraph
from harness.fares import FareCalculator
from harness.scorer import Scorer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# System name parametrize fixture
# ---------------------------------------------------------------------------

@pytest.fixture(params=["marta", "doha", "bart", "taipei", "cta"])
def system_name(request):
    return request.param


@pytest.fixture(params=["marta", "doha", "bart", "taipei", "cta"])
def system_dir(request):
    return PROJECT_ROOT / "data" / "systems" / request.param


# ---------------------------------------------------------------------------
# Per-system graphs (session-scoped, loaded once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def marta_graph():
    return MetroGraph(PROJECT_ROOT / "data" / "systems" / "marta")


@pytest.fixture(scope="session")
def doha_graph():
    return MetroGraph(PROJECT_ROOT / "data" / "systems" / "doha")


@pytest.fixture(scope="session")
def bart_graph():
    return MetroGraph(PROJECT_ROOT / "data" / "systems" / "bart")


@pytest.fixture(scope="session")
def taipei_graph():
    return MetroGraph(PROJECT_ROOT / "data" / "systems" / "taipei")


@pytest.fixture(scope="session")
def cta_graph():
    return MetroGraph(PROJECT_ROOT / "data" / "systems" / "cta")


@pytest.fixture
def graph(system_name, marta_graph, doha_graph, bart_graph, taipei_graph, cta_graph):
    return {"marta": marta_graph, "doha": doha_graph, "bart": bart_graph, "taipei": taipei_graph, "cta": cta_graph}[system_name]


# ---------------------------------------------------------------------------
# Per-system fare calculators (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def marta_fare_calc():
    return FareCalculator(PROJECT_ROOT / "data" / "systems" / "marta")


@pytest.fixture(scope="session")
def doha_fare_calc():
    return FareCalculator(PROJECT_ROOT / "data" / "systems" / "doha")


@pytest.fixture(scope="session")
def bart_fare_calc():
    return FareCalculator(PROJECT_ROOT / "data" / "systems" / "bart")


@pytest.fixture(scope="session")
def taipei_fare_calc():
    return FareCalculator(PROJECT_ROOT / "data" / "systems" / "taipei")


@pytest.fixture(scope="session")
def cta_fare_calc():
    return FareCalculator(PROJECT_ROOT / "data" / "systems" / "cta")


@pytest.fixture
def fare_calc(system_name, marta_fare_calc, doha_fare_calc, bart_fare_calc, taipei_fare_calc, cta_fare_calc):
    return {"marta": marta_fare_calc, "doha": doha_fare_calc, "bart": bart_fare_calc, "taipei": taipei_fare_calc, "cta": cta_fare_calc}[system_name]


# ---------------------------------------------------------------------------
# Stub judge for tests (returns full marks — tests focus on tier 1 scoring)
# ---------------------------------------------------------------------------

class _StubJudge:
    """Returns full marks for all tier 2 rubrics so tests can focus on tier 1."""
    def score_advisory_content(self, result, case):
        return 10, "stub"
    def score_policy_acknowledged(self, result, case):
        return 10, "stub"
    def score_cultural_accuracy(self, result, case):
        return 10, "stub"
    def score_safety_response(self, result, case):
        return case.get("scoring", {}).get("safety_response_quality", 15), "stub"
    def score_no_fabrication(self, result, case, system_context=None):
        return 10, "stub"
    def score_temporal_accuracy(self, result, case):
        return 10, "stub"
    def score_scope_adherence(self, result, case):
        return 5, "stub"

_stub_judge = _StubJudge()

# ---------------------------------------------------------------------------
# Per-system scorers (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def marta_scorer():
    return Scorer("marta", judge=_stub_judge)


@pytest.fixture(scope="session")
def doha_scorer():
    return Scorer("doha", judge=_stub_judge)


@pytest.fixture(scope="session")
def bart_scorer():
    return Scorer("bart", judge=_stub_judge)


@pytest.fixture(scope="session")
def taipei_scorer():
    return Scorer("taipei", judge=_stub_judge)


@pytest.fixture(scope="session")
def cta_scorer():
    return Scorer("cta", judge=_stub_judge)


@pytest.fixture
def scorer(system_name, marta_scorer, doha_scorer, bart_scorer, taipei_scorer, cta_scorer):
    return {"marta": marta_scorer, "doha": doha_scorer, "bart": bart_scorer, "taipei": taipei_scorer, "cta": cta_scorer}[system_name]
