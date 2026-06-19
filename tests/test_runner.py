"""Tests for harness.runner — pure function tests (no LLM, no network)."""

import json
import pytest
from pathlib import Path

from harness.runner import BenchmarkRunner


@pytest.fixture
def marta_runner():
    return BenchmarkRunner(
        llm_base_url="http://localhost:1234/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        mock_server_url="http://localhost:8100",
        system_name="marta",
    )


@pytest.fixture
def doha_runner():
    return BenchmarkRunner(
        llm_base_url="http://localhost:1234/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        mock_server_url="http://localhost:8100",
        system_name="doha",
    )


@pytest.fixture
def bart_runner():
    return BenchmarkRunner(
        llm_base_url="http://localhost:1234/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        mock_server_url="http://localhost:8100",
        system_name="bart",
    )


@pytest.fixture
def taipei_runner():
    return BenchmarkRunner(
        llm_base_url="http://localhost:1234/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        mock_server_url="http://localhost:8100",
        system_name="taipei",
    )


@pytest.fixture
def cta_runner():
    return BenchmarkRunner(
        llm_base_url="http://localhost:1234/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        mock_server_url="http://localhost:8100",
        system_name="cta",
    )


# ===== System prompt construction =====

class TestBuildSystemPrompt:
    def test_marta_includes_breeze_card(self, marta_runner):
        prompt = marta_runner._build_system_prompt()
        assert "Breeze Card" in prompt

    def test_marta_includes_dollar(self, marta_runner):
        prompt = marta_runner._build_system_prompt()
        assert "$2.50" in prompt

    def test_doha_includes_travel_card(self, doha_runner):
        prompt = doha_runner._build_system_prompt()
        assert "Travel Card" in prompt

    def test_doha_includes_qr(self, doha_runner):
        prompt = doha_runner._build_system_prompt()
        assert "QR" in prompt

    def test_doha_includes_gold_class(self, doha_runner):
        prompt = doha_runner._build_system_prompt()
        assert "gold_class" in prompt
        assert "QR10" in prompt

    def test_bart_includes_clipper_card(self, bart_runner):
        prompt = bart_runner._build_system_prompt()
        assert "Clipper Card" in prompt

    def test_bart_distance_model(self, bart_runner):
        prompt = bart_runner._build_system_prompt()
        assert '"distance"' in prompt
        assert "2.15" in prompt

    def test_bart_includes_surcharges(self, bart_runner):
        prompt = bart_runner._build_system_prompt()
        assert "Transbay" in prompt
        assert "SFO" in prompt

    def test_bart_includes_fare_brackets(self, bart_runner):
        prompt = bart_runner._build_system_prompt()
        assert "fare_brackets" in prompt
        assert '"max_miles": 6' in prompt

    def test_bart_senior_percentage(self, bart_runner):
        prompt = bart_runner._build_system_prompt()
        assert "0.375" in prompt  # multiplier in JSON

    def test_taipei_includes_easycard(self, taipei_runner):
        prompt = taipei_runner._build_system_prompt()
        assert "EasyCard" in prompt

    def test_taipei_includes_ntd(self, taipei_runner):
        prompt = taipei_runner._build_system_prompt()
        assert "NT$" in prompt

    def test_taipei_distance_model(self, taipei_runner):
        prompt = taipei_runner._build_system_prompt()
        assert '"distance"' in prompt
        assert "NT$20" in prompt

    def test_taipei_includes_cultural_notes(self, taipei_runner):
        prompt = taipei_runner._build_system_prompt()
        assert "No eating" in prompt or "no eating" in prompt

    def test_cta_includes_ventra_card(self, cta_runner):
        prompt = cta_runner._build_system_prompt()
        assert "Ventra Card" in prompt

    def test_cta_includes_dollar(self, cta_runner):
        prompt = cta_runner._build_system_prompt()
        assert "$2.50" in prompt

    def test_cta_flat_with_exceptions_model(self, cta_runner):
        prompt = cta_runner._build_system_prompt()
        assert '"flat_with_exceptions"' in prompt

    def test_cta_includes_ohare_override(self, cta_runner):
        prompt = cta_runner._build_system_prompt()
        assert "5.00" in prompt

    def test_cta_includes_cultural_notes(self, cta_runner):
        prompt = cta_runner._build_system_prompt()
        assert "Loop" in prompt

    def test_disruption_section_absent_without_disruptions(self, marta_runner):
        case = {"events": [], "system_context": {"active_disruptions": []}}
        prompt = marta_runner._build_system_prompt(case)
        assert "Disruption Handling" not in prompt

    def test_disruption_section_present_with_disruptions(self, marta_runner):
        case = {
            "events": [],
            "system_context": {
                "active_disruptions": [{"type": "station_closure"}],
            },
        }
        prompt = marta_runner._build_system_prompt(case)
        assert "Disruption Handling" in prompt
        assert "disruption_feed" in prompt

    def test_accessibility_section_absent_without_mode(self, marta_runner):
        case = {"events": [], "system_context": {}}
        prompt = marta_runner._build_system_prompt(case)
        assert "## Accessibility\n" not in prompt

    def test_accessibility_section_present_with_mode(self, marta_runner):
        case = {"events": [], "system_context": {"accessibility_mode": True}}
        prompt = marta_runner._build_system_prompt(case)
        assert "Accessibility" in prompt
        assert "station_info" in prompt

    def test_policy_section_absent_without_policy(self, marta_runner):
        case = {"events": [], "system_context": {}}
        prompt = marta_runner._build_system_prompt(case)
        assert "Policy Update" not in prompt

    def test_policy_section_present_with_policy(self, marta_runner):
        case = {
            "events": [],
            "system_context": {
                "policy_change": {
                    "text": "EFFECTIVE TODAY: Senior citizens (65+) ride free.",
                    "policy_id": "seniors_free",
                },
            },
        }
        prompt = marta_runner._build_system_prompt(case)
        assert "Policy Update" in prompt
        assert "Senior citizens (65+) ride free" in prompt
        assert "supersedes standard fare rules" in prompt


# ===== User message construction =====

class TestBuildUserMessage:
    def test_station_selected(self, marta_runner):
        case = {
            "events": [
                {"type": "station_selected", "field": "origin", "value": "Airport"},
                {"type": "station_selected", "field": "destination", "value": "Five Points"},
            ],
        }
        msg = marta_runner._build_user_message(case)
        assert "Origin: Airport" in msg
        assert "Destination: Five Points" in msg

    def test_passenger_count(self, marta_runner):
        case = {
            "events": [
                {"type": "passenger_count_changed", "adults": 2, "children": 1},
            ],
        }
        msg = marta_runner._build_user_message(case)
        assert "Passengers:" in msg
        assert "2 adults" in msg
        assert "1 children" in msg

    def test_freetext_input(self, marta_runner):
        case = {
            "events": [
                {"type": "freetext_input", "text": "I need wheelchair access."},
            ],
        }
        msg = marta_runner._build_user_message(case)
        assert "I need wheelchair access." in msg

    def test_payment_method_selected(self, doha_runner):
        case = {
            "events": [
                {"type": "payment_method_selected", "method": "gold_travel_card"},
            ],
        }
        msg = doha_runner._build_user_message(case)
        assert "Payment method:" in msg
        assert "Gold Travel Card" in msg

    def test_disruption_update(self, marta_runner):
        case = {
            "events": [
                {
                    "type": "disruption_update",
                    "disruption": {"message": "Five Points closed"},
                },
            ],
        }
        msg = marta_runner._build_user_message(case)
        assert "DISRUPTION ALERT" in msg
        assert "Five Points closed" in msg


# ===== Knowledge base prompt section =====

class TestKnowledgeBasePrompt:
    def test_knowledge_query_section_present(self, marta_runner):
        case = {"events": [], "system_context": {"knowledge_query": True}}
        prompt = marta_runner._build_system_prompt(case)
        assert "knowledge_base" in prompt.lower() or "Knowledge Base" in prompt

    def test_knowledge_query_section_absent(self, marta_runner):
        case = {"events": [], "system_context": {}}
        prompt = marta_runner._build_system_prompt(case)
        assert "Knowledge Base" not in prompt


# ===== Temporal prompt section =====

class TestTemporalPrompt:
    def test_operating_hours_always_present(self, marta_runner):
        """Operating hours should appear in system prompt even without temporal context."""
        prompt = marta_runner._build_system_prompt()
        assert "Service Hours" in prompt
        assert "05:00-01:00" in prompt

    def test_operating_hours_cta_24h(self, cta_runner):
        prompt = cta_runner._build_system_prompt()
        assert "twenty_four_hour_lines" in prompt
        assert "Red Line" in prompt

    def test_temporal_context_present(self, marta_runner):
        case = {
            "events": [],
            "system_context": {
                "temporal_context": {
                    "current_time": "2026-03-10T23:30:00",
                    "day_of_week": "Wednesday",
                    "notes": "Operating hours: 05:00-01:00",
                },
            },
        }
        prompt = marta_runner._build_system_prompt(case)
        assert "Current Time & Service Hours" in prompt
        assert "23:30" in prompt
        assert "Wednesday" in prompt

    def test_temporal_context_absent(self, marta_runner):
        case = {"events": [], "system_context": {}}
        prompt = marta_runner._build_system_prompt(case)
        assert "Current Time & Service Hours" not in prompt


# ===== Default configuration =====

class TestDefaults:
    def test_max_tool_rounds_default(self):
        runner = BenchmarkRunner(
            llm_base_url="http://localhost:1234/v1",
            llm_api_key="test",
            llm_model="test",
            mock_server_url="http://localhost:8100",
            system_name="marta",
        )
        assert runner.max_tool_rounds == 20


# ===== Policies loading =====

class TestPoliciesLoading:
    def test_nested_policies_json_structure(self):
        """policies.json has {"system": ..., "policies": [...]}, server must extract the list."""
        systems_dir = Path(__file__).resolve().parent.parent / "data" / "systems"
        for system_dir in systems_dir.iterdir():
            policies_path = system_dir / "policies.json"
            if not policies_path.exists():
                continue
            with open(policies_path) as f:
                raw = json.load(f)
            # Simulate what mock_server does after the fix
            policies = raw["policies"] if isinstance(raw, dict) and "policies" in raw else raw
            assert isinstance(policies, list), f"{system_dir.name}: policies should be a list"
            assert len(policies) > 0, f"{system_dir.name}: no policies"
            for p in policies:
                assert "policy_id" in p, f"{system_dir.name}: policy missing policy_id"
                assert "synonyms" in p, f"{system_dir.name}: policy missing synonyms"
