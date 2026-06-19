"""Rule-based baseline agent — no LLM, deterministic tool forwarding."""

import asyncio
import hashlib
import json
import time
import argparse
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone

import httpx
import yaml

from harness.runner import CaseResult


class RuleAgent:
    """Scripted agent that calls mock server tools in fixed order."""

    def __init__(self, mock_server_url: str, system_name: str, parallel: int = 4):
        self.mock_server_url = mock_server_url.rstrip("/")
        self.system_name = system_name
        self.parallel = parallel
        self.semaphore = asyncio.Semaphore(parallel)

        # Load framebook for operating hours
        data_dir = Path(__file__).parent.parent / "data" / "systems" / system_name
        with open(data_dir / "framebook.yaml") as f:
            self.framebook = yaml.safe_load(f)

    def _parse_events(self, events: list[dict]) -> dict:
        """Extract structured fields from case events."""
        origin = destination = origin_id = destination_id = None
        adults = children = seniors = disabled = 0
        freetext = None
        payment_method = None
        has_pax = False

        for e in events:
            t = e.get("type", "")
            if t == "station_selected":
                if e.get("field") == "origin":
                    origin = e.get("value")
                    origin_id = e.get("station_id")
                elif e.get("field") == "destination":
                    destination = e.get("value")
                    destination_id = e.get("station_id")
            elif t == "passenger_count_changed":
                adults = e.get("adults", 0)
                children = e.get("children", 0)
                seniors = e.get("seniors", 0)
                disabled = e.get("disabled", 0)
                has_pax = True
            elif t == "freetext_input":
                freetext = e.get("text", "")
            elif t == "payment_method_selected":
                payment_method = e.get("method")

        return {
            "origin": origin,
            "destination": destination,
            "origin_id": origin_id,
            "destination_id": destination_id,
            "adults": adults,
            "children": children,
            "seniors": seniors,
            "disabled": disabled,
            "has_pax": has_pax,
            "freetext": freetext,
            "payment_method": payment_method,
        }

    def _check_service_hours(self, case: dict) -> bool:
        """Check if current time is within service hours. Returns True if service available."""
        temporal = case.get("system_context", {}).get("temporal_context")
        if not temporal:
            return True

        service_available = temporal.get("service_available")
        if service_available is not None:
            return service_available

        # Default: assume service available
        return True

    async def _call_tool(self, client: httpx.AsyncClient, tool_name: str, args: dict, case_id: str) -> tuple[dict | None, str | None]:
        """Call a mock server tool. Returns (result, error)."""
        payload = dict(args)
        if tool_name == "disruption_feed":
            payload["case_id"] = case_id
        try:
            resp = await client.post(f"{self.mock_server_url}/{tool_name}", json=payload, timeout=30.0)
            if resp.status_code >= 400:
                return None, resp.text[:200]
            return resp.json(), None
        except Exception as e:
            return None, str(e)

    def _build_fare_quote(self, fare_result: dict, pax: dict) -> dict:
        """Build fare_quote from fare_calculator result and passenger counts."""
        line_items = []
        for item in fare_result.get("line_items", []):
            # Parse "Adult x2" or "Child x1" style labels
            label = item.get("label", "")
            parts = label.lower().split(" x")
            rider_type = parts[0].strip() if parts else "adult"
            count = int(parts[1]) if len(parts) > 1 else 1
            unit_fare = item["amount"] / count if count > 0 else item["amount"]
            line_items.append({
                "rider_type": rider_type,
                "count": count,
                "unit_fare": round(unit_fare, 2),
                "subtotal": item["amount"],
                "currency": item.get("currency", fare_result.get("currency", "USD")),
            })

        # Count free riders (children under threshold, etc.)
        total_pax = pax["adults"] + pax["children"] + pax["seniors"] + pax["disabled"]
        ticketed = sum(i["count"] for i in line_items)
        free_riders = max(0, total_pax - ticketed)

        return {
            "passenger_summary": {
                "adults": pax["adults"],
                "children": pax["children"],
                "seniors": pax["seniors"],
                "disabled": pax["disabled"],
                "free_riders": free_riders,
            },
            "line_items": line_items,
            "discounts": fare_result.get("discounts", []),
            "total": fare_result["total"],
            "currency": fare_result.get("currency", "USD"),
        }

    def _build_route(self, route_result: dict) -> dict:
        """Build route from route_planner result."""
        return {
            "origin": route_result["stops"][0]["station_name"],
            "destination": route_result["stops"][-1]["station_name"],
            "stops": [s["station_name"] for s in route_result["stops"]],
            "transfers": route_result["transfers"],
            "estimated_minutes": route_result["estimated_minutes"],
            "distance_miles": route_result["distance_miles"],
            "line_sequence": route_result.get("line_sequence", []),
        }

    async def _run_single_case(self, client: httpx.AsyncClient, case: dict) -> CaseResult:
        """Run a single case with rule-based logic."""
        case_id = case["id"]
        start_time = time.monotonic()
        tool_calls_made = []

        # Set disruptions
        active_disruptions = case.get("system_context", {}).get("active_disruptions", [])
        await client.post(
            f"{self.mock_server_url}/set_disruptions",
            json={"case_id": case_id, "disruptions": active_disruptions},
            timeout=5.0,
        )

        # Handle multi-turn: flatten all event groups
        turn_groups = case.get("multi_turn_events")
        if turn_groups:
            all_events = []
            for group in turn_groups:
                all_events.extend(group)
        else:
            all_events = case["events"]

        parsed = self._parse_events(all_events)

        def record_tool(name, args, result, error=None):
            tool_calls_made.append({"name": name, "arguments": args, "result": result, "error": error})

        # --- Decision tree ---

        # 1. No stations → freetext-only (Cat J info queries, Cat H freetext-only)
        if not parsed["origin"] or not parsed["destination"]:
            if parsed["freetext"]:
                # Try knowledge base
                kb_args = {"query": parsed["freetext"], "category": "general"}
                kb_result, kb_err = await self._call_tool(client, "knowledge_base", kb_args, case_id)
                record_tool("knowledge_base", kb_args, kb_result, kb_err)

                if kb_result and kb_result.get("found"):
                    content = kb_result["results"][0]["content"] if kb_result["results"] else ""
                    submit_args = {
                        "outcome": "policy_answer_only",
                        "kiosk_action": {"action": "display_info", "reason_code": "ok"},
                        "assistant_message": content[:300],
                    }
                else:
                    submit_args = {
                        "outcome": "request_declined",
                        "kiosk_action": {"action": "block_purchase", "reason_code": "unsupported_request"},
                        "assistant_message": "This request is outside kiosk capabilities.",
                    }
            else:
                submit_args = {
                    "outcome": "request_declined",
                    "kiosk_action": {"action": "block_purchase", "reason_code": "invalid_request"},
                    "assistant_message": "Please select origin and destination stations.",
                }

            sub_result, sub_err = await self._call_tool(client, "submit_assistant_state", submit_args, case_id)
            record_tool("submit_assistant_state", submit_args, sub_result, sub_err)

            e2e_ms = (time.monotonic() - start_time) * 1000
            return self._make_result(case_id, submit_args, tool_calls_made, e2e_ms)

        # 2. Check service hours (Cat I)
        if not self._check_service_hours(case):
            submit_args = {
                "outcome": "service_unavailable",
                "kiosk_action": {"action": "block_purchase", "reason_code": "no_service"},
                "assistant_message": "Service is not available at the requested time.",
            }
            sub_result, sub_err = await self._call_tool(client, "submit_assistant_state", submit_args, case_id)
            record_tool("submit_assistant_state", submit_args, sub_result, sub_err)

            e2e_ms = (time.monotonic() - start_time) * 1000
            return self._make_result(case_id, submit_args, tool_calls_made, e2e_ms)

        # 3. Call route_planner
        route_args = {"origin": parsed["origin"], "destination": parsed["destination"]}
        route_result, route_err = await self._call_tool(client, "route_planner", route_args, case_id)
        record_tool("route_planner", route_args, route_result, route_err)

        if route_err:
            submit_args = {
                "outcome": "request_declined",
                "kiosk_action": {"action": "block_purchase", "reason_code": "invalid_request"},
                "assistant_message": f"Could not plan route: {route_err[:100]}",
            }
            sub_result, sub_err = await self._call_tool(client, "submit_assistant_state", submit_args, case_id)
            record_tool("submit_assistant_state", submit_args, sub_result, sub_err)

            e2e_ms = (time.monotonic() - start_time) * 1000
            return self._make_result(case_id, submit_args, tool_calls_made, e2e_ms)

        # 4. Call fare_calculator (default 1 adult if no pax specified)
        pax = {
            "adults": parsed["adults"] if parsed["has_pax"] else 1,
            "children": parsed["children"],
            "seniors": parsed["seniors"],
            "disabled": parsed["disabled"],
        }
        fare_args = {
            "route_id": route_result["route_id"],
            "passengers": pax,
            "ticket_type": "single",
        }
        if parsed["payment_method"]:
            fare_args["payment_method"] = parsed["payment_method"]
        fare_result, fare_err = await self._call_tool(client, "fare_calculator", fare_args, case_id)
        record_tool("fare_calculator", fare_args, fare_result, fare_err)

        if fare_err:
            submit_args = {
                "outcome": "request_declined",
                "kiosk_action": {"action": "block_purchase", "reason_code": "invalid_request"},
                "assistant_message": f"Could not calculate fare: {fare_err[:100]}",
            }
            sub_result, sub_err = await self._call_tool(client, "submit_assistant_state", submit_args, case_id)
            record_tool("submit_assistant_state", submit_args, sub_result, sub_err)

            e2e_ms = (time.monotonic() - start_time) * 1000
            return self._make_result(case_id, submit_args, tool_calls_made, e2e_ms)

        route_data = self._build_route(route_result)
        fare_data = self._build_fare_quote(fare_result, pax)

        # 5. Check disruptions
        advisory_banners = []
        outcome = "route_and_fare_ready"
        action = "prompt_purchase"
        reason_code = "ok"

        if active_disruptions:
            dis_args = {"severity_filter": "all"}
            dis_result, dis_err = await self._call_tool(client, "disruption_feed", dis_args, case_id)
            record_tool("disruption_feed", dis_args, dis_result, dis_err)

            if dis_result:
                route_stops = {s["station_id"] for s in route_result["stops"]}
                route_lines = set(route_result.get("line_sequence", []))
                for d in dis_result.get("disruptions", []):
                    affected_stations = set(d.get("segment") or [])
                    affected_line = d.get("line")
                    if affected_stations & route_stops or (affected_line and affected_line in route_lines):
                        advisory_banners.append({
                            "severity": "critical" if d["severity"] == "critical" else "warning",
                            "title": d["type"].replace("_", " ").title(),
                            "body": d["message"],
                        })
                        outcome = "advisory_only"
                        action = "display_info"

        # 6. Check accessibility
        if case.get("system_context", {}).get("accessibility_mode"):
            for stop in route_result["stops"]:
                si_args = {"station_id": stop["station_id"], "query_type": "accessibility"}
                si_result, si_err = await self._call_tool(client, "station_info", si_args, case_id)
                record_tool("station_info", si_args, si_result, si_err)

                if si_result:
                    acc = si_result.get("accessibility", {})
                    issues = []
                    if not acc.get("step_free"):
                        issues.append("not step-free")
                    if not acc.get("elevators"):
                        issues.append("no elevators")
                    if issues:
                        advisory_banners.append({
                            "severity": "warning",
                            "title": f"Accessibility: {stop['station_name']}",
                            "body": f"{stop['station_name']}: {', '.join(issues)}",
                        })
                        outcome = "advisory_only"
                        action = "display_info"
                        reason_code = "accessibility_issue"

        # 7. Build assistant message
        msg_parts = [f"Route: {route_data['origin']} to {route_data['destination']}"]
        msg_parts.append(f"{route_data['transfers']} transfer(s), ~{route_data['estimated_minutes']} min")
        msg_parts.append(f"Fare: {fare_data['total']} {fare_data['currency']}")
        if advisory_banners:
            for b in advisory_banners:
                msg_parts.append(f"{b['severity'].upper()}: {b['body']}")
        assistant_message = ". ".join(msg_parts)

        # 8. Submit
        submit_args = {
            "outcome": outcome,
            "route": route_data,
            "kiosk_action": {"action": action, "reason_code": reason_code},
            "assistant_message": assistant_message,
        }
        if outcome == "route_and_fare_ready":
            submit_args["fare_quote"] = fare_data
        if advisory_banners:
            submit_args["advisory_banners"] = advisory_banners

        sub_result, sub_err = await self._call_tool(client, "submit_assistant_state", submit_args, case_id)
        record_tool("submit_assistant_state", submit_args, sub_result, sub_err)

        e2e_ms = (time.monotonic() - start_time) * 1000
        return self._make_result(case_id, submit_args, tool_calls_made, e2e_ms)

    def _make_result(self, case_id: str, submit_args: dict, tool_calls_made: list, e2e_ms: float) -> CaseResult:
        """Build CaseResult matching runner output format."""
        parsed = {
            "outcome": submit_args.get("outcome", ""),
            "kiosk_action": submit_args.get("kiosk_action", {}),
            "reasoning": "",
            "ui_updates": {
                "route": submit_args.get("route"),
                "fare_quote": submit_args.get("fare_quote"),
                "advisory_banners": submit_args.get("advisory_banners", []),
                "assistant_message": submit_args.get("assistant_message", ""),
            },
        }
        return CaseResult(
            case_id=case_id,
            response=parsed,
            tool_calls_made=tool_calls_made,
            raw_content=json.dumps(submit_args),
            reasoning_content="",
            messages=[],
            ttft_ms=0.0,
            e2e_ms=round(e2e_ms, 1),
            input_tokens=0,
            output_tokens=0,
            api_rounds=0,
            error=None,
        )

    async def run(self, cases: list[dict]) -> list[CaseResult]:
        """Run all cases through the rule-based agent."""
        async with httpx.AsyncClient() as client:
            tasks = [self._run_with_semaphore(client, case) for case in cases]
            return await asyncio.gather(*tasks)

    async def _run_with_semaphore(self, client: httpx.AsyncClient, case: dict) -> CaseResult:
        async with self.semaphore:
            try:
                return await self._run_single_case(client, case)
            except Exception as e:
                return CaseResult(
                    case_id=case["id"],
                    response=None,
                    tool_calls_made=[],
                    raw_content="",
                    reasoning_content="",
                    messages=[],
                    ttft_ms=0.0,
                    e2e_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    api_rounds=0,
                    error=str(e),
                )


def main():
    parser = argparse.ArgumentParser(description="Rule-based baseline agent")
    parser.add_argument("--cases", required=True, help="Path to cases JSON")
    parser.add_argument("--system", default="marta", help="Transit system name")
    parser.add_argument("--mock-url", default="http://localhost:8100", help="Mock server URL")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel requests")
    parser.add_argument("--output", default=None, help="Output path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    args = parser.parse_args()

    with open(args.cases) as f:
        cases = json.load(f)
    if args.limit:
        cases = cases[:args.limit]

    print(f"Running {len(cases)} cases with rule-based agent")
    print(f"Mock server: {args.mock_url}, parallel: {args.parallel}")

    agent = RuleAgent(
        mock_server_url=args.mock_url,
        system_name=args.system,
        parallel=args.parallel,
    )

    cases_checksum = hashlib.sha256(Path(args.cases).read_bytes()).hexdigest()[:12]
    started_at = datetime.now(timezone.utc).isoformat()
    results = asyncio.run(agent.run(cases))
    finished_at = datetime.now(timezone.utc).isoformat()

    output = {
        "metadata": {
            "harness_version": "0.4.0",
            "started_at": started_at,
            "finished_at": finished_at,
            "llm_base_url": "rule-based",
            "llm_model": "rule-based",
            "temperature": 0.0,
            "max_tokens": 0,
            "max_tool_rounds": 1,
            "thinking": False,
            "parallel": args.parallel,
            "system": args.system,
            "cases_file": args.cases,
            "cases_checksum_sha256": cases_checksum,
        },
        "model": "rule-based",
        "system": args.system,
        "thinking": False,
        "cases_total": len(cases),
        "cases_succeeded": sum(1 for r in results if r.error is None),
        "cases_failed": sum(1 for r in results if r.error is not None),
        "results": [asdict(r) for r in results],
    }

    output_path = args.output or f"results/{args.system}_rule_based.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {output_path}")
    print(f"  Succeeded: {output['cases_succeeded']}/{output['cases_total']}")
    print(f"  Failed: {output['cases_failed']}/{output['cases_total']}")
    for r in results:
        status = "OK" if r.error is None else f"ERR: {r.error[:60]}"
        print(f"  {r.case_id}: {status} ({len(r.tool_calls_made)} tool calls, {r.e2e_ms:.0f}ms)")


if __name__ == "__main__":
    main()
