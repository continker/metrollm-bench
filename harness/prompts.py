"""Shared system-prompt builder.

Single source of truth for the kiosk system prompt. `harness.runner` and
any external front-end that drives the kiosk LLM build their prompt from
here, so the model sees the same prompt distribution regardless of which
stack drives it.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_system_prompt(
    system_name: str,
    system_context: dict | None = None,
) -> str:
    """Build the kiosk system prompt for `system_name`.

    When `system_context` is provided, the conditional Disruption /
    Accessibility / Policy / Temporal / Knowledge-Base blocks are
    appended exactly as in training, so the model sees the same prompt
    shape whether it is driven by the benchmark runner or by an external
    front-end.
    """
    ctx = system_context or {}
    system_dir = REPO_ROOT / "data" / "systems" / system_name

    with open(system_dir / "framebook.yaml") as f:
        framebook = yaml.safe_load(f)["framebook"]
    with open(system_dir / "fares.json") as f:
        fares = json.load(f)
    with open(system_dir / "lines.json") as f:
        lines = json.load(f)

    currency_symbol = framebook["currency_symbol"]
    currency_code = framebook["currency_code"]
    terminology = framebook["terminology"]
    line_names = ", ".join(l["name"] for l in lines)
    base_fare = fares["base_fare"]
    fare_display = framebook["fare_display_format"]
    fare_model = fares.get("model", "flat")

    prompt = f"""You are a transit kiosk assistant for {framebook['org_name']} ({framebook['full_name']}).

## System Information
- Lines: {line_names}
"""

    fare_rules = {
        "model": fare_model,
        "base_fare": f"{currency_symbol}{base_fare}",
        "currency": currency_code,
        "format": fare_display,
        "payment": [terminology["smartcard"], terminology["contactless"]],
    }
    for k in ("discounts", "fare_brackets", "surcharges", "station_overrides", "payment_methods"):
        if fares.get(k):
            fare_rules[k] = fares[k]
    if "gold_fare" in fares:
        fare_rules["gold_class"] = {
            "fare": f"{currency_symbol}{fares['gold_fare']}",
            "card": terminology.get("smartcard_premium", "Gold Card"),
        }
    prompt += f"- Fare rules: {json.dumps(fare_rules)}\n"
    prompt += f"- Respond in English (the local language is {framebook['primary_language']})\n"

    cultural_notes = framebook.get("cultural_notes", [])
    if cultural_notes:
        prompt += "\n## Cultural Notes\n"
        for note in cultural_notes:
            prompt += f"- {note}\n"

    operating_hours = framebook.get("operating_hours", {})
    if operating_hours:
        prompt += f"\n## Service Hours\n{json.dumps(operating_hours)}\n"

    temporal_ctx = ctx.get("temporal_context")
    if temporal_ctx:
        prompt += "\n## Current Time & Service Hours\n"
        prompt += f"- Current time: {temporal_ctx['current_time']}\n"
        if temporal_ctx.get("day_of_week"):
            prompt += f"- Day: {temporal_ctx['day_of_week']}\n"
        if temporal_ctx.get("notes"):
            prompt += f"- {temporal_ctx['notes']}\n"
        prompt += "- Check whether the requested journey falls within service hours and warn the passenger if not\n"
        prompt += "- Consider headway frequency at the requested time\n"

    prompt += f"""
## Your Role
You help passengers plan trips, calculate fares, and provide station information.
Use the available tools to look up routes, calculate fares, and get station details.
Always use tools rather than guessing — do not fabricate route or fare information.

## Workflow
1. Use route_planner, fare_calculator, station_info to gather information
2. When you have all the information needed, call submit_assistant_state with your final kiosk state
3. You MUST always finish by calling submit_assistant_state — never respond with plain text
4. Set the outcome field to indicate the result: route_and_fare_ready (normal trip), advisory_only (disrupted but route shown), service_unavailable (no service), request_declined (invalid request), or policy_answer_only (info-only)
5. Set kiosk_action to indicate what the kiosk should do: prompt_purchase (ready to buy), display_info (information only), block_purchase (cannot proceed), or refer_to_staff (need human help)

## Reason Code Semantics
- Use `ok` when the kiosk can complete the request normally
- Use `no_service` when service is unavailable for the requested trip or time
- Use `invalid_request` when the request is contradictory or impossible as asked
- Use `unsupported_request` when the question is outside kiosk capabilities
- Use `accessibility_issue` when the route does not satisfy the passenger's stated accessibility requirement
- Use `policy_exception` when a special policy changes the normal fare or purchase flow and that exception should be surfaced

## Advisory Banners
advisory_banners is a primary passenger-facing information channel. Use it to surface important context alongside the route and fare. Severity levels:
- `critical`: service unavailable, block_purchase required, safety issue
- `warning`: disruption affecting the route, accessibility concern, approaching last train
- `info`: security/ID rules, payment requirements, operating-hour reminders, policy context, station-specific notes, late-night service info
- `positive`: a discount, exception, or pass applied in the passenger's favor

Write banners that are specific to this trip — reference affected stations, specific times, or exact policy items from the system prompt. Avoid generic boilerplate. Multiple banners are fine when they address distinct concerns.

## Rules
- Use {terminology['smartcard']} (not "metro card" or other names)
- Fare totals must be numbers (2.50), not strings ("{currency_symbol}2.50")
- Line names in line_sequence must be lowercase (e.g. "red", not "Red")
- Pass route_planner stop objects directly into route.stops (each with station_id, station_name, line, is_transfer)
- If submit_assistant_state returns an error, fix the issues and call it again
- Include fare_quote with passenger_summary and line_items when outcome is route_and_fare_ready
"""

    if ctx.get("active_disruptions"):
        prompt += """
## Disruption Handling
- A DISRUPTION ALERT is included in the passenger query — use the disruption_feed tool to get current service status
- Check if the planned route passes through any affected segments or stations
- Include advisory_banners in your submit_assistant_state with the appropriate severity (critical, warning, or info)
- If the route is affected, warn the passenger and suggest alternatives if available
- If the disruption makes the route unusable, set outcome to service_unavailable and kiosk_action to block_purchase
- When a disruption describes an entire line or a named segment between two stations, call line_info to resolve the topology and encode the closure via route_planner's line_closures parameter (do not enumerate individual stations in station_restrictions)
- If multiple lines are disrupted, pass all of them to line_info's `lines` array in a single call rather than issuing one request per line
"""

    if ctx.get("accessibility_mode"):
        prompt += """
## Accessibility
- The passenger has indicated an accessibility requirement
- Use the station_info tool with query_type "accessibility" to check stations along the route
- Check EACH station on the route for elevator and step-free access
- If any station has an accessibility issue (e.g. elevator out of service), warn the passenger in your advisory_banners
- Include the affected station name and the specific issue in the advisory
"""

    policy_change = ctx.get("policy_change")
    if policy_change:
        prompt += "\n## Policy Update\n"
        prompt += "IMPORTANT: The following policy is in effect and supersedes standard fare rules.\n\n"
        prompt += policy_change["text"] + "\n\n"
        prompt += "Apply this policy when calculating fares. If fare_calculator returns a fare based on old rules, adjust the total in submit_assistant_state.\n"

    policies_path = system_dir / "policies.json"
    if policies_path.exists():
        with open(policies_path) as f:
            policies_data = json.load(f)
        policy_list = policies_data.get("policies", policies_data) if isinstance(policies_data, dict) else policies_data
        if policy_list:
            prompt += "\n## Available Policies\n"
            for p in policy_list:
                prompt += f"- [{p['policy_id']}] {p['title']}\n"
            prompt += "Use knowledge_base with policy_id for exact lookup.\n"

    if ctx.get("knowledge_query"):
        prompt += """
## Knowledge Base
- The passenger has a question about transit policies or service information
- Use the knowledge_base tool with the appropriate policy_id to look up relevant policies
- If the passenger asks about multiple topics, make separate knowledge_base calls for each
- If you are unsure which policy applies, use the query parameter to search
- Include the relevant policy information in your submit_assistant_state
- If no matching policies are found, provide a helpful general response
"""

    return prompt


def build_user_message(events: list[dict]) -> str:
    """Convert a list of case events into the user message string.

    Same format the runner uses for benchmark cases — one line per event,
    joined by newlines. The LoRA was trained on this shape, so HF Space
    and /simulate must produce identical user messages for equivalent
    scenarios.

    Recognized event types:
      station_selected {field, value}        → "Origin: X" / "Destination: X"
      passenger_count_changed {adults?, …}   → "Passengers: 2 adults, 1 children"
      freetext_input {text}                  → "<text>"
      payment_method_selected {method}       → "Payment method: Gold Travel Card"
      disruption_update {disruption.message} → "⚠ DISRUPTION ALERT: <msg>"
    """
    parts: list[str] = []
    for event in events:
        kind = event.get("type")
        if kind == "station_selected":
            parts.append(f"{event['field'].title()}: {event['value']}")
        elif kind == "passenger_count_changed":
            pax_parts = []
            for key in ["adults", "children", "seniors", "disabled"]:
                if key in event and event[key] != 0:
                    pax_parts.append(f"{event[key]} {key}")
            parts.append(f"Passengers: {', '.join(pax_parts)}")
        elif kind == "freetext_input":
            parts.append(event["text"])
        elif kind == "payment_method_selected":
            parts.append(f"Payment method: {event['method'].replace('_', ' ').title()}")
        elif kind == "disruption_update":
            disruption = event.get("disruption", {})
            msg = disruption.get("message", "Service disruption in effect")
            parts.append(f"⚠ DISRUPTION ALERT: {msg}")
    return "\n".join(parts)
