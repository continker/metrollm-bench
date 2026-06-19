"""Benchmark runner — sends cases to LLM, handles tool calls via mock server."""

import asyncio
import hashlib
import json
import subprocess
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import httpx
import yaml


@dataclass
class CaseResult:
    case_id: str
    response: dict | None       # parsed LLM response (full message content)
    tool_calls_made: list[dict] # [{name, arguments, result}]
    raw_content: str            # raw text content from LLM
    reasoning_content: str      # thinking trace (Qwen3.5 thinking mode)
    messages: list[dict]        # full conversation transcript
    ttft_ms: float              # time to first token (0 if not streaming)
    e2e_ms: float               # end-to-end time
    input_tokens: int           # sum of prompt_tokens across all rounds (= total billed)
    output_tokens: int          # sum of completion_tokens across all rounds
    api_rounds: int             # number of LLM API calls made
    error: str | None
    empty_rounds: int = 0       # rounds where the API returned no content/tool_calls/reasoning (truncation or vendor hiccup)
    truncated: bool = False     # any round finished with finish_reason="length" (output cut off mid-response)


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "route_planner",
            "description": "Find optimal route between two stations. Supports station restrictions for disruption-aware routing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin station name or ID"},
                    "destination": {"type": "string", "description": "Destination station name or ID"},
                    "departure_time": {"type": "string", "description": "ISO 8601 departure time (optional)"},
                    "accessibility": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Accessibility requirements (optional)"
                    },
                    "station_restrictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "station": {"type": "string", "description": "Station name to restrict"},
                                "restriction": {
                                    "type": "string",
                                    "enum": ["closed", "skip", "no_transfer"],
                                    "description": "closed: no service. skip: trains pass without stopping. no_transfer: cannot change lines."
                                }
                            },
                            "required": ["station", "restriction"]
                        },
                        "description": "Stations with operational restrictions from disruption info"
                    },
                    "segment_closures": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "description": "Pairs of adjacent stations where track is closed"
                    },
                    "line_closures": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "line": {"type": "string", "description": "Line id or name"},
                                "from_station": {"type": "string", "description": "Inclusive start of the closed range (omit both endpoints for whole-line closure)"},
                                "to_station": {"type": "string", "description": "Inclusive end of the closed range"}
                            },
                            "required": ["line"]
                        },
                        "description": "Line-level closures. Omit from_station/to_station to close the entire line. Prefer this over listing individual stations in station_restrictions."
                    }
                },
                "required": ["origin", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fare_calculator",
            "description": "Calculate fare for a journey",
            "parameters": {
                "type": "object",
                "properties": {
                    "route_id": {"type": "string", "description": "Route ID from route_planner"},
                    "passengers": {
                        "type": "object",
                        "properties": {
                            "adults": {"type": "integer"},
                            "children": {"type": "integer"},
                            "seniors": {"type": "integer"},
                            "disabled": {"type": "integer"}
                        }
                    },
                    "ticket_type": {"type": "string", "enum": ["single", "return", "day_pass", "weekly", "monthly"]},
                    "payment_method": {"type": "string", "enum": ["smartcard", "contactless", "cash", "mobile", "gold_travel_card", "clipper_card", "easycard", "ventra", "disposable_ticket"]}
                },
                "required": ["route_id", "passengers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "station_info",
            "description": "Get station facility and accessibility information. Use station_ids to check multiple stations in one call (e.g. all stops on a route).",
            "parameters": {
                "type": "object",
                "properties": {
                    "station_id": {"type": "string", "description": "Single station ID or name"},
                    "station_ids": {"type": "array", "items": {"type": "string"}, "description": "Multiple station IDs to check at once"},
                    "query_type": {
                        "type": "string",
                        "enum": ["accessibility", "facilities", "exits", "connections", "real_time_status"]
                    }
                },
                "required": ["query_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "line_info",
            "description": "Get a line's station sequence, loop/terminal metadata, and per-station transfers (other lines at each station). Use before encoding line-level disruptions so station IDs come from the tool, not from memory. Use lines to look up multiple lines in one call (e.g. when several lines are disrupted).",
            "parameters": {
                "type": "object",
                "properties": {
                    "line": {"type": "string", "description": "Single line id or natural-language name (e.g. \"10\" or \"Line 10\")"},
                    "lines": {"type": "array", "items": {"type": "string"}, "description": "Multiple line ids or names to look up at once (preferred when several lines are impacted)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "disruption_feed",
            "description": "Get current service disruptions and advisories. Call this when a disruption alert is reported to get detailed status information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line": {"type": "string", "description": "Filter by line name (optional)"},
                    "station": {"type": "string", "description": "Filter by station name or ID (optional)"},
                    "severity_filter": {
                        "type": "string",
                        "enum": ["all", "major", "minor"],
                        "description": "Filter by severity level (default: all)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_base",
            "description": "Look up transit policies, FAQ, and service information. Use policy_id for exact lookup (preferred) or query for keyword search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id": {"type": "string", "description": "Exact policy ID from the available policies list"},
                    "query": {"type": "string", "description": "Keyword search query (when policy_id is not known)"},
                    "category": {"type": "string", "description": "Optional category filter"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_assistant_state",
            "description": "Submit the final assistant kiosk state for rendering. You MUST call this tool as your last action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "enum": ["route_and_fare_ready", "advisory_only", "service_unavailable", "request_declined", "policy_answer_only"],
                        "description": "The outcome state of this interaction"
                    },
                    "route": {
                        "type": "object",
                        "description": "Route information. Required when outcome is route_and_fare_ready or advisory_only.",
                        "properties": {
                            "origin": {"type": "string"},
                            "destination": {"type": "string"},
                            "stops": {"type": "array", "items": {
                                "type": "object",
                                "properties": {
                                    "station_id": {"type": "string"},
                                    "station_name": {"type": "string"},
                                    "line": {"type": "string"},
                                    "is_transfer": {"type": "boolean"}
                                },
                                "required": ["station_id"]
                            }, "description": "Stop objects from route_planner result"},
                            "transfers": {"type": "integer"},
                            "estimated_minutes": {"type": "integer"},
                            "distance_miles": {"type": "number"},
                            "line_sequence": {"type": "array", "items": {"type": "string"}, "description": "Line names used in order"}
                        },
                        "required": ["origin", "destination", "stops", "transfers", "estimated_minutes", "distance_miles", "line_sequence"]
                    },
                    "fare_quote": {
                        "type": "object",
                        "description": "Fare breakdown. Required when outcome is route_and_fare_ready.",
                        "properties": {
                            "passenger_summary": {
                                "type": "object",
                                "properties": {
                                    "adults": {"type": "integer", "default": 0},
                                    "children": {"type": "integer", "default": 0},
                                    "seniors": {"type": "integer", "default": 0},
                                    "disabled": {"type": "integer", "default": 0},
                                    "free_riders": {"type": "integer", "default": 0}
                                }
                            },
                            "line_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "rider_type": {"type": "string"},
                                        "count": {"type": "integer"},
                                        "unit_fare": {"type": "number"},
                                        "subtotal": {"type": "number"},
                                        "currency": {"type": "string"}
                                    },
                                    "required": ["rider_type", "count", "unit_fare", "subtotal", "currency"]
                                }
                            },
                            "discounts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "amount": {"type": "number"},
                                        "currency": {"type": "string"}
                                    }
                                }
                            },
                            "total": {"type": "number", "description": "Total fare as a number (e.g. 2.50, NOT '$2.50')"},
                            "currency": {"type": "string"}
                        },
                        "required": ["total", "currency"]
                    },
                    "kiosk_action": {
                        "type": "object",
                        "description": "What the kiosk should do with this state",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["display_info", "prompt_purchase", "block_purchase", "refer_to_staff"]
                            },
                            "reason_code": {
                                "type": "string",
                                "enum": ["ok", "no_service", "invalid_request", "unsupported_request", "accessibility_issue", "policy_exception"]
                            }
                        },
                        "required": ["action", "reason_code"]
                    },
                    "advisory_banners": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "severity": {"type": "string", "enum": ["info", "warning", "critical", "positive"]},
                                "title": {"type": "string"},
                                "body": {"type": "string"}
                            },
                            "required": ["severity", "title", "body"]
                        }
                    },
                    "assistant_message": {
                        "type": "string",
                        "description": "Human-readable message for the kiosk screen"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Internal analysis of the query"
                    }
                },
                "required": ["outcome", "kiosk_action", "assistant_message"]
            }
        }
    }
]


class BenchmarkRunner:
    def __init__(
        self,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        mock_server_url: str,
        system_name: str,
        parallel: int = 2,
        max_tokens: int = 4096,
        thinking: bool = True,
        temperature: float = 0.0,
        max_tool_rounds: int = 20,
        extra_body: dict | None = None,
    ):
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.mock_server_url = mock_server_url.rstrip("/")
        self.system_name = system_name
        self.parallel = parallel
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.temperature = temperature
        self.max_tool_rounds = max_tool_rounds
        self.extra_body = extra_body or {}
        self.semaphore = asyncio.Semaphore(parallel)

    def _build_system_prompt(self, case: dict | None = None) -> str:
        """Thin wrapper around `harness.prompts.build_system_prompt` so the
        runner and the HF Space share one source of truth."""
        from harness.prompts import build_system_prompt
        ctx = case.get("system_context") if case else None
        return build_system_prompt(self.system_name, ctx)

    def _build_user_message(self, case: dict) -> str:
        """Thin wrapper around `harness.prompts.build_user_message` so the
        runner and HF Space format user messages identically."""
        from harness.prompts import build_user_message
        return build_user_message(case["events"])

    async def _call_mock_tool(self, client: httpx.AsyncClient, tool_name: str, arguments: dict, case_id: str | None = None, case: dict | None = None) -> dict:
        """Forward a tool call to the mock server."""
        url = f"{self.mock_server_url}/{tool_name}"
        payload = dict(arguments)
        # Inject case_id so mock server routes to the correct system data.
        if case_id:
            payload["case_id"] = case_id
        # Inject current_time for disruption_feed temporal filtering.
        if tool_name == "disruption_feed" and case is not None:
            current_time = (
                case.get("system_context", {})
                .get("temporal_context", {})
                .get("current_time")
                or case.get("system_context", {}).get("current_time")
            )
            if current_time:
                payload["current_time"] = current_time
        resp = await client.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    async def _run_single_case(self, client: httpx.AsyncClient, case: dict) -> CaseResult:
        """Run a single test case against the LLM."""
        case_id = case["id"]
        system_prompt = self._build_system_prompt(case)
        user_message = self._build_user_message(case)

        # Multi-turn support: Cat G sends events in phases
        turn_groups = case.get("multi_turn_events")
        if turn_groups:
            first_msg = self._build_user_message({"events": turn_groups[0]})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": first_msg},
            ]
            remaining_turns = list(turn_groups[1:])
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            remaining_turns = []

        # Set active disruptions on mock server for this case (keyed by case_id)
        active_disruptions = case.get("system_context", {}).get("active_disruptions", [])
        await client.post(
            f"{self.mock_server_url}/set_disruptions",
            json={"case_id": case_id, "system": self.system_name, "disruptions": active_disruptions},
            timeout=5.0,
        )

        tool_calls_made = []
        total_input_tokens = 0
        total_output_tokens = 0
        api_rounds = 0
        first_token_ms = 0.0
        empty_rounds = 0
        truncated = False
        # Consecutive-empty cap: legitimate transient empty responses
        # (Azure / OpenAI vendor hiccups) are rare and resolve within 1–2
        # retries. Beyond this we treat the empty stream as a structural
        # problem (token budget too small, model in a stuck state) and
        # surface as an error rather than silently burning the round budget.
        MAX_CONSECUTIVE_EMPTY = 2
        consecutive_empty = 0

        start_time = time.monotonic()

        # Azure OpenAI: URL like https://{resource}.cognitiveservices.azure.com/openai/deployments/{deployment}?api-version=X
        # Use api-key header, preserve query string when appending /chat/completions
        is_azure = "azure.com" in self.llm_base_url
        if is_azure:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.llm_base_url)
            new_path = parsed.path.rstrip("/") + "/chat/completions"
            chat_endpoint = urlunparse(parsed._replace(path=new_path))
            request_headers = {"api-key": self.llm_api_key}
        else:
            chat_endpoint = f"{self.llm_base_url}/chat/completions"
            request_headers = {"Authorization": f"Bearer {self.llm_api_key}"}

        try:
            for round_num in range(self.max_tool_rounds):
                # llama-server, OpenAI GPT-5+, and Azure OpenAI need max_completion_tokens
                use_completion = (
                    "localhost" in self.llm_base_url
                    or "127.0.0.1" in self.llm_base_url
                    or "api.openai.com" in self.llm_base_url
                    or is_azure
                )
                token_limit_key = "max_completion_tokens" if use_completion else "max_tokens"
                request_body = {
                    "model": self.llm_model,
                    "messages": messages,
                    "tools": TOOL_DEFINITIONS,
                    token_limit_key: self.max_tokens,
                }
                if self.temperature is not None:
                    request_body["temperature"] = self.temperature
                # GPT-5 family (direct or via Azure) takes reasoning_effort instead of
                # thinking-style controls; medium keeps parity with v22 GPT-5-mini runs.
                if is_azure or (self.llm_model or "").startswith("gpt-5"):
                    request_body["reasoning_effort"] = "medium"
                # llama-server specific: disable thinking mode via chat_template_kwargs
                if not self.thinking and ("localhost" in self.llm_base_url or "127.0.0.1" in self.llm_base_url):
                    request_body["chat_template_kwargs"] = {"enable_thinking": False}

                # Caller-supplied extra body fields, shallow-merged; caller wins on key collisions.
                if self.extra_body:
                    request_body.update(self.extra_body)

                # Retry with backoff on 429 rate limits
                for attempt in range(5):
                    resp = await client.post(
                        chat_endpoint,
                        headers=request_headers,
                        json=request_body,
                        timeout=240.0,
                    )
                    if resp.status_code == 429 and attempt < 4:
                        wait = 2 ** attempt  # 1, 2, 4, 8s
                        await asyncio.sleep(wait)
                        continue
                    break
                if resp.status_code >= 400:
                    error_detail = resp.text[:500]
                    raise httpx.HTTPStatusError(
                        f"{resp.status_code}: {error_detail}",
                        request=resp.request,
                        response=resp,
                    )
                result = resp.json()

                if api_rounds == 0:
                    first_token_ms = resp.elapsed.total_seconds() * 1000

                choice = result["choices"][0]
                message = choice["message"]
                finish_reason = choice.get("finish_reason", "")

                usage = result.get("usage", {})
                total_input_tokens += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)
                api_rounds += 1

                # If the model made tool calls, forward them
                if message.get("tool_calls"):
                    messages.append(message)  # add assistant message with tool calls

                    submitted = None
                    for tc in message["tool_calls"]:
                        fn_name = tc["function"]["name"]
                        fn_args = json.loads(tc["function"]["arguments"])

                        try:
                            tool_result = await self._call_mock_tool(client, fn_name, fn_args, case_id=case_id, case=case)
                            tool_calls_made.append({
                                "name": fn_name,
                                "arguments": fn_args,
                                "result": tool_result,
                                "error": None,
                            })
                            # If submit_assistant_state was accepted, capture it
                            if fn_name == "submit_assistant_state" and tool_result.get("accepted"):
                                submitted = fn_args
                        except httpx.HTTPStatusError as e:
                            # Validation error from mock server (422) — send error back to model
                            error_body = e.response.text
                            tool_result = {"error": error_body}
                            tool_calls_made.append({
                                "name": fn_name,
                                "arguments": fn_args,
                                "result": None,
                                "error": error_body,
                            })
                        except Exception as e:
                            tool_result = {"error": str(e)}
                            tool_calls_made.append({
                                "name": fn_name,
                                "arguments": fn_args,
                                "result": None,
                                "error": str(e),
                            })

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(tool_result),
                        })

                    # If submit_assistant_state was accepted, check for remaining turns
                    if submitted is not None:
                        if remaining_turns:
                            # Inject next turn's events as new user message
                            next_events = remaining_turns.pop(0)
                            next_msg = self._build_user_message({"events": next_events})
                            messages.append({"role": "user", "content": next_msg})
                            submitted = None
                            continue

                        e2e_ms = (time.monotonic() - start_time) * 1000
                        reasoning = message.get("reasoning_content", "")
                        # Reshape submit_assistant_state args into the scoring format
                        parsed = {
                            "outcome": submitted.get("outcome", ""),
                            "kiosk_action": submitted.get("kiosk_action", {}),
                            "reasoning": submitted.get("reasoning", ""),
                            "ui_updates": {
                                "route": submitted.get("route"),
                                "fare_quote": submitted.get("fare_quote"),
                                "advisory_banners": submitted.get("advisory_banners", []),
                                "assistant_message": submitted.get("assistant_message", ""),
                            },
                        }
                        return CaseResult(
                            case_id=case_id,
                            response=parsed,
                            tool_calls_made=tool_calls_made,
                            raw_content=json.dumps(submitted),
                            reasoning_content=reasoning,
                            messages=messages,
                            ttft_ms=round(first_token_ms, 1),
                            e2e_ms=round(e2e_ms, 1),
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            api_rounds=api_rounds,
                            error=None,
                            empty_rounds=empty_rounds,
                            truncated=truncated,
                        )

                    continue  # next round (submit_assistant_state not yet called, or was rejected)

                # No tool calls — model responded with plain text
                raw_content = message.get("content", "") or ""
                reasoning = message.get("reasoning_content", "")

                # Multi-turn: if there are remaining turns, treat text or
                # thinking-only response as conversational and inject next turn
                if remaining_turns and (raw_content.strip() or reasoning):
                    messages.append(message)
                    next_events = remaining_turns.pop(0)
                    next_msg = self._build_user_message({"events": next_events})
                    messages.append({"role": "user", "content": next_msg})
                    continue

                # Empty response handling. The model returned no tool_calls,
                # no content, and no reasoning_content. Two distinct causes:
                #   (a) finish_reason="length" — output truncated mid-response
                #       by max_completion_tokens. Retrying with the same prompt
                #       reproduces the truncation deterministically. Surface as
                #       a fatal error so the run config can be fixed.
                #   (b) finish_reason="stop"/other — vendor hiccup. Retry up
                #       to MAX_CONSECUTIVE_EMPTY times, then fail.
                if not raw_content.strip() and not reasoning:
                    empty_rounds += 1
                    consecutive_empty += 1
                    is_truncation = finish_reason == "length"
                    if is_truncation:
                        truncated = True
                    print(
                        f"  [{case_id}] empty round {api_rounds} "
                        f"(finish_reason={finish_reason!r}, "
                        f"completion_tokens={usage.get('completion_tokens', 0)}, "
                        f"consecutive_empty={consecutive_empty})",
                        flush=True,
                    )
                    if is_truncation:
                        e2e_ms = (time.monotonic() - start_time) * 1000
                        return CaseResult(
                            case_id=case_id, response=None,
                            tool_calls_made=tool_calls_made,
                            raw_content="", reasoning_content="",
                            messages=messages,
                            ttft_ms=round(first_token_ms, 1),
                            e2e_ms=round(e2e_ms, 1),
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            api_rounds=api_rounds,
                            error=f"truncated: finish_reason=length on round {api_rounds} "
                                  f"(completion_tokens={usage.get('completion_tokens', 0)} ≈ max_tokens cap); "
                                  f"raise --max-tokens for this model/effort combination",
                            empty_rounds=empty_rounds,
                            truncated=True,
                        )
                    if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                        e2e_ms = (time.monotonic() - start_time) * 1000
                        return CaseResult(
                            case_id=case_id, response=None,
                            tool_calls_made=tool_calls_made,
                            raw_content="", reasoning_content="",
                            messages=messages,
                            ttft_ms=round(first_token_ms, 1),
                            e2e_ms=round(e2e_ms, 1),
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            api_rounds=api_rounds,
                            error=f"empty-response retry exhausted: "
                                  f"{consecutive_empty} consecutive empty rounds "
                                  f"(finish_reason={finish_reason!r})",
                            empty_rounds=empty_rounds,
                            truncated=truncated,
                        )
                    # Retry the same context (within budget)
                    continue
                # Non-empty response — reset consecutive counter
                consecutive_empty = 0

                e2e_ms = (time.monotonic() - start_time) * 1000
                parsed = None
                try:
                    parsed = json.loads(raw_content)
                except (json.JSONDecodeError, TypeError):
                    pass

                return CaseResult(
                    case_id=case_id,
                    response=parsed,
                    tool_calls_made=tool_calls_made,
                    raw_content=raw_content,
                    reasoning_content=reasoning,
                    messages=messages,
                    ttft_ms=round(first_token_ms, 1),
                    e2e_ms=round(e2e_ms, 1),
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    api_rounds=api_rounds,
                    error=None,
                    empty_rounds=empty_rounds,
                    truncated=truncated,
                )

            # Exhausted tool rounds
            e2e_ms = (time.monotonic() - start_time) * 1000
            return CaseResult(
                case_id=case_id, response=None, tool_calls_made=tool_calls_made,
                raw_content="", reasoning_content="", messages=messages,
                ttft_ms=round(first_token_ms, 1), e2e_ms=round(e2e_ms, 1),
                input_tokens=total_input_tokens, output_tokens=total_output_tokens,
                api_rounds=api_rounds,
                error=f"Exhausted {self.max_tool_rounds} tool call rounds",
                empty_rounds=empty_rounds, truncated=truncated,
            )

        except Exception as e:
            e2e_ms = (time.monotonic() - start_time) * 1000
            return CaseResult(
                case_id=case_id, response=None, tool_calls_made=tool_calls_made,
                raw_content="", reasoning_content="", messages=messages,
                ttft_ms=round(first_token_ms, 1), e2e_ms=round(e2e_ms, 1),
                input_tokens=total_input_tokens, output_tokens=total_output_tokens,
                api_rounds=api_rounds,
                error=str(e),
                empty_rounds=empty_rounds, truncated=truncated,
            )

    async def _run_with_semaphore(self, client: httpx.AsyncClient, case: dict) -> CaseResult:
        async with self.semaphore:
            return await self._run_single_case(client, case)

    async def run(self, cases: list[dict]) -> list[CaseResult]:
        """Run all cases with controlled parallelism."""
        async with httpx.AsyncClient() as client:
            tasks = [self._run_with_semaphore(client, case) for case in cases]
            results = await asyncio.gather(*tasks)
        return list(results)


def main():
    parser = argparse.ArgumentParser(description="MetroLLM-Bench Runner")
    parser.add_argument("--cases", required=True, help="Path to cases JSON (e.g., cases/marta_cases.json)")
    parser.add_argument("--output", default=None, help="Output path (default: results/{model}_{timestamp}.json)")
    parser.add_argument("--llm-url", default="http://localhost:8080/v1", help="LLM API base URL")
    parser.add_argument("--llm-key", default="local", help="LLM API key")
    parser.add_argument("--llm-model", default="qwen3.5", help="Model name")
    parser.add_argument("--mock-url", default="http://localhost:8100", help="Mock server URL")
    parser.add_argument("--system", default="marta", help="Transit system name")
    parser.add_argument("--parallel", type=int, default=2, help="Parallel requests")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per response")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases (for testing)")
    parser.add_argument("--case-ids", default=None, help="Comma-separated case IDs to run (filters cases file)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 for reproducibility)")
    parser.add_argument("--max-tool-rounds", type=int, default=20, help="Max tool call rounds per case")
    parser.add_argument("--thinking", dest="thinking", action="store_true", default=True, help="Enable thinking mode (default)")
    parser.add_argument("--no-thinking", dest="thinking", action="store_false", help="Disable thinking mode")
    parser.add_argument("--extra-body-json", default=None, help="JSON string shallow-merged into each chat/completions request body")
    args = parser.parse_args()

    with open(args.cases) as f:
        cases = json.load(f)

    if args.case_ids:
        wanted = {cid.strip() for cid in args.case_ids.split(",") if cid.strip()}
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            print(f"Warning: case IDs not found: {sorted(missing)}")

    if args.limit:
        cases = cases[:args.limit]

    thinking_label = "thinking" if args.thinking else "non-thinking"
    print(f"Running {len(cases)} cases against {args.llm_model} ({thinking_label}) at {args.llm_url}")
    print(f"Mock server: {args.mock_url}, parallel: {args.parallel}")

    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None

    runner = BenchmarkRunner(
        llm_base_url=args.llm_url,
        llm_api_key=args.llm_key,
        llm_model=args.llm_model,
        mock_server_url=args.mock_url,
        system_name=args.system,
        parallel=args.parallel,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        temperature=args.temperature,
        max_tool_rounds=args.max_tool_rounds,
        extra_body=extra_body,
    )

    # Compute cases file checksum for reproducibility
    cases_checksum = hashlib.sha256(Path(args.cases).read_bytes()).hexdigest()[:12]

    # Git revision + dirty flag (best-effort)
    try:
        git_hash = subprocess.check_output(
            ["git", "describe", "--always", "--dirty"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = None

    started_at = datetime.now(timezone.utc).isoformat()
    results = asyncio.run(runner.run(cases))
    finished_at = datetime.now(timezone.utc).isoformat()

    # Build output
    output = {
        "metadata": {
            "harness_version": "0.4.0",
            "started_at": started_at,
            "finished_at": finished_at,
            "git_hash": git_hash,
            "llm_base_url": args.llm_url,
            "llm_model": args.llm_model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_tool_rounds": args.max_tool_rounds,
            "thinking": args.thinking,
            "parallel": args.parallel,
            "system": args.system,
            "cases_file": args.cases,
            "cases_checksum_sha256": cases_checksum,
        },
        "model": args.llm_model,
        "system": args.system,
        "thinking": args.thinking,
        "cases_total": len(cases),
        "cases_succeeded": sum(1 for r in results if r.error is None),
        "cases_failed": sum(1 for r in results if r.error is not None),
        "results": [asdict(r) for r in results],
    }

    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"{args.llm_model}_{ts}.json"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {output_path}")
    print(f"  Succeeded: {output['cases_succeeded']}/{output['cases_total']}")
    print(f"  Failed: {output['cases_failed']}/{output['cases_total']}")

    # Quick summary
    for r in results:
        status = "OK" if r.error is None else f"ERR: {r.error[:60]}"
        tools = len(r.tool_calls_made)
        print(f"  {r.case_id}: {status} ({tools} tool calls, {r.e2e_ms:.0f}ms)")


if __name__ == "__main__":
    main()
