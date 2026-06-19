"""Mock tool server for MetroLLM-Bench.

Exposes the transit tool endpoints that the benchmark runner forwards LLM
tool calls to:
  POST /route_planner
  POST /fare_calculator
  POST /station_info
  POST /line_info
  POST /disruption_feed
  POST /knowledge_base
  POST /submit_assistant_state

Run via:
  uvicorn harness.mock_server:app --port 8100
or via the project entry-point:
  mock-server --system marta --port 8100
"""

import argparse
import dataclasses
import datetime as dt
import hashlib
import html
import json
import os
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse

from harness.graph import MetroGraph
from harness.fares import FareCalculator

# ---------------------------------------------------------------------------
# LLM config (set by main(), used by /simulate)
# ---------------------------------------------------------------------------

_llm_base_url: str = "https://api.anthropic.com/v1"
_llm_api_key: str = ""
_llm_model: str = "claude-haiku-4-5-20251001"
_port: int = 8100

# ---------------------------------------------------------------------------
# Application state — populated at startup
# ---------------------------------------------------------------------------

app = FastAPI(title="MetroLLM-Bench Mock Tool Server")


# ---------------------------------------------------------------------------
# Optional hosted-simulator mode — opt-in via SIMULATOR_MODE=1, OFF by default.
# When on it adds per-user phrase auth, signed-cookie sessions, daily caps,
# JSONL usage logging, and an /admin aggregates page, for running a small
# invite-only hosted instance. It expects a `deploy/users.yaml` access list
# that is NOT part of this distribution (you supply your own); the
# `dashboard/simulator.html` UI it serves is included. If the access list is
# absent the mode disables itself gracefully. For the benchmark and local
# development, leave SIMULATOR_MODE unset, and the server is then an open,
# unauthenticated tool API (its only documented use).
# ---------------------------------------------------------------------------

_users_by_phrase: dict[str, str] = {}
_admin_phrase: str = ""
_simulator_mode: bool = False
_log_path: Optional[Path] = None
_daily_cap_user: int = 100
_daily_cap_global: int = 1000
_daily_counters: dict = {"date": "", "global": 0, "users": defaultdict(int)}
_counters_lock = threading.Lock()


def _utc_today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _check_and_bump_caps(user: str) -> Optional[str]:
    """None if within caps, else an error string. Resets at UTC midnight."""
    today = _utc_today()
    with _counters_lock:
        if _daily_counters["date"] != today:
            _daily_counters["date"] = today
            _daily_counters["global"] = 0
            _daily_counters["users"].clear()
        if _daily_counters["global"] >= _daily_cap_global:
            return f"Daily global cap ({_daily_cap_global}) reached. Try again tomorrow (UTC)."
        if _daily_counters["users"][user] >= _daily_cap_user:
            return f"Your daily cap ({_daily_cap_user} rollouts) is reached. Resets at UTC midnight."
        _daily_counters["global"] += 1
        _daily_counters["users"][user] += 1
    return None


def _log_usage(record: dict) -> None:
    """Append one JSONL record. Caller passes the fields; we add ts."""
    if not _log_path:
        return
    record["ts"] = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        with _log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as e:
        print(f"[simulator] log write failed: {e}", file=sys.stderr)


def _aggregate_usage() -> dict:
    """Read the JSONL log and produce summary stats for /admin."""
    today = _utc_today()
    out = {
        "today_count": 0,
        "today_users": defaultdict(int),
        "total_count": 0,
        "total_users": defaultdict(int),
        "total_in_tokens": 0,
        "total_out_tokens": 0,
        "last_seen": {},
        "estimated_cost_usd": 0.0,
    }
    if not _log_path or not _log_path.exists():
        return out
    with _log_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
            except Exception:
                continue
            user = r.get("user", "anon")
            ts = r.get("ts", "")
            out["total_count"] += 1
            out["total_users"][user] += 1
            out["total_in_tokens"] += r.get("input_tokens", 0) or 0
            out["total_out_tokens"] += r.get("output_tokens", 0) or 0
            out["last_seen"][user] = ts
            if ts.startswith(today):
                out["today_count"] += 1
                out["today_users"][user] += 1
    # Haiku pricing (May 2026): $1/M input, $5/M output.
    out["estimated_cost_usd"] = round(
        (out["total_in_tokens"] / 1_000_000) * 1.0
        + (out["total_out_tokens"] / 1_000_000) * 5.0,
        4,
    )
    return out


_LOGIN_CSS = (
    "body{font-family:'IBM Plex Sans',system-ui,sans-serif;max-width:480px;"
    "margin:80px auto;padding:0 20px;color:#1A2128;background:#F6F7F9}"
    "h1{margin:0 0 8px;font-family:'Tektur','IBM Plex Sans',sans-serif;"
    "font-weight:600;font-size:1.8rem}"
    "p{color:#3A4654;margin:0 0 24px}"
    ".err{color:#C21924;background:#FEE;padding:10px 12px;"
    "border:1px solid #FCC;margin-bottom:16px;font-size:0.9rem}"
    "input{display:block;width:100%;padding:10px 12px;font:inherit;"
    "border:1px solid #6A737D;border-radius:0;box-sizing:border-box;"
    "margin-bottom:12px;background:#FFF}"
    "button{padding:10px 18px;background:#1A2128;color:#fff;border:0;"
    "cursor:pointer;font:inherit;text-transform:uppercase;"
    "letter-spacing:1.2px;font-size:0.78rem}"
    "button:hover{background:#000}"
    ".hint{font-size:0.78rem;color:#6A737D;margin-top:24px;"
    "font-family:'IBM Plex Mono',ui-monospace,monospace}"
)


def _render_login(error: str = "") -> str:
    err = f'<div class="err">{html.escape(error)}</div>' if error else ""
    return (
        f"<!doctype html><html><head><title>MetroLLM Simulator · Sign in</title>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<style>{_LOGIN_CSS}</style></head><body>"
        f"<h1>MetroLLM Simulator</h1>"
        f"<p>Paste the access phrase you were sent.</p>"
        f"{err}"
        f"<form method='post' action='/login'>"
        f"<input type='password' name='phrase' autocomplete='off' "
        f"autofocus required placeholder='access phrase'>"
        f"<button type='submit'>Enter</button>"
        f"</form>"
        f"<div class='hint'>By design — limited invited access. "
        f"Usage is attributed to your name for cost tracking.</div>"
        f"</body></html>"
    )


def _render_admin_login(error: str = "") -> str:
    err = f'<div class="err">{html.escape(error)}</div>' if error else ""
    return (
        f"<!doctype html><html><head><title>Admin · MetroLLM</title>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<style>{_LOGIN_CSS}</style></head><body>"
        f"<h1>Admin</h1>"
        f"<p>Paste the admin phrase.</p>"
        f"{err}"
        f"<form method='post' action='/admin'>"
        f"<input type='password' name='phrase' autocomplete='off' "
        f"autofocus required>"
        f"<button type='submit'>Enter</button>"
        f"</form>"
        f"</body></html>"
    )


def _render_admin_aggregates() -> str:
    a = _aggregate_usage()
    rows = []
    sorted_users = sorted(a["total_users"].items(), key=lambda x: -x[1])
    for user, count in sorted_users:
        today = a["today_users"].get(user, 0)
        last = a["last_seen"].get(user, "—")
        rows.append(
            f"<tr><td>{html.escape(user)}</td>"
            f"<td>{today}</td><td>{count}</td>"
            f"<td>{html.escape(last[:19])}</td></tr>"
        )
    table_body = (
        "".join(rows)
        or '<tr><td colspan="4" style="color:#6A737D;text-align:center;'
           'padding:24px">No usage yet.</td></tr>'
    )
    return (
        f"<!doctype html><html><head><title>MetroLLM · Admin</title>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<style>"
        f"body{{font-family:'IBM Plex Sans',system-ui,sans-serif;"
        f"max-width:840px;margin:40px auto;padding:0 20px;color:#1A2128}}"
        f"h1{{margin:0 0 16px;font-family:'Tektur',sans-serif;font-weight:600}}"
        f"table{{width:100%;border-collapse:collapse;font-size:0.9rem}}"
        f"th,td{{padding:8px 10px;text-align:left;"
        f"border-bottom:1px solid #DEE2E6}}"
        f"th{{background:#F6F7F9;font-weight:500;"
        f"text-transform:uppercase;letter-spacing:1px;font-size:0.72rem}}"
        f".summary{{display:flex;gap:16px;margin-bottom:32px}}"
        f".summary div{{flex:1;padding:16px 18px;background:#F6F7F9;"
        f"border:1px solid #DEE2E6;font-family:'IBM Plex Mono',monospace;"
        f"font-size:0.78rem;color:#3A4654}}"
        f".summary .num{{font-size:1.6rem;color:#1A2128;display:block;"
        f"margin-bottom:4px;font-family:'IBM Plex Sans',sans-serif}}"
        f"</style></head><body>"
        f"<h1>MetroLLM Simulator · Admin</h1>"
        f"<div class='summary'>"
        f"<div><span class='num'>{a['today_count']}</span>"
        f"rollouts today (UTC)</div>"
        f"<div><span class='num'>{a['total_count']}</span>"
        f"rollouts all-time</div>"
        f"<div><span class='num'>${a['estimated_cost_usd']:.2f}</span>"
        f"est. spend (Haiku)</div>"
        f"</div>"
        f"<table><tr><th>User</th><th>Today</th><th>All-time</th>"
        f"<th>Last seen (UTC)</th></tr>{table_body}</table>"
        f"</body></html>"
    )


def _load_simulator_config() -> None:
    """Called at module load. Conditionally enables simulator mode + auth."""
    global _users_by_phrase, _admin_phrase, _simulator_mode, _log_path
    global _daily_cap_user, _daily_cap_global

    _simulator_mode = os.environ.get("SIMULATOR_MODE", "").lower() in (
        "1", "true", "yes", "on",
    )
    if not _simulator_mode:
        return

    users_path = (
        Path(__file__).resolve().parent.parent / "deploy" / "users.yaml"
    )
    if not users_path.exists():
        print(
            f"[simulator] users.yaml not found at {users_path}; "
            f"simulator mode disabled",
            file=sys.stderr,
        )
        _simulator_mode = False
        return

    cfg = yaml.safe_load(users_path.read_text()) or {}
    _admin_phrase = (cfg.get("admin_phrase") or "").strip()
    for u in cfg.get("users") or []:
        name = (u.get("name") or "").strip()
        phrase = (u.get("phrase") or "").strip()
        if name and phrase:
            _users_by_phrase[phrase] = name

    log_dir = Path(os.environ.get("LOG_DIR", "/var/log/metrollm"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        _log_path = log_dir / "usage.jsonl"
    except OSError as e:
        print(
            f"[simulator] could not create log dir {log_dir}: {e}; "
            f"logging disabled",
            file=sys.stderr,
        )
        _log_path = None

    _daily_cap_user = int(os.environ.get("DAILY_CAP_USER", "100"))
    _daily_cap_global = int(os.environ.get("DAILY_CAP_GLOBAL", "1000"))

    cookie_key = os.environ.get("COOKIE_SIGNING_KEY", "")
    if not cookie_key or len(cookie_key) < 32:
        print(
            "[simulator] COOKIE_SIGNING_KEY missing or < 32 chars — "
            "generating ephemeral key (sessions reset on every restart)",
            file=sys.stderr,
        )
        cookie_key = os.urandom(48).hex()

    # Routes (registered only in simulator mode so non-simulator deploys
    # keep their open API surface unchanged).

    @app.get("/")
    async def index_page() -> HTMLResponse:
        simulator_html = (
            Path(__file__).resolve().parent.parent
            / "dashboard"
            / "simulator.html"
        )
        if not simulator_html.exists():
            raise HTTPException(status_code=404, detail="simulator.html not found")
        return HTMLResponse(simulator_html.read_text())

    @app.get("/login")
    async def login_form() -> HTMLResponse:
        return HTMLResponse(_render_login())

    @app.post("/login")
    async def login_submit(request: Request):
        form = await request.form()
        phrase = (form.get("phrase") or "").strip()
        name = _users_by_phrase.get(phrase)
        if not name:
            return HTMLResponse(
                _render_login("Phrase not recognized."),
                status_code=401,
            )
        request.session["user"] = name
        request.session["login_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        return RedirectResponse(url="/", status_code=303)

    @app.post("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse(url="/login", status_code=303)

    @app.get("/me")
    async def me(request: Request) -> JSONResponse:
        return JSONResponse({"user": request.session.get("user")})

    @app.get("/admin")
    async def admin_form() -> HTMLResponse:
        return HTMLResponse(_render_admin_login())

    @app.post("/admin")
    async def admin_view(request: Request):
        form = await request.form()
        phrase = (form.get("phrase") or "").strip()
        if not _admin_phrase or phrase != _admin_phrase:
            return HTMLResponse(
                _render_admin_login("Phrase not recognized."),
                status_code=401,
            )
        return HTMLResponse(_render_admin_aggregates())

    # Tool endpoints called by the BenchmarkRunner via localhost loopback
    # during /simulate. Those calls don't carry the user's cookie, so they
    # have to be exempt from session auth. They're not security-sensitive
    # (pure data lookups against the loaded systems); the costly endpoint
    # is /simulate which stays gated.
    _PUBLIC_TOOL_PATHS = {
        "/route_planner", "/fare_calculator", "/station_info",
        "/line_info", "/set_disruptions", "/disruption_feed",
        "/knowledge_base", "/submit_assistant_state", "/verify/route",
    }

    # Auth middleware — registered BEFORE SessionMiddleware so the latter
    # wraps it (Starlette runs the most-recently-added middleware first).
    @app.middleware("http")
    async def _simulator_auth(request: Request, call_next):
        path = request.url.path
        if path in ("/login", "/logout", "/me", "/health"):
            return await call_next(request)
        if path.startswith("/admin"):
            # /admin has its own admin_phrase form; not session-gated
            return await call_next(request)
        if path in _PUBLIC_TOOL_PATHS:
            return await call_next(request)

        user = request.session.get("user")
        if not user:
            if path == "/" or path == "":
                return RedirectResponse(url="/login", status_code=303)
            return JSONResponse({"error": "auth_required"}, status_code=401)

        request.state.user = user
        return await call_next(request)

    # SessionMiddleware — added LAST so it's the OUTERMOST: runs first,
    # populates request.session, then auth middleware can read it.
    app.add_middleware(
        SessionMiddleware,
        secret_key=cookie_key,
        max_age=30 * 24 * 3600,
        same_site="lax",
        https_only=False,
    )

    print(
        f"[simulator] mode=on users={len(_users_by_phrase)} "
        f"caps=user/{_daily_cap_user} global/{_daily_cap_global} "
        f"log={_log_path or '(disabled)'}",
        flush=True,
    )


_load_simulator_config()

@dataclasses.dataclass
class _SystemData:
    """Per-system data loaded lazily and cached."""
    metro: MetroGraph
    fares: FareCalculator
    policies: list[dict]
    line_alias: dict[str, str]
    route_cache: dict[str, dict] = dataclasses.field(default_factory=dict)


_systems: dict[str, _SystemData] = {}           # system_name → data (lazy cache)
_case_system: dict[str, str] = {}               # case_id → system_name
_system_name: str = ""                           # default system (set at startup)
_disruptions_by_case: dict[str, list[dict]] = {}


def _build_line_alias(system_dir: Path) -> dict[str, str]:
    """Build alias→canonical_id map from lines.json.

    For a line with id="1", name="Line 1", generates:
      "1" → "1", "line 1" → "1"
    For id="red", name="Red Line":
      "red" → "red", "red line" → "red"
    """
    alias: dict[str, str] = {}
    lines_path = system_dir / "lines.json"
    if not lines_path.exists():
        return alias
    with open(lines_path) as f:
        lines = json.load(f)
    for line in lines:
        lid = line["id"]
        alias[lid.lower()] = lid
        if line.get("name"):
            alias[line["name"].lower()] = lid
    return alias


_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "systems"


def _load_system(name: str) -> _SystemData:
    """Load system data from disk, caching for subsequent calls."""
    if name in _systems:
        return _systems[name]
    sys_dir = _DATA_ROOT / name
    if not sys_dir.is_dir():
        raise ValueError(f"Unknown system: {name}")
    policies_path = sys_dir / "policies.json"
    if policies_path.exists():
        raw = json.loads(policies_path.read_text())
        policies: list[dict] = raw["policies"] if isinstance(raw, dict) and "policies" in raw else raw
    else:
        policies = []
    sd = _SystemData(
        metro=MetroGraph(sys_dir),
        fares=FareCalculator(sys_dir),
        policies=policies,
        line_alias=_build_line_alias(sys_dir),
    )
    _systems[name] = sd
    return sd


def _system_for_case(case_id: str | None) -> _SystemData:
    """Resolve system data for a case, falling back to startup default."""
    name = _case_system.get(case_id or "", _system_name)
    if not name:
        raise RuntimeError("No system configured")
    return _load_system(name)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

# --- /route_planner ----------------------------------------------------------

class StationRestriction(BaseModel):
    station: str
    restriction: str  # "closed", "skip", "no_transfer"


class LineClosure(BaseModel):
    line: str
    from_station: Optional[str] = None
    to_station: Optional[str] = None


class RoutePlannerRequest(BaseModel):
    origin: str
    destination: str
    departure_time: Optional[str] = None
    accessibility: Optional[list[str]] = None
    station_restrictions: Optional[list[StationRestriction]] = None
    segment_closures: Optional[list[list[str]]] = None
    line_closures: Optional[list[LineClosure]] = None
    case_id: Optional[str] = None


class StopInfo(BaseModel):
    station_id: str
    station_name: str
    line: Optional[str]
    is_transfer: bool
    transfer_to: Optional[str]


class RoutePlannerResponse(BaseModel):
    route_id: str
    stops: list[StopInfo]
    transfers: int
    estimated_minutes: float
    distance_miles: float
    line_sequence: list[str]


# --- /fare_calculator --------------------------------------------------------

class PassengerCounts(BaseModel):
    adults: int = Field(default=0, ge=0)
    children: int = Field(default=0, ge=0)
    seniors: int = Field(default=0, ge=0)
    disabled: int = Field(default=0, ge=0)


class FareCalculatorRequest(BaseModel):
    route_id: str
    passengers: PassengerCounts
    ticket_type: str = "single"
    payment_method: str = "breeze_card"
    case_id: Optional[str] = None


class LineItem(BaseModel):
    label: str
    amount: float
    currency: str


class Discount(BaseModel):
    label: str
    amount: float
    currency: str


class FareCalculatorResponse(BaseModel):
    fare_id: str
    line_items: list[LineItem]
    subtotal: float
    discounts: list[Discount]
    total: float
    currency: str


# --- /station_info -----------------------------------------------------------

VALID_QUERY_TYPES = frozenset(
    {"accessibility", "facilities", "exits", "connections", "real_time_status"}
)


class StationInfoRequest(BaseModel):
    station_id: Optional[str] = None
    station_ids: Optional[list[str]] = None
    query_type: str
    case_id: Optional[str] = None


class StationInfoResponse(BaseModel):
    station_id: str
    data: dict


class StationInfoBatchResponse(BaseModel):
    results: list[StationInfoResponse]


# --- /line_info --------------------------------------------------------------

class LineInfoRequest(BaseModel):
    line: Optional[str] = None
    lines: Optional[list[str]] = None
    case_id: Optional[str] = None


class LineStationEntry(BaseModel):
    station_id: str
    station_name: str
    position: int
    is_terminus: bool
    connections: list[str]  # other line ids at this station (empty if single-line)


class LineInfoResponse(BaseModel):
    line_id: str
    line_name: str
    color: str
    station_count: int
    is_loop: bool
    terminals: list[str]
    stations: list[LineStationEntry]


class LineInfoBatchResponse(BaseModel):
    results: list[LineInfoResponse]


# --- /disruption_feed -------------------------------------------------------

class DisruptionFeedRequest(BaseModel):
    case_id: Optional[str] = None  # internal: set by runner, not exposed to LLM
    current_time: Optional[str] = None  # ISO 8601 naive timestamp for temporal filtering
    line: Optional[str] = None
    station: Optional[str] = None
    severity_filter: str = "all"  # all, major, minor


class DisruptionEntry(BaseModel):
    id: str
    line: Optional[str] = None
    segment: Optional[list[str]] = None
    type: str
    severity: str
    message: str
    alternative: Optional[str] = None
    eta_resolution: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


class DisruptionFeedResponse(BaseModel):
    disruptions: list[DisruptionEntry]


# --- /knowledge_base --------------------------------------------------------

class KnowledgeBaseRequest(BaseModel):
    policy_id: str = ""
    query: str = ""
    category: str = "general"
    case_id: Optional[str] = None


class KnowledgeBaseResult(BaseModel):
    title: str
    content: str
    policy_id: str


class KnowledgeBaseResponse(BaseModel):
    results: list[KnowledgeBaseResult]
    found: bool


# --- /submit_assistant_state ------------------------------------------------

class RouteInfo(BaseModel):
    origin: str
    destination: str
    stops: list
    transfers: int
    estimated_minutes: float
    distance_miles: float
    line_sequence: list[str]

class AdvisoryBanner(BaseModel):
    severity: str
    title: str
    body: str

class FareQuoteInfo(BaseModel):
    passenger_summary: Optional[dict] = None
    line_items: list[dict] = Field(default_factory=list)
    discounts: list[dict] = Field(default_factory=list)
    total: float
    currency: str

class KioskAction(BaseModel):
    action: str
    reason_code: str

VALID_OUTCOMES = frozenset({
    "route_and_fare_ready", "advisory_only", "service_unavailable",
    "request_declined", "policy_answer_only",
})
VALID_ACTIONS = frozenset({
    "display_info", "prompt_purchase", "block_purchase", "refer_to_staff",
})
VALID_REASON_CODES = frozenset({
    "ok", "no_service", "invalid_request", "unsupported_request",
    "accessibility_issue", "policy_exception",
})

class SubmitAssistantStateRequest(BaseModel):
    outcome: str
    route: Optional[RouteInfo] = None
    fare_quote: Optional[FareQuoteInfo] = None
    kiosk_action: KioskAction
    advisory_banners: list[AdvisoryBanner] = Field(default_factory=list)
    assistant_message: str
    reasoning: str = ""
    case_id: Optional[str] = None


# --- /simulate ---------------------------------------------------------------

class SimulateRequest(BaseModel):
    system: str
    origin: str
    destination: str
    adults: int = 1
    children: int = 0
    seniors: int = 0
    disabled: int = 0
    current_time: str = ""    # ISO naive local, e.g. "2026-04-06T14:00:00"
    day_of_week: str = ""
    disruptions: list[dict] = Field(default_factory=list)
    freetext: str = ""
    accessibility_mode: bool = False
    # Cat F routing-impact policies (permanent operating patterns like
    # BART Yellow night shuttle, MARTA Green short-turn, CTA State/Lake
    # closure) are announced as prompt-level policy text, not disruptions.
    policy_change: Optional[dict] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _route_id(origin: str, destination: str) -> str:
    """Deterministic route_id derived from origin + destination."""
    raw = f"route:{origin.lower()}:{destination.lower()}"
    return "route_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _fare_id(route_id: str, passengers: PassengerCounts, ticket_type: str) -> str:
    """Deterministic fare_id derived from route + passengers + ticket type."""
    raw = (
        f"fare:{route_id}:{passengers.adults}:{passengers.children}:"
        f"{passengers.seniors}:{passengers.disabled}:{ticket_type}"
    )
    return "fare_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _station_subset(station_data: dict, query_type: str) -> dict:
    """Return the relevant subset of station data for the requested query type."""
    if query_type == "accessibility":
        return {
            "name": station_data.get("name"),
            "accessibility": station_data.get("accessibility", {}),
        }
    if query_type == "facilities":
        # Return all scalar/non-graph metadata; most systems store extra keys here
        return {
            k: v
            for k, v in station_data.items()
            if k not in {"connections"}
        }
    if query_type == "exits":
        # Exits may not be a dedicated field; surface what is available
        return {
            "name": station_data.get("name"),
            "type": station_data.get("type"),
            "zone": station_data.get("zone"),
        }
    if query_type == "connections":
        return {
            "name": station_data.get("name"),
            "lines": station_data.get("lines", []),
            "connections": station_data.get("connections", []),
        }
    if query_type == "real_time_status":
        # The mock server has no live data; return a static "operational" status
        return {
            "name": station_data.get("name"),
            "status": "operational",
            "alerts": [],
        }
    # Should not reach here after validation, but return everything as fallback
    return station_data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/route_planner", response_model=RoutePlannerResponse)
def route_planner(req: RoutePlannerRequest) -> RoutePlannerResponse:
    sd = _system_for_case(req.case_id)
    metro = sd.metro

    try:
        if req.station_restrictions or req.segment_closures or req.line_closures:
            restrictions = [
                {"station": r.station, "restriction": r.restriction}
                for r in (req.station_restrictions or [])
            ]
            segments = [tuple(s) for s in (req.segment_closures or [])]
            if req.line_closures:
                closures_as_dicts = []
                for lc in req.line_closures:
                    cd = {"line": sd.line_alias.get(lc.line.lower(), lc.line)}
                    if lc.from_station is not None:
                        cd["from_station"] = lc.from_station
                    if lc.to_station is not None:
                        cd["to_station"] = lc.to_station
                    closures_as_dicts.append(cd)
                segments.extend(metro.expand_line_closures(closures_as_dicts))
            result = metro.shortest_path_with_restrictions(
                req.origin, req.destination,
                station_restrictions=restrictions,
                segment_closures=segments,
            )
        else:
            result = metro.shortest_path(req.origin, req.destination)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except nx.NetworkXNoPath as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except nx.NodeNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    stops = [
        StopInfo(
            station_id=s["station_id"],
            station_name=s["station_name"],
            line=s.get("line"),
            is_transfer=s.get("is_transfer", False),
            transfer_to=s.get("transfer_to"),
        )
        for s in result.stations
    ]

    rid = _route_id(req.origin, req.destination)

    # Cache route details for fare_calculator surcharge lookups
    origin_id = result.stations[0]["station_id"] if result.stations else req.origin
    dest_id = result.stations[-1]["station_id"] if result.stations else req.destination
    sd.route_cache[rid] = {
        "origin": origin_id,
        "destination": dest_id,
        "distance_miles": result.distance_miles,
    }

    return RoutePlannerResponse(
        route_id=rid,
        stops=stops,
        transfers=result.transfers,
        estimated_minutes=result.estimated_minutes,
        distance_miles=result.distance_miles,
        line_sequence=result.line_sequence,
    )


@app.post("/fare_calculator", response_model=FareCalculatorResponse)
def fare_calculator(req: FareCalculatorRequest) -> FareCalculatorResponse:
    sd = _system_for_case(req.case_id)

    passengers_dict = {
        "adults": req.passengers.adults,
        "children": req.passengers.children,
        "seniors": req.passengers.seniors,
        "disabled": req.passengers.disabled,
    }

    # Look up cached route details for distance-based fare models
    cached = sd.route_cache.get(req.route_id, {})

    try:
        result = sd.fares.calculate(
            passengers=passengers_dict,
            ticket_type=req.ticket_type,
            payment_method=req.payment_method,
            route_distance_miles=cached.get("distance_miles"),
            origin_id=cached.get("origin"),
            destination_id=cached.get("destination"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return FareCalculatorResponse(
        fare_id=_fare_id(req.route_id, req.passengers, req.ticket_type),
        line_items=[LineItem(**item) for item in result.items],
        subtotal=result.subtotal,
        discounts=[Discount(**d) for d in result.discounts],
        total=result.total,
        currency=result.currency,
    )


@app.post("/station_info")
def station_info(req: StationInfoRequest) -> StationInfoResponse | StationInfoBatchResponse:
    if req.query_type not in VALID_QUERY_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid query_type '{req.query_type}'. "
                f"Must be one of: {sorted(VALID_QUERY_TYPES)}"
            ),
        )

    # Batch mode: multiple stations in one call
    ids = req.station_ids or ([req.station_id] if req.station_id else [])
    if not ids:
        raise HTTPException(status_code=422, detail="Provide station_id or station_ids")

    sd = _system_for_case(req.case_id)
    results = []
    for sid in ids:
        data = sd.metro.station_info(sid)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Station '{sid}' not found",
            )
        results.append(StationInfoResponse(
            station_id=sid,
            data=_station_subset(data, req.query_type),
        ))

    # Single station: return flat response (backwards compatible)
    if len(results) == 1 and not req.station_ids:
        return results[0]

    return StationInfoBatchResponse(results=results)


def _build_line_info(sd, requested: str) -> LineInfoResponse:
    metro = sd.metro
    line_id = sd.line_alias.get(requested.lower(), requested)
    if line_id not in metro.lines:
        raise HTTPException(status_code=404, detail=f"Unknown line: {requested}")
    line = metro.lines[line_id]
    ordered: list[str] = list(line.get("stations", []))
    terminals = metro.line_terminals(line_id)
    is_loop = metro.is_loop_line(line_id)
    entries: list[LineStationEntry] = []
    for pos, sid in enumerate(ordered):
        station = metro.stations.get(sid, {})
        connections = sorted(metro.station_lines.get(sid, set()) - {line_id})
        entries.append(LineStationEntry(
            station_id=sid,
            station_name=station.get("name", sid),
            position=pos,
            is_terminus=sid in terminals,
            connections=connections,
        ))
    return LineInfoResponse(
        line_id=line_id,
        line_name=line.get("name", line_id),
        color=line.get("color", ""),
        station_count=len(ordered),
        is_loop=is_loop,
        terminals=terminals,
        stations=entries,
    )


@app.post("/line_info")
def line_info(req: LineInfoRequest) -> LineInfoResponse | LineInfoBatchResponse:
    requested = req.lines or ([req.line] if req.line else [])
    if not requested:
        raise HTTPException(status_code=422, detail="Provide line or lines")
    sd = _system_for_case(req.case_id)
    results = [_build_line_info(sd, r) for r in requested]
    if len(results) == 1 and not req.lines:
        return results[0]
    return LineInfoBatchResponse(results=results)


@app.post("/set_disruptions")
def set_disruptions(payload: dict) -> dict:
    case_id = payload.get("case_id", "_default")
    disruptions = payload.get("disruptions", [])
    # Normalize segment to a flat list of station IDs. events.yaml stores
    # segment as {from_station, to_station, stations: [...]}; cases/generator.py
    # flattens it when materializing case JSONs (see :2387, :4500). Mirror that
    # here so the demo can forward events.yaml shapes directly without each
    # caller having to know the contract — DisruptionEntry.segment is list[str]
    # and the station-filter check in disruption_feed uses `in` on the value.
    for d in disruptions:
        seg = d.get("segment")
        if isinstance(seg, dict):
            d["segment"] = list(seg.get("stations") or [])
    _disruptions_by_case[case_id] = disruptions
    if payload.get("system"):
        _case_system[case_id] = payload["system"]
    return {"ok": True}


@app.post("/disruption_feed", response_model=DisruptionFeedResponse)
def disruption_feed(req: DisruptionFeedRequest) -> DisruptionFeedResponse:
    filtered = _disruptions_by_case.get(req.case_id or "_default", [])

    if req.line:
        # Normalize requested line to canonical ID via alias map
        sd = _system_for_case(req.case_id)
        req_canonical = sd.line_alias.get(req.line.lower())
        # Keep disruptions that match the canonical line OR have no line (station closures affect all lines)
        filtered = [
            d for d in filtered
            if not d.get("line") or d["line"].lower() == (req_canonical or req.line).lower()
        ]

    if req.station:
        filtered = [d for d in filtered
                    if (d.get("segment") and req.station in d["segment"])
                    or req.station.lower() in d.get("message", "").lower()]

    if req.severity_filter == "major":
        filtered = [d for d in filtered if d.get("severity") in ("critical", "warning")]
    elif req.severity_filter == "minor":
        filtered = [d for d in filtered if d.get("severity") == "info"]

    # Temporal filtering: remove expired disruptions when current_time is provided.
    # - Expired: valid_until is set and valid_until < current_time  → filter out
    # - Future:  valid_from is set and valid_from > current_time    → keep (announced, not yet active)
    # - No temporal bounds: always active (backwards compatible)
    # Uses lexicographic ISO 8601 string comparison (works for naive timestamps).
    if req.current_time:
        now = req.current_time
        filtered = [
            d for d in filtered
            if not (d.get("valid_until") and d["valid_until"] < now)
        ]

    entries = [DisruptionEntry(**d) for d in filtered]
    return DisruptionFeedResponse(disruptions=entries)


@app.post("/knowledge_base", response_model=KnowledgeBaseResponse)
def knowledge_base(req: KnowledgeBaseRequest) -> KnowledgeBaseResponse:
    sd = _system_for_case(req.case_id)
    policies = sd.policies

    # Exact lookup by policy_id (preferred path)
    if req.policy_id:
        for p in policies:
            if p.get("policy_id") == req.policy_id:
                return KnowledgeBaseResponse(
                    results=[KnowledgeBaseResult(
                        title=p.get("title", ""),
                        content=p.get("content", ""),
                        policy_id=req.policy_id,
                    )],
                    found=True,
                )
        return KnowledgeBaseResponse(results=[], found=False)

    # Fallback: keyword search across all policies (no category gate)
    if not req.query:
        return KnowledgeBaseResponse(results=[], found=False)

    query_words = [w.lower() for w in req.query.split() if len(w) > 2]
    scored: list[tuple[int, dict]] = []
    for policy in policies:
        text = (policy.get("title", "") + " " + policy.get("content", "")).lower()
        syns = " ".join(policy.get("synonyms", []))
        text += " " + syns.lower()
        hits = sum(1 for w in query_words if w in text)
        if hits > 0:
            scored.append((hits, policy))

    # Sort by hit count descending, take top 3
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in scored[:3]]

    results = [
        KnowledgeBaseResult(
            title=p.get("title", ""),
            content=p.get("content", ""),
            policy_id=p.get("policy_id", p.get("id", "")),
        )
        for p in top
    ]

    return KnowledgeBaseResponse(results=results, found=len(results) > 0)


@app.post("/submit_assistant_state")
def submit_assistant_state(req: SubmitAssistantStateRequest) -> dict:
    """Validate and accept the LLM's final assistant kiosk state.

    Returns {"accepted": True} on success. On validation failure, Pydantic
    raises a 422 with field-level error details before this handler runs.
    Additional structural checks return 422 for conditional field violations.
    """
    # Validate enum values
    if req.outcome not in VALID_OUTCOMES:
        raise HTTPException(status_code=422, detail=f"Invalid outcome: {req.outcome}")
    if req.kiosk_action.action not in VALID_ACTIONS:
        raise HTTPException(status_code=422, detail=f"Invalid action: {req.kiosk_action.action}")
    if req.kiosk_action.reason_code not in VALID_REASON_CODES:
        raise HTTPException(status_code=422, detail=f"Invalid reason_code: {req.kiosk_action.reason_code}")

    # Conditional field validation
    if req.outcome in ("route_and_fare_ready", "advisory_only") and req.route is None:
        raise HTTPException(status_code=422, detail=f"route required when outcome={req.outcome}")
    if req.outcome == "route_and_fare_ready" and req.fare_quote is None:
        raise HTTPException(status_code=422, detail="fare_quote required when outcome=route_and_fare_ready")

    # Validate route.stops are known station IDs
    if req.route and req.route.stops:
        sd = _system_for_case(req.case_id)
        stop_ids = [s.get("station_id", s) if isinstance(s, dict) else s for s in req.route.stops]
        invalid = [s for s in stop_ids if s not in sd.metro.stations]
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown station IDs in route.stops: {invalid[:5]}. Use station_id values from route_planner (e.g. MARTA-AP).",
            )

    return {"accepted": True, "response_id": hashlib.sha256(
        req.model_dump_json().encode()
    ).hexdigest()[:12]}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Verify / interactive map endpoints
# ---------------------------------------------------------------------------

_verify_graphs: dict[str, MetroGraph] = {}  # system_name → MetroGraph for all systems

def _get_verify_graph(system: str) -> MetroGraph:
    if system not in _verify_graphs:
        system_dir = Path(__file__).resolve().parent.parent / "data" / "systems" / system
        if not system_dir.is_dir():
            raise ValueError(f"Unknown system: {system}")
        _verify_graphs[system] = MetroGraph(system_dir)
    return _verify_graphs[system]

@app.get("/verify")
def verify_page() -> HTMLResponse:
    verify_html = Path(__file__).resolve().parent.parent / "dashboard" / "verify.html"
    if not verify_html.exists():
        raise HTTPException(status_code=404, detail="verify.html not found")
    return HTMLResponse(verify_html.read_text())


@app.get("/verify_data.json")
def verify_data() -> JSONResponse:
    verify_json = Path(__file__).resolve().parent.parent / "dashboard" / "verify_data.json"
    if not verify_json.exists():
        raise HTTPException(status_code=404, detail="verify_data.json not found — run: uv run python data/verify.py --export-map")
    return JSONResponse(json.loads(verify_json.read_text()))


@app.get("/annotate")
def annotate_page() -> HTMLResponse:
    annotate_html = Path(__file__).resolve().parent.parent / "dashboard" / "annotate.html"
    if not annotate_html.exists():
        raise HTTPException(status_code=404, detail="annotate.html not found")
    return HTMLResponse(annotate_html.read_text())


@app.get("/simulator")
def simulator_page() -> HTMLResponse:
    simulator_html = Path(__file__).resolve().parent.parent / "dashboard" / "simulator.html"
    if not simulator_html.exists():
        raise HTTPException(status_code=404, detail="simulator.html not found")
    return HTMLResponse(simulator_html.read_text())


@app.get("/systems")
def list_systems() -> JSONResponse:
    systems_dir = Path(__file__).resolve().parent.parent / "data" / "systems"
    systems = sorted(
        d.name for d in systems_dir.iterdir()
        if d.is_dir() and (d / "framebook.yaml").exists()
    )
    return JSONResponse(systems)


@app.get("/stations/{system}")
def stations_for_system(system: str) -> JSONResponse:
    system_dir = Path(__file__).resolve().parent.parent / "data" / "systems" / system
    if not system_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown system: {system}")
    stations_path = system_dir / "stations.json"
    lines_path = system_dir / "lines.json"
    stations = json.loads(stations_path.read_text()) if stations_path.exists() else []
    lines = json.loads(lines_path.read_text()) if lines_path.exists() else []
    return JSONResponse({"stations": stations, "lines": lines})


@app.post("/simulate")
async def simulate(req: SimulateRequest, request: Request) -> JSONResponse:
    """Run a single interactive kiosk case through the LLM and return the result."""
    import httpx as _httpx
    from harness.runner import BenchmarkRunner

    # Simulator mode: enforce per-user + global daily caps before doing any
    # work; attribute the rollout to the logged-in user for usage logging.
    user = getattr(request.state, "user", None) if _simulator_mode else None
    if _simulator_mode and user:
        cap_err = _check_and_bump_caps(user)
        if cap_err:
            return JSONResponse({"error": cap_err}, status_code=429)

    # Build disruption objects for each user-injected disruption
    disruptions = []
    for i, d in enumerate(req.disruptions):
        entry = {
            "id": f"sim-disruption-{i}",
            "type": d.get("type", "delay"),
            "severity": d.get("severity", "warning"),
            "message": d.get("message", ""),
            "line": d.get("line") or None,
            "segment": d.get("segment") or None,
            "alternative": d.get("alternative") or None,
            "valid_from": d.get("valid_from") or None,
            "valid_until": d.get("valid_until") or None,
        }
        disruptions.append(entry)

    # Build case dict matching the structure used by _run_single_case
    case_id = f"sim-{req.system}-{req.origin[:8]}-{req.destination[:8]}".replace(" ", "_").lower()

    events: list[dict] = [
        {"type": "station_selected", "field": "origin", "value": req.origin},
        {"type": "station_selected", "field": "destination", "value": req.destination},
        {"type": "passenger_count_changed",
         "adults": req.adults, "children": req.children,
         "seniors": req.seniors, "disabled": req.disabled},
    ]
    # Emit one disruption_update event per disruption. The runner's prompt
    # builder appends each as a separate "⚠ DISRUPTION ALERT" line so the
    # model sees every disruption in the user query, not just the first.
    for d in disruptions:
        events.append({
            "type": "disruption_update",
            "disruption": d,
        })
    if req.freetext:
        events.append({"type": "freetext_input", "text": req.freetext})

    system_context: dict = {}
    if req.accessibility_mode:
        system_context["accessibility_mode"] = True
    if disruptions:
        system_context["active_disruptions"] = disruptions
    if req.current_time or req.day_of_week:
        system_context["temporal_context"] = {
            "current_time": req.current_time or "",
            "day_of_week": req.day_of_week or "",
        }
    if req.policy_change:
        system_context["policy_change"] = req.policy_change

    case = {
        "id": case_id,
        "system": req.system,
        "category": "simulator",
        "events": events,
        "system_context": system_context,
    }

    # Register case system + disruptions for tool endpoint routing
    _case_system[case_id] = req.system
    _disruptions_by_case[case_id] = disruptions
    try:
        _load_system(req.system)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown system: {req.system}")

    # GPT-5 family (including Azure deployments) only accepts temperature=1
    is_gpt5_family = (
        "azure.com" in _llm_base_url
        or (_llm_model or "").startswith("gpt-5")
    )
    simulator_temperature = 1.0 if is_gpt5_family else 0.0

    runner = BenchmarkRunner(
        llm_base_url=_llm_base_url,
        llm_api_key=_llm_api_key,
        llm_model=_llm_model,
        mock_server_url=f"http://localhost:{_port}",
        system_name=req.system,
        parallel=1,
        max_tokens=4096,
        thinking=False,   # disable thinking for Haiku / API models
        temperature=simulator_temperature,
        max_tool_rounds=20,
    )

    try:
        async with _httpx.AsyncClient() as client:
            result = await runner._run_single_case(client, case)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}") from e
    finally:
        _case_system.pop(case_id, None)
        _disruptions_by_case.pop(case_id, None)

    if _simulator_mode and user:
        _log_usage({
            "user": user,
            "system": req.system,
            "origin": req.origin,
            "destination": req.destination,
            "model": _llm_model,
            "input_tokens": getattr(result, "input_tokens", 0) or 0,
            "output_tokens": getattr(result, "output_tokens", 0) or 0,
            "rounds": getattr(result, "api_rounds", 0) or 0,
            "ms": getattr(result, "e2e_ms", 0) or 0,
            "ok": getattr(result, "error", None) is None,
        })

    return JSONResponse({"case": case, **dataclasses.asdict(result)})


@app.get("/calibration_cases_blind.json")
def calibration_cases_blind() -> JSONResponse:
    cal_json = Path(__file__).resolve().parent.parent / "dashboard" / "calibration_cases_blind.json"
    if not cal_json.exists():
        raise HTTPException(status_code=404, detail="calibration_cases_blind.json not found")
    return JSONResponse(json.loads(cal_json.read_text()))


@app.get("/calibration_cases.json")
def calibration_cases_full() -> JSONResponse:
    """Full calibration file with judge scores + reasoning. UI enforces blindness."""
    cal_json = Path(__file__).resolve().parent.parent / "results" / "calibration_cases.json"
    if not cal_json.exists():
        raise HTTPException(status_code=404, detail="calibration_cases.json not found")
    return JSONResponse(json.loads(cal_json.read_text()))


class VerifyRouteRequest(BaseModel):
    origin: str
    destination: str
    system: str = ""


@app.post("/verify/route")
def verify_route(req: VerifyRouteRequest):
    sys_name = req.system or _system_name
    try:
        metro = _get_verify_graph(sys_name) if sys_name else _load_system(_system_name).metro
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    try:
        result = metro.shortest_path(req.origin, req.destination)
    except (ValueError, nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    return {
        "origin": req.origin,
        "destination": req.destination,
        "path": result.path,
        "stops": [
            {
                "station_id": s["station_id"],
                "station_name": s["station_name"],
                "line": s.get("line"),
                "is_transfer": s.get("is_transfer", False),
            }
            for s in result.stations
        ],
        "transfers": result.transfers,
        "distance_miles": result.distance_miles,
        "estimated_minutes": result.estimated_minutes,
        "line_sequence": result.line_sequence,
        "system": _system_name,
    }


def main() -> None:
    # Resolve default LLM config from .env before parsing args.
    # Prefer Azure gpt-5.4-mini when AZURE_* vars are present; fall back to Anthropic Haiku.
    default_llm_url = "https://api.anthropic.com/v1"
    default_llm_key = ""
    default_llm_model = "claude-haiku-4-5-20251001"
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        azure_endpoint = os.environ.get("AZURE_ENDPOINT", "").rstrip("/")
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        azure_mini = os.environ.get("AZURE_MINI_LLM_DEPLOYMENT", "")
        if azure_endpoint and azure_key and azure_mini:
            default_llm_url = f"{azure_endpoint}/openai/deployments/{azure_mini}?api-version=2024-10-21"
            default_llm_key = azure_key
            default_llm_model = azure_mini
        else:
            default_llm_key = os.environ.get("ANTHROPIC_API_KEY", "")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="MetroLLM-Bench mock tool server")
    parser.add_argument(
        "--system",
        default="marta",
        help="Transit system name (must exist under data/systems/). Default: marta",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8100,
        help="Port to listen on. Default: 8100",
    )
    parser.add_argument(
        "--llm-url",
        default=default_llm_url,
        help="LLM API base URL for /simulate endpoint. Default: Azure gpt-5.4-mini if AZURE_* env vars set, else https://api.anthropic.com/v1",
    )
    parser.add_argument(
        "--llm-key",
        default=default_llm_key,
        help="LLM API key. Default: AZURE_OPENAI_API_KEY or ANTHROPIC_API_KEY from .env",
    )
    parser.add_argument(
        "--llm-model",
        default=default_llm_model,
        help="LLM model name (Azure deployment name for Azure). Default: AZURE_MINI_LLM_DEPLOYMENT or claude-haiku-4-5-20251001",
    )
    args = parser.parse_args()

    system_dir = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "systems"
        / args.system
    )

    if not system_dir.is_dir():
        print(
            f"Error: system directory not found: {system_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    global _system_name
    global _llm_base_url, _llm_api_key, _llm_model, _port
    _system_name = args.system
    _llm_base_url = args.llm_url
    _llm_api_key = args.llm_key
    _llm_model = args.llm_model
    _port = args.port

    sd = _load_system(args.system)
    print(f"Loaded system '{args.system}' ({len(sd.policies)} policies) from {system_dir}")
    print(f"Simulator LLM: {args.llm_model} @ {args.llm_url}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
