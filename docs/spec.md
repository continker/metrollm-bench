# MetroLLM-Bench v3 — Build Specification

## Project Structure

```
metrollm-bench/
├── data/
│   ├── systems/
│   │   ├── marta/            # Atlanta MARTA (38 stations, 4 lines, flat fare)
│   │   ├── doha/             # Doha Metro (37 stations, 3 lines, flat fare)
│   │   ├── bart/             # SF BART (50 stations, 6 lines, distance fare)
│   │   ├── taipei/           # Taipei MRT (107 stations, 5 lines, distance fare)
│   │   ├── cta/              # Chicago CTA L (142 stations, 8 lines, flat-with-exceptions)
│   │   └── beijing/          # Beijing Subway (414 stations, 25 lines, distance + airport flat)
│   ├── splits/               # Pre-registered 75/25 held-out partition spec (seed=42)
│   └── gtfs/                 # Wikipedia/GTFS-derived station coordinates
├── harness/
│   ├── mock_server.py        # FastAPI tool server (transit tools + submit_assistant_state)
│   ├── runner.py             # Async runner against an OpenAI-compatible endpoint
│   ├── scorer.py             # 22-component scorer (Tier 1 deterministic + Tier 2 judge)
│   ├── judge.py              # LLM judge (Claude Haiku) for Tier 2
│   ├── graph.py              # Station graph (pathfinding, restrictions)
│   ├── fares.py              # Per-system fare calculators
│   └── rule_agent.py         # Deterministic rule-based baseline
├── cases/
│   ├── generator.py          # Generates Cat A-K cases with ground truth
│   └── {system}_cases.json   # Per-system case files (955 cases) + 75/25 split slices
├── scripts/                  # PEFT training/export, Mac-bench, paper analysis
├── dashboard/                # Results dashboard, calibration annotator, simulator
├── tests/                    # Deterministic test suite (no network)
└── results/                  # Output JSON per model run (gitignored)
```

---

## 1. Metro System Definitions

Each system directory contains: `stations.json`, `lines.json`, `graph.json`, `fares.json`, `accessibility.json`, `framebook.yaml`.

### 1.1 MARTA — Atlanta (Small)

- stations: 38
- lines: 4 (Red, Gold, Blue, Green)
- fare_model: flat
- base_fare_usd: 2.50
- payment: Breeze Card, contactless
- children_free_under: 5 (limit 2 per paying adult)
- reduced_fare: seniors 65+, disabled — half fare
- hours: 05:00–01:00 (varies by line/day)
- 24h_service: false
- languages: [en]
- signature_events:
  - super_bowl (station closures, extended hours, crowd control)
  - hurricane_warning (phased suspension protocol)
  - peachtree_road_race (July 4, station crowding, extended AM service)
- key_challenge: simplest system, flat fare, baseline for minimum viable LLM
- accessibility_notes: all stations ADA compliant, elevators at every station

### 1.2 BART — San Francisco (Medium)

- stations: 50
- lines: 6 (Antioch, Richmond, Warm Springs, Dublin/Pleasanton, Berryessa, SFO/Millbrae)
- fare_model: distance_plus_surcharges
- fare_rules:
  - base: $2.15 for ≤6mi
  - distance_increment: varies by mileage bracket
  - surcharges:
    - transbay_tube: $1.40
    - sfo_airport: $4.95
    - oakland_airport: $6.70
    - san_mateo_county: $1.45 (except $1.25 for Daly City)
  - max_fare: $17.60 (with all surcharges)
  - max_fare_no_surcharge: $10.30
- payment: Clipper card, Tap and Ride (contactless bank card, Apple/Google Pay)
- discounts:
  - high_value: 6.25% on autoload $48/$64
  - youth_5_18: 50%
  - senior_disabled: 62.5%
  - clipper_start_low_income: 50%
  - children_under_5: free
- hours: ~05:00–00:00 (varies)
- 24h_service: false
- languages: [en, es, zh]
- signature_events:
  - earthquake (seismic early warning → service pause → inspection → phased resumption)
  - fleet_week (crowd surges at Embarcadero)
  - bay_bridge_closure (increased ridership, capacity strain)
  - giants_game / warriors_game (station-specific crowding)
- key_challenge: multi-component fare arithmetic (distance + multiple surcharges stacking), Transbay tube as binary cost component, airport connections
- accessibility_notes: all stations accessible, some older stations have long ramp paths

### 1.3 Taipei MRT (Medium-International)

- stations: ~131
- lines: 6 main + Danhai LRT + Maokong Gondola
  - Red (Tamsui-Xinyi), Blue (Bannan), Green (Songshan-Xindian), Orange (Zhonghe-Xinlu), Brown (Wenhu), Yellow (Circular)
- fare_model: distance_matrix
- fare_rules:
  - range: NT$20–65
  - maokong_gondola: NT$180 flat
  - taoyuan_airport_mrt: separate system, NT$30–160
- payment: EasyCard, iPASS, icash, contactless bank card
- cashback_program:
  - tiers: [{trips: "1-10", rate: 0}, {trips: "11-20", rate: 0.10}, {trips: "21-30", rate: 0.15}, {trips: "31-40", rate: 0.20}, {trips: "41-50", rate: 0.25}, {trips: "51+", rate: 0.30}]
  - application: retroactive_to_all_trips_in_month
  - settlement: monthly
- children_free_under: 6 (with adult)
- strict_rules: [no_eating, no_drinking, no_gum, no_smoking — fines enforced past yellow line]
- hours: ~06:00–00:00
- 24h_service: true_on_new_years_eve_only
- languages: [zh-TW, en, ja, ko; some stations add th, vi]
- signature_events:
  - typhoon:
    - wenhu_line (elevated): suspend at gusts ≥ Beaufort 10 or avg wind ≥ Beaufort 7
    - other_elevated_sections: same thresholds, underground continues with adjusted headway (12-15 min)
    - maokong_gondola: suspend first (lowest threshold)
    - youbike: suspend entirely
    - school_work_closure: all lines 15-30 min intervals
  - lunar_new_year (extended hours, special fares, crowd advisories)
  - new_years_eve (24h service)
- key_challenge: cashback tier arithmetic, typhoon tiered suspension protocol (elevated vs underground), multilingual, culturally distinct rules (no eating/drinking), Maokong Gondola as boundary system
- accessibility_notes: full wheelchair access, elevator status in app, priority seats strictly observed

### 1.4 CTA L — Chicago (Large)

- stations: 146
- lines: 8 (Red, Blue, Brown, Green, Orange, Pink, Purple, Yellow)
- fare_model: flat_with_exceptions
- fare_rules:
  - rail: $2.50 (Ventra), $3.50 (disposable ticket), $3.00 (contactless bank card)
  - bus: $2.25 (Ventra), $2.50 (cash, exact change, no transfers)
  - ohare_blue_line: $5.00
  - transfer: free within 2 hours (Ventra only), up to 2 transfers
- payment: Ventra card, contactless, Apple/Google Pay
- passes:
  - 1_day: available
  - 3_day: available
  - 7_day: available
  - 30_day: available
- children_free_under: 7 (limit 2 per paying adult)
- reduced_fare: seniors 65+, disabled, students — $1.10 bus, $1.25 rail
- hours: varies by line
- 24h_service: [Red, Blue] — only two lines, makes Chicago one of three cities worldwide with 24h rail
- languages: [en, es, zh, pl] (signage); announcements [en, es]
- signature_events:
  - polar_vortex:
    - extreme cold (-30°F windchill) → rail switch heater failures
    - elevated outdoor sections: reduced service or suspension
    - subway sections: continue normally
    - protocol mirrors Taipei typhoon (elevated vs underground split)
  - blizzard (service suspensions on outdoor sections, bus chains)
  - cubs_game / sox_game / bears_game (station-specific crowding at Addison, Sox-35th, etc.)
  - lollapalooza / marathon (route diversions, extended service)
  - state_lake_station_closure (closed Jan 2026 for rebuild through 2029)
- key_challenge: Loop topology (multiple lines sharing elevated Loop track), 24h service on Red/Blue creates last-train asymmetry, O'Hare surcharge exception to flat fare, 146 stations is the largest graph, State/Lake closure is a live long-duration disruption
- accessibility_notes: majority of stations have elevators/ramps but NOT all — this is a real gap the LLM must know about

### 1.5 Beijing Metro (Extra-Large / Scale Test)

- stations: 414
- lines: 27 (Lines 1–19, Yizhuang, Fangshan, Yanfang, S1, Changping, Capital Airport Express, Daxing Airport Express, Xijiao)
- fare_model: distance_based
- fare_rules:
  - brackets: ¥3 (≤6km) to ¥10 (>50km), 8 tiers
  - capital_airport_express: ¥25 flat (replaces distance fare)
  - daxing_airport_express: ¥35 flat (replaces distance fare)
- payment: Yikatong Card, Mobile Pay (Alipay/WeChat Pay), single-journey ticket (requires ID scan)
- discounts:
  - children_under_120cm: free
  - senior_65_plus: free (Beijing Senior Card holders only, not visitors)
  - disabled: free (disability card)
  - yikatong_monthly: 50% off after ¥100 cumulative spend (not modeled — see D8)
- hours: ~05:00–23:00 (varies by line, no 24h service)
- languages: [zh, en]
- signature_events:
  - smog_warning: 3-phase (advisory at AQI>200, outdoor lines suspended at AQI>300, full shutdown at AQI>500)
  - national_day_crowd: Tiananmen flow control, extended security screening
  - olympic_park_events: crowd advisory at Line 8/15 stations
- key_challenge: scale stress test — 414 stations is 3× the next largest system (CTA 142). Tests whether tool-mediated architecture degrades with network size. Real-name ticketing and mandatory security screening add Beijing-specific policy complexity. Airport express lines use flat fares that replace (not supplement) distance fares.
- accessibility_notes: most stations have elevators, 8 older stations (Lines 1, 2, 6) lack step-free access
- cultural_notes: mandatory X-ray security screening at all stations, no eating/drinking on trains, real-name ID scan for vending machine purchases, priority seating culturally enforced

---

## 2. Tool API

All models receive identical tool definitions. Tools return scripted deterministic responses per test case.

```yaml
tools:
  - name: route_planner
    description: "Find optimal route between two stations"
    params:
      origin: string (station_id or name)
      destination: string (station_id or name)
      departure_time: string (ISO 8601, optional)
      accessibility: array of string (optional) ["step_free", "elevator", "wide_gate"]
    returns: {route_id, stops[], transfers, estimated_minutes, distance, line_sequence[]}

  - name: fare_calculator
    description: "Calculate fare for a journey"
    params:
      route_id: string
      passengers: {adults: int, children: int, seniors: int, disabled: int}
      ticket_type: enum [single, return, day_pass, weekly, monthly]
      payment_method: enum [smartcard, contactless, cash, mobile]
    returns: {fare_id, line_items[], subtotal, discounts[], total, currency}

  - name: ticket_issuer
    description: "Issue ticket after payment confirmation"
    params:
      fare_id: string
      payment_token: string
    returns: {ticket_id, confirmation_code, issued_at}

  - name: payment_processor
    description: "Process payment"
    params:
      amount_cents: int
      method: enum [contactless, chip_pin, cash, mobile_pay]
    returns: {payment_token, status, receipt_url}

  - name: station_info
    description: "Station facility and accessibility information"
    params:
      station_id: string
      query_type: enum [accessibility, facilities, exits, connections, real_time_status]
    returns: {station_id, data: object}

  - name: disruption_feed
    description: "Current service disruptions and advisories"
    params:
      line: string (optional)
      station: string (optional)
      severity_filter: enum [all, major, minor]
    returns: {disruptions[]: {id, line, segment, type, severity, message, alternative, eta_resolution}}

  - name: knowledge_base
    description: "Query metro system policies, FAQs, regulations"
    params:
      query: string
    returns: {answer: string, source: string, confidence: float}
```

**Constraint**: The LLM has NO tools beyond these 7. Any fabricated tool call is a tool hallucination and scored as a penalty.

---

## 3. Interaction Model

### 3.1 Two Input Modes

**Structured events** (70% of cases): User actions from pre-built UI.

```json
{"event": "station_selected", "field": "origin", "value": "Powell St", "station_id": "BART-POWL"}
{"event": "passenger_count_changed", "adults": 2, "children": 1}
{"event": "disruption_update", "disruption": {...}}
```

**Freetext fallback** (30% of cases): Natural language input for edge cases.

```json
{"event": "freetext_input", "text": "Can I bring my bike on BART?", "language": "en"}
```

### 3.2 Output Format

LLM outputs a JSON response. The benchmark evaluates **intermediate reasoning correctness** separately from **structural validity**.

```yaml
response_schema:
  reasoning:          # Freetext — LLM's internal reasoning about the query
    route_decision: string
    fare_reasoning: string
    warnings_identified: string[]
  
  ui_updates:         # Structured — what changes in the kiosk UI
    advisory_banners:
      - severity: enum [info, warning, critical, positive]
        title: string
        body: string (≤ 2 sentences)
        affected_lines: string[]
        affected_stations: string[]
    route:
      origin: string
      destination: string
      stops: [{station, line, is_transfer, transfer_to, transfer_walk_min, accessibility_issues[]}]
      transfers: int
      estimated_minutes: int
      distance_km: float
    fare:
      items: [{label, amount, currency}]
      total: float
      currency: string
      daily_cap_info: string (optional)
      payment_methods: string[]
    passenger_summary:
      adults: int
      children: int
      seniors: int
      free_riders: int
      note: string (optional, e.g. "Children under 5 ride free")
    assistant_message:
      text: string
      language: string
    ticket_ready: bool
  
  tool_calls: [{name, args}]   # Actual tool invocations made
```

**Key design**: `reasoning` is evaluated for content correctness (right route, right fare logic, right warnings). `ui_updates` is evaluated for structural validity and internal consistency. These are scored independently. A model can score high on reasoning but low on structure (→ needs constrained decoding or rendering layer) or vice versa.

### 3.3 Framebook Integration

Each system has a `framebook.yaml` loaded into the system prompt. It specifies:

```yaml
framebook:
  org_name: "BART"
  primary_language: en
  secondary_languages: [es, zh]
  currency_symbol: "$"
  currency_code: "USD"
  fare_display_format: "$X.XX"
  terminology:
    smartcard: "Clipper card"
    contactless: "Tap and Ride"
    reduced_fare: "Clipper START / Senior / Youth"
  advisory_severity_mapping:
    service_suspended: critical
    major_delay: warning
    minor_delay: info
    planned_works: info
    crowd_advisory: info
  accessibility_labels:
    step_free: "Wheelchair Accessible"
    elevator: "Elevator Available"
    escalator_out: "Escalator Out of Service"
  ui_components_available:
    - route_map
    - fare_breakdown
    - advisory_banner
    - station_selector
    - passenger_counter
    - payment_panel
    - assistant_chat
  cultural_notes: []  # System-specific, e.g. Taipei has no-eating rules
```

Evaluation checks that LLM output uses correct terminology, currency formatting, and language per framebook.

---

## 4. Event Templates

Non-regular events injected into system context. Each template is parameterized.

### 4.1 BART — Earthquake

```yaml
event: earthquake_warning
params:
  magnitude: float (e.g. 4.2)
  epicenter_distance_km: float
  shaking_intensity: enum [light, moderate, strong]
  bart_response:
    initial: "all_trains_hold" (immediate stop wherever they are)
    inspection_duration_minutes: int (15-45)
    phased_resumption: bool
    lines_cleared_first: [underground_segments]
    elevated_delay_additional_minutes: int
  timestamp: ISO 8601
```

### 4.2 Taipei — Typhoon

```yaml
event: typhoon_warning
params:
  typhoon_name: string
  wind_gust_beaufort: int
  avg_wind_beaufort: int
  wenhu_line_status: enum [normal, speed_reduced, suspended]
  elevated_sections_status: enum [normal, speed_reduced, suspended]
  underground_status: enum [normal, reduced_frequency]
  underground_interval_minutes: int (12-30)
  maokong_gondola: enum [normal, suspended]
  youbike: enum [normal, suspended]
  school_work_closure: bool
  flood_risk_stations: string[]
  timestamp: ISO 8601
```

### 4.3 Chicago — Polar Vortex

```yaml
event: polar_vortex
params:
  temperature_f: int
  windchill_f: int
  affected_outdoor_lines: string[] (e.g. [Green, Orange, Pink, Yellow, Purple, Brown])
  outdoor_line_status: enum [normal, reduced, suspended]
  indoor_lines_status: enum [normal] (Red/Blue subway sections always run)
  switch_heater_failures: string[] (specific junction points)
  bus_chains_required: bool
  timestamp: ISO 8601
```

### 4.4 BART — Game Day / Fleet Week

```yaml
event: crowd_surge
params:
  event_name: string
  affected_stations: string[]
  crowd_level: enum [moderate, heavy, extreme]
  expected_duration: {start: ISO8601, end: ISO8601}
  additional_trains: bool
  advisory_message: string
```

### 4.5 Generic — Planned Maintenance

```yaml
event: planned_maintenance
params:
  system: string
  line: string
  segment: [station_id, station_id]
  dates: {start: date, end: date}
  schedule: enum [all_day, nights_weekends, weekends_only]
  replacement_service: enum [bus, shuttle, none]
  fare_adjustment: enum [normal, disruption_fare, free_replacement]
```

### 4.6 Generic — Station Closure

```yaml
event: station_closure
params:
  station_id: string
  reason: string
  duration: enum [temporary, long_term]
  trains_skip: bool (pass through without stopping)
  transfers_affected: string[] (lines that can no longer interchange here)
  nearest_alternatives: string[] (station_ids)
  start: ISO 8601
  end: ISO 8601 (or null for indefinite)
```

---

## 5. Test Cases

### 5.1 Distribution (1000 total)

| Category | MARTA | BART | Taipei | CTA | Total |
|----------|-------|------|--------|-----|-------|
| A: Standard routing | 20 | 30 | 35 | 45 | 130 |
| B: Fare calculation | 15 | 35 | 35 | 30 | 115 |
| C: Disruptions/events | 15 | 30 | 30 | 40 | 115 |
| D: Accessibility | 15 | 20 | 25 | 30 | 90 |
| E: Multilingual/cultural | 5 | 15 | 35 | 15 | 70 |
| F: Policy change (zero-shot) | 10 | 15 | 20 | 20 | 65 |
| G: Multi-turn complex | 15 | 20 | 25 | 30 | 90 |
| H: Adversarial/safety | 15 | 20 | 20 | 25 | 80 |
| I: Temporal reasoning | 10 | 20 | 20 | 25 | 75 |
| J: Tool hallucination | 10 | 20 | 20 | 20 | 70 |
| K: Compound stress | 5 | 10 | 10 | 15 | 40 |
| L: Framebook conformance | 10 | 10 | 15 | 25 | 60 |
| **Total** | **145** | **245** | **290** | **320** | **1000** |

Category J is new — dedicated tool hallucination cases where no appropriate tool exists or only distractor tools are available. Category K contains compound cases combining 3+ failure modes. Category L tests framebook/styleguide adherence specifically.

### 5.2 Route Partitioning

Within categories A, C, G, I: each system's cases are split into:
- **memorizable** (50%): common routes likely in training data
- **novel** (50%): unusual origin-destination pairs unlikely in training data

This distinguishes recall from reasoning.

### 5.3 Test Case Schema

```json
{
  "id": "BART-C-017",
  "system": "bart",
  "category": "C",
  "difficulty": "hard",
  "interaction_mode": "hybrid",
  "route_type": "memorizable",
  "title": "Earthquake hold during active journey",
  
  "events": [
    {"type": "station_selected", "field": "origin", "value": "Powell St", "station_id": "BART-POWL"},
    {"type": "station_selected", "field": "destination", "value": "SFO Airport", "station_id": "BART-SFIA"},
    {"type": "passenger_count_changed", "adults": 1},
    {"type": "disruption_update", "disruption": {
      "template": "earthquake_warning",
      "params": {"magnitude": 4.5, "shaking_intensity": "moderate", "bart_response": {"initial": "all_trains_hold", "inspection_duration_minutes": 25}}
    }}
  ],
  
  "system_context": {
    "current_time": "2025-09-15T16:30:00-07:00",
    "active_disruptions_before": [],
    "framebook": "bart"
  },
  
  "ground_truth": {
    "route_before_disruption": {
      "path": ["Powell St", "Civic Center", "16th St Mission", "24th St Mission", "Glen Park", "Balboa Park", "Daly City", "Colma", "South San Francisco", "San Bruno", "SFO Airport"],
      "line": "SFO/Millbrae",
      "transfers": 0,
      "distance_miles": 14.1
    },
    "fare": {
      "base": 9.65,
      "sfo_surcharge": 4.95,
      "total": 14.60,
      "currency": "USD"
    },
    "post_disruption": {
      "route_still_valid": true,
      "delay_minutes_added": 25,
      "advisory_required": true,
      "advisory_severity": "critical",
      "advisory_must_mention": ["earthquake", "trains held", "inspection"],
      "should_mention_flight": true,
      "ticket_ready": false
    }
  },
  
  "scoring": {
    "route_correct": 15,
    "fare_correct": 20,
    "disruption_detected": 15,
    "advisory_issued": 10,
    "advisory_content_correct": 10,
    "temporal_reasoning": 10,
    "tool_calls_correct": 10,
    "no_tool_hallucination": 5,
    "framebook_conformance": 5
  },
  
  "tolerances": {
    "fare_usd": 0.50,
    "time_minutes": 10,
    "distance_miles": 1.0
  }
}
```

---

## 6. Scoring

### 6.1 Per-Case (100 points)

| Component | Points | Verification |
|-----------|--------|-------------|
| Content: route correctness | 15 | Graph validation: is path valid? Within 20% of optimal? |
| Content: fare correctness | 20 | Fare engine comparison: within tolerance? |
| Content: advisory/warning completeness | 15 | All required warnings issued? Severity correct? |
| Content: accessibility accuracy | 5 | Correct flags per station metadata? |
| Content: temporal reasoning | 5 | Time calculations correct? Last train considered? |
| Tool usage: correct calls | 10 | Right tools, right args, right order? |
| Tool usage: no hallucination | 10 | Zero fabricated tools? (-10 per hallucinated call, floor 0) |
| Structure: schema validity | 5 | Parseable JSON? Conforms to response schema? |
| Structure: internal consistency | 5 | Route and fare agree? Banners match route? |
| Framebook: terminology/language/formatting | 5 | Correct currency, card name, language per framebook? |
| Interaction efficiency | 5 | Completed in ≤ expected turn count? |

### 6.2 First-Class Metrics (reported separately)

- **Task Success Rate (SR)**: % of cases scoring ≥ 70
- **Fare Error Rate (FER)**: % of cases where fare is wrong beyond tolerance
- **Tool Hallucination Rate (THR)**: % of cases with ≥1 fabricated tool call
- **Advisory Miss Rate (AMR)**: % of disruption cases where no advisory was issued
- **Schema Validity Rate (SVR)**: % of cases producing parseable, schema-conformant output
- **Transaction Turn Count (TTC)**: mean turns to complete standard transactions (Cat A)
- **Median TTFT**: time to first token (ms)
- **Median E2E**: end-to-end response time (ms)
- **Tokens per response**: mean output tokens

### 6.3 Composite Score

```
MetroLLM-1000 = weighted_mean(per_case_scores)

System weights: MARTA=0.10, BART=0.25, Taipei=0.25, CTA=0.40
(CTA weighted highest: largest graph, most complex operations)
```

### 6.4 Deployment Readiness

| Class | MetroLLM-1000 | FER | THR | AMR | SVR |
|-------|---------------|-----|-----|-----|-----|
| Production Ready | ≥ 82 | < 2% | < 2% | < 5% | > 95% |
| Pilot Ready | ≥ 70 | < 5% | < 5% | < 10% | > 85% |
| Research Only | ≥ 50 | < 15% | < 15% | < 25% | > 60% |
| Not Viable | < 50 | any | any | any | any |

---

## 7. Model Matrix

### 7.1 FOSS Models (primary targets)

| Model | Active Params | Total Params | Arch | Min GPU (Q4) |
|-------|--------------|-------------|------|-------------|
| Qwen3.5-0.8B | 0.8B | 0.8B | Dense | Any |
| Qwen3.5-2B | 2B | 2B | Dense | Any |
| Qwen3.5-4B | 4B | 4B | Dense | 8GB |
| Qwen3.5-9B | 9B | 9B | Dense | 16GB |
| Qwen3.5-27B | 27B | 27B | Dense | 24GB |
| Qwen3.5-35B-A3B | 3B | 35B | MoE | 24GB |
| Qwen3.5-122B-A10B | 10B | 122B | MoE | 96GB |
| Llama-3.3-70B | 70B | 70B | Dense | 96GB |

### 7.2 Private Baselines (via API)

| Model | Provider |
|-------|----------|
| GPT-4.1 | OpenAI |
| GPT-4o-mini | OpenAI |
| Claude Sonnet 4.6 | Anthropic |
| Claude Haiku 4.5 | Anthropic |
| Gemini 2.5 Pro | Google |

### 7.3 Test Variables

Each FOSS model tested with:
- **Quantization**: highest that fits target GPU (BF16 > FP8 > Q6 > Q4 > Q3 > Q2)
- **Decoding mode**: unconstrained vs schema-constrained (guided_json)
- **Thinking mode**: enable_thinking=false (default) vs enable_thinking=true (subset of Cat G, I, K only)

3 runs per configuration (temperature=0.7 introduces variance). Report mean ± std.

### 7.4 Reference Configuration

```yaml
reference:
  model: Qwen3.5-35B-A3B
  quantization: UD-Q4_K_XL
  gpu: RTX 5090 (32GB)
  serving: llama-server
  context: 16384
  parallel: 2
  thinking: false
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  expected_tok_s: ~180
```

---

## 8. Evaluation Harness

### 8.1 Runner

Input: OpenAI-compatible endpoint URL, model name, test cases JSON.
Process:
1. Load system context (graph, framebook, active events) into system prompt
2. For each case: send event sequence as user messages, collect responses
3. Record: response JSON, TTFT, E2E latency, token counts, raw tool calls
4. Output: results JSON with raw responses + timing

### 8.2 Scorer

Input: results JSON, ground truth JSON.
Process per case:
1. **Parse response**: attempt JSON parse → schema validate → extract fields
2. **Route check**: validate against graph (is path connected? are consecutive stops adjacent?)
3. **Fare check**: compute expected fare from fare engine, compare to response fare
4. **Advisory check**: compare issued banners against required banners from ground truth
5. **Tool call check**: compare actual tool calls against expected sequence; flag any calls to non-existent tools
6. **Framebook check**: verify terminology, currency format, language
7. **Consistency check**: route distance matches fare tier? banner affected_stations match route?
8. **Score**: apply rubric, output per-case score breakdown

### 8.3 Reporting

Output: leaderboard JSON with:
- per-model: composite score, all first-class metrics, per-category breakdown, per-system breakdown
- per-case: score, failure reasons, response excerpt
- scaling_curve: active_params vs MetroLLM-1000 (with confidence intervals)
- pareto_frontier: cost_per_query vs quality (FOSS and private on same chart)

---

## 9. Leaderboard Site

Static site generated from results JSON.

Pages:
- **Leaderboard**: sortable table of all models with MetroLLM-1000, per-system scores, key metrics
- **Model detail**: per-category radar chart, failure case examples, latency distribution
- **Scaling curves**: interactive chart of score vs active params, with category breakdowns
- **System detail**: per-metro-system results, hardest cases, event handling performance
- **Methodology**: link to this spec document

---

## 10. Build Order

1. **Data: Station graphs** — MARTA (38), BART (50), Taipei (131), CTA (146). JSON adjacency lists with distances, transfer times, accessibility metadata. Verify all edges bidirectional, distances consistent.

2. **Data: Fare engines** — Python module per system. Deterministic. Unit tested against known fare pairs from official calculators.

3. **Data: Framebooks** — YAML per system. Terminology, language rules, formatting conventions.

4. **Data: Event templates** — YAML. Parameterized. Include 3-5 concrete instantiations each.

5. **Cases: Generator** — Script that produces test cases from graphs + fare engines + event templates. Computes ground truth automatically. Outputs cases.json.

6. **Cases: Manual review** — Human review of ~100 cases for correctness. Fix any graph/fare errors.

7. **Harness: Runner** — Python. OpenAI-compatible. Records responses + timing.

8. **Harness: Scorer** — Python. Deterministic scoring against ground truth. Outputs results JSON.

9. **Run: Reference model** — Qwen3.5-35B-A3B on RTX 5090. Full 1000 cases × 3 runs.

10. **Run: Model matrix** — Remaining FOSS models + private baselines.

11. **Site: Leaderboard** — Static site from results.

---

## 11. Extensibility

New metro systems added by providing:
1. `stations.json` — id, name, name_local, line, lat/lon, accessibility features
2. `lines.json` — id, name, color, stations (ordered), type (subway/elevated/at_grade)
3. `graph.json` — edges with distances and transfer walk times
4. `fares.json` — fare rules (flat, distance, zone, matrix) + calculator params
5. `framebook.yaml` — org-specific UI/UX guide
6. `events/` — 3+ event templates with concrete instantiations
7. Run case generator → produces cases for new system → merge into cases.json

Planned Phase 2 systems: NYC Subway, London Underground, Tokyo Metro, Berlin U-Bahn, Paris Métro, Beijing Subway, Amsterdam GVB, Madrid Metro, São Paulo Metrô, Hangzhou Metro.