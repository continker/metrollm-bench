# Design Decisions

This document records key design decisions for the MetroLLM-Bench benchmark, intended for the research publication.

## D1: Approach A (tools unaware) for Category F policy changes

**Decision**: Policy overrides are injected into the system prompt only. Tools (`fare_calculator`) continue to return the old fare. The LLM must recognize the conflict and submit the policy-adjusted fare.

**Alternatives considered**:

- **B: Tool-aware** — Policy modifies FareCalculator at runtime. Tools return the new fare. LLM just relays. *Rejected*: does not test reasoning — a weak model that blindly relays tool output scores identically to a strong model.
- **C: Hybrid** — Policy in prompt + tools parameterized by policy context. *Rejected*: more complex, still doesn't isolate prompt comprehension as the variable.

**Rationale**:
1. Simplest for the operator — editing one line of prompt text is the entire workflow.
2. Tests the thesis directly — the prompt IS the authoritative source. Tools are utilities.
3. Discriminative power — Categories A/B cannot distinguish a model that blindly relays tool output from one that reasons about instructions. Category F can.
4. Realistic — mirrors how real LLM deployments work (system prompt overrides, backend lags behind policy).

The "tools contradict the prompt" concern is the feature, not a bug.

## D2: Tree topologies (MARTA/Doha) vs cyclic graphs (BART)

**Decision**: Include both tree-topology systems (MARTA, Doha) and a cyclic-graph system (BART) to vary routing difficulty.

**Implication**: On tree topologies, there is exactly one path between any station pair. Route correctness tests relay accuracy, not path-finding reasoning. Disruption re-routing is degenerate (no detours exist). BART's SFO–San Bruno–Millbrae triangle provides the first cycle, enabling meaningful alternative-route scoring.

## D3: No new tools for BART — universal tool design

**Decision**: The same 5 tools (route_planner, fare_calculator, station_info, disruption_feed, submit_response) serve all three systems without modification.

**Rationale**: Validates the thesis that the tool interface is system-agnostic. System-specific behavior (distance-based fares, surcharges, multiplier discounts) is encoded in data files and resolved by the tool server at runtime.

## D4: Simplified BART fares (brackets + surcharges vs real fare matrix)

**Decision**: Model BART's distance-based fares as 6 brackets with 3 surcharges (Transbay, SFO, OAK), rather than the full 50×50 OD fare matrix.

**Rationale**: The real BART fare matrix has station-pair-specific fares that don't follow a clean distance function. The bracket model captures the key complexity (distance sensitivity, surcharge stacking, multiplier discounts for seniors/disabled) while remaining auditable and reproducible. The benchmark tests whether the LLM can reason about fare structure, not whether it memorized a lookup table.

## D5: Same-line branch transfers modeled as seamless

**Decision**: Branch junctions (Doha Red Line airport branch at Oqba Ibn Nafie, Taipei Red Line Xinbeitou branch at Beitou, Taipei Green Line Xiaobitan branch at Qizhang) are modeled as zero-cost forks within the same line. No transfer penalty is applied.

**Reality**: Passengers must change trains at the junction station (cross-platform transfer, ~5–10 min wait). The mainline train does not continue onto the branch — a separate branch service operates.

**Rationale**: The expanded line graph uses `(station, line)` nodes for transfer-aware routing. Since the main line and branch share the same line ID, no transfer edge is generated at the fork. Introducing sub-line service IDs (e.g., `red_main` vs `red_airport`) would correctly model the platform transfer but adds complexity disproportionate to the benchmark's scope. The benchmark tests whether the LLM identifies the correct path and fare — the platform transfer is operational detail a kiosk display would not typically surface. Affects 3 branch junctions across 2 of 5 systems (Doha, Taipei).

## D6: Purple Line Express modeled at local-train speed

**Decision**: The CTA Purple Line Express (Howard–Wilson non-stop) uses the same travel time per mile as Red Line local service on the shared track, despite the express skipping intermediate stops.

**Reality**: Express service is faster because it does not decelerate/stop at intermediate stations. Real-world Howard–Wilson is ~8 min express vs ~15 min local.

**Rationale**: Modeling per-stop dwell times requires timetable-level data beyond what the benchmark's geographic graph provides. The routing algorithm correctly prefers Purple Express for transfers (fewer stops = fewer nodes traversed) but does not reflect the time advantage. This is a known simplification — the benchmark evaluates tool use and reasoning, not schedule fidelity.

## D7: Thinning removed — all stations retained

**Decision**: All stations from the wiki source data are retained in the benchmark graph. No intermediate stations are collapsed.

**Alternatives considered**: Thinning (collapsing single-line pass-through stations into longer edges) was initially used to reduce station counts. This was removed because: (1) collapsed edges produced inflated distances due to haversine triangle inequality, (2) it complicated coordinate validation, and (3) it made the route planner return unrealistic stop lists. With Wikipedia-sourced coordinates for all stations, real haversine distances are computed directly and no thinning is needed.

**Station counts**: MARTA 38, BART 50, CTA 142, Taipei 107, Doha 37, Beijing 414 (total: 788).

## D8: Single-journey fares only

**Decision**: All fare calculations assume single-journey tickets. Day passes, monthly caps, and multi-ride bundles are out of scope.

**Critics may argue**: Real ticketing systems offer passes, loyalty discounts, and accumulated-spend caps (e.g. Beijing Yikatong's ¥100/month threshold for 50% off). Testing only single fares limits ecological validity.

**Rationale**: Single-journey fares are stateless — each kiosk interaction is independent. Pass economics require session state (how many rides this week/month, cumulative spend) that contradicts the benchmark's single-interaction design. The benchmark tests whether an LLM can reason about fare structure (distance brackets, surcharges, passenger-type discounts, policy overrides), not whether it can maintain a running account balance. Category F policy changes ("all fares half price today") test the same arithmetic reasoning that passes would, without requiring persistent state.

**Future work**: A multi-session extension could test pass recommendations ("you've made 18 rides this month — a weekly pass would save you ¥X"), but this requires a fundamentally different harness architecture.

## D9: English response language regardless of system locale

**Decision**: The system prompt instructs "Respond in English (the local language is {primary_language})" for all systems, even Beijing (zh) and Taipei (zh-TW).

**Discovery**: GPT-5-mini responded in Chinese for Beijing, triggered by `Language: zh` in the system prompt. This caused framebook_conformance to fail 97% of cases (scorer checks for English smartcard name "Yikatong Card", model used 一卡通) and degraded the LLM judge's ability to evaluate responses.

**Rationale**: The benchmark's scoring pipeline (keyword matching, terminology checks, Claude Haiku judge) is English-centric. Building equivalent Chinese/Arabic scoring would be a separate research effort. Forcing English keeps measurement consistent across systems. The same issue exists latently for Doha (Arabic) and Taipei (Mandarin) — models happened to respond in English for those systems but could switch unpredictably.

**Impact**: Beijing tier 1 improved from 88.5% to 90.5% (+2.0) with the English prompt, confirming this was a measurement artifact, not a capability difference. Taipei composite improved +4.3 (87.3%→91.6%) from the same fix.

**Limitation**: A production Beijing kiosk would need Chinese responses. The benchmark does not evaluate multilingual generation quality — Category E tests multilingual *comprehension* (can the model understand a query in Japanese/Korean/Chinese?) but always expects English output.

## D10: Network scale does not materially degrade performance

**Finding**: Beijing Metro (414 stations, 27 lines) scores 90.5% tier 1 with GPT-5-mini, within 0.6 points of the 5-system average (91.1%). The 4× increase in station count from the largest previous system (CTA, 142 stations) produces negligible performance degradation.

**Explanation**: The model never sees all 414 stations simultaneously. Tool responses (route_planner, station_info) return only the relevant subset. The routing graph complexity is handled server-side by the tool server, not by the LLM. Network scale primarily affects: (a) tool response payload size (more stops in a route), (b) station name confusion (more similar-sounding names), and (c) the model's ability to resist fabrication (larger name space to hallucinate from).

**Implication**: The benchmark's tool-mediated architecture scales to real-world network sizes. The binding constraint is contract adherence and domain reasoning, not network memorization.

## D11: Airport express surcharges as base-fare replacement

**Decision**: Beijing's Capital Airport Express (¥25) and Daxing Airport Express (¥35) replace the distance-based bracket fare entirely, rather than adding a surcharge on top (as BART does for SFO/OAK).

**Implementation**: The surcharge engine supports a `replaces_base: true` flag. When set, the surcharge amount becomes the per-ride fare, ignoring the distance bracket. This models the real-world behavior where airport express lines have flat fares independent of distance.

**Rationale**: Beijing's airport express is a separate fare product — passengers tap in and out on the express line at a fixed price. This differs from BART where the airport surcharge is additive to the distance fare. The generic surcharge engine now supports both patterns through configuration, not code changes — validating the thesis that system-specific fare rules can be data-driven.

## D12: Route in advisory_only is informational context, not a travel recommendation

**Decision**: When a disruption makes a route unusable and the model submits `outcome: advisory_only`, the presence of `route.stops` is permitted. The scorer does not penalize a route that traverses closed segments under `advisory_only`.

**Rationale**: The `outcome` field is the authority on whether a route is actionable. `advisory_only` explicitly signals that normal service is disrupted. The route, when present, serves as spatial context — it shows the passenger what the normal path is and which segment is affected. Removing it would leave the advisory as an abstract statement with no geographic reference. Real transit systems (e.g., TfL) follow this pattern: show the normal route with a disruption overlay. The scorer already validates disruption handling through `outcome_correct`, `advisory_issued`, and `advisory_content_correct`. Adding a route-vs-closure cross-check would penalize useful UX for marginal signal.

## D13: Hybrid knowledge_base with policy index in prompt

**Decision**: The system prompt includes a one-line index of all available policies (ID + title). The `knowledge_base` tool accepts `policy_id` for exact lookup alongside the existing `query` parameter for keyword search. The category pre-filter is removed.

**Rationale**: With 12 policies per system, search is the wrong abstraction — a catalog is sufficient. Injecting titles adds ~200 tokens (negligible). The model sees what's available and picks by ID, eliminating matching fragility. This was validated in smoke tests: GPT-5-mini consistently used `policy_id` for exact lookups across Beijing, MARTA, and CTA. In the Beijing child-fare case, the model used the KB policy to override a buggy fare_calculator label — demonstrating the hybrid design where policies are authoritative and tools are computational.

## D14: Fare discount keys are passenger-type, not rule-descriptive

**Decision**: Discount keys in `fares.json` use the passenger type (`children`, `senior_65_plus`, `disabled`) with a separate `qualifier` field for system-specific rules (e.g., `"qualifier": "under 120cm"` for Beijing, `"qualifier": "under 7"` for CTA).

**Rationale**: The previous rule-descriptive keys (`children_under_5`, `children_under_7`, `children_under_120cm`) required the fare calculator to know which key each system used — a hardcoded mapping that silently failed for Beijing. Passenger-type keys are a direct dict lookup (`discounts.get("children", {})`) with no string matching. The `qualifier` field drives the label shown to passengers.

## D15: No prompt fix or guardrail for Cat H terminal submission failures

**Decision**: When a model responds to an impossible request with plain text instead of calling `submit_assistant_state`, the benchmark penalizes this as-is. No prompt example, no runner guardrail.

**Alternatives rejected**:

- **Prompt example** (submit refusal template): Tested during v18 prompt ablation. Submit examples and other prompt additions were net negative on 4B — each change that helped one category hurt another. All prompt changes reverted to v17.
- **Runner guardrail** (inject synthetic submit or force another round): Would mask the behavioral gap and inflate scores. The failure to comply with the API contract is the signal, not noise.

**Rationale**: Cat H measures whether RLHF-trained helpfulness instincts override explicit system-prompt constraints. Models that understand the impossibility but choose conversational clarification over terminal submission demonstrate a real production failure mode — the kiosk pipeline cannot render plain-text responses. The 25/75 point penalty is correct and discriminative. PEFT training data includes proper refusal examples from compliant models (27B/35B submit `request_declined` reliably), so fine-tuning should close this gap without prompt engineering.

## D16: Cat D accessibility action instability is accepted variance

**Decision**: When the model correctly identifies an accessibility issue (`reason_code: accessibility_issue`) but varies between `display_info` and `refer_to_staff` across identical reruns, the scorer does not special-case this. The ground truth pins one action and the 2.5-point penalty for mismatches stands.

**Rationale**: Both actions are defensible for accessibility warnings — "show the info" and "refer to staff" are operator policy decisions, not objective facts. However, pinning one action implies a policy judgment the benchmark shouldn't make, and accepting either adds special-case logic for a 2.5-point component. Over 15 Cat D cases per system across a full matrix, this variance is statistical noise that averages out. The instability is a known property of temperature-induced classification boundary variance, not a scoring defect.

## D17: cultural_accuracy moved from LLM judge to deterministic keyword check (Tier 1)

**Decision**: `cultural_accuracy` is scored by case-insensitive substring presence of `must_mention` keywords in the response text. The LLM judge no longer evaluates this rubric.

**Alternatives rejected**:
- **Keep LLM judge with tightened prompt**: Haiku consistently hallucinated requirements (e.g. "elaborate on the cultural significance of Banqiao and Shilin") not present in the rubric, penalising correct terse responses. Adding "do not penalise for lack of elaboration" clauses reduced but did not eliminate the pattern.
- **Hybrid keyword + judge**: Adds noise without adding signal — the rubric's ground truth is a `must_mention` list, which is exactly what a keyword check evaluates.

**Rationale**: Category E tests whether the model recognises culturally-specific terminology (Doha gold class, Taipei no-eating rules, Beijing Yikatong, Chinese/Arabic station names). The ground truth is structured data: a list of required mentions. An LLM judge adds interpretive variance without recovering any signal the keyword check misses. This is consistent with tau-bench's finding that rule-based evaluation outperforms LLM judges on objective verification tasks (Yao et al., 2406.12045).

**Impact on calibration**: cultural_accuracy exact agreement with human annotator rose from 73% to 93% after the change.

## D18: LLM judge rubric hardening with explicit ceilings and quantitative thresholds

**Decision**: All LLM judge rubrics in `harness/judge.py` carry explicit "do not penalise" clauses, bounded success criteria, and (where applicable) quantitative thresholds:
- **advisory_content**: do not require apology language or background elaboration
- **policy_acknowledged**: do not require verbatim quotation of the policy
- **temporal_accuracy**: last-train warnings are required only when service ends within 60 minutes of current time; otherwise omitting a warning is not a deduction
- **safety_response**: a short clear rejection is a full-mark answer; no extensive alternatives or apologies required
- **scope_adherence**: referring the user to station staff or customer service hotline is within scope

**Rationale**: Academic literature on LLM judges (MT-Bench, Zheng et al. 2306.05685; JudgeBench, Tan et al. 2410.12784) identifies verbosity bias and rubric hallucination — the judge invents requirements not in the prompt, shaped by RLHF helpfulness norms. Explicit ceilings and "do not penalise" clauses reduce this directly. In our calibration, the three temporal disagreements where the human accepted a response without a last-train warning at >60 min resolved correctly after the threshold was added to the judge prompt. Two safety disagreements where the judge hallucinated a station-name-validity problem resolved after the rubric was updated to use the ground-truth `rejection_reason` instead of concatenating event fields.

## D19: submit_assistant_state kiosk_action structural check loosened for service-unavailable cases

**Decision**: When `service_available = False` (Cat I), the structural check in `temporal_accuracy` passes as long as `kiosk_action.action ≠ prompt_purchase`. Previously, only `block_purchase` earned the 3-point structural score.

**Rationale**: The model legitimately varies between `block_purchase`, `display_info` (showing first-train info), and `refer_to_staff` when service is currently unavailable. All three correctly decline to sell a ticket; only `prompt_purchase` is actually wrong. Human annotators consistently accepted these variants as correct kiosk behaviour while the old check penalised them. The loosened rule matches human judgment and removes 4 structural-driven disagreements from the calibration set.

## D20: Advisory banners elevated to always-on prompt instruction

**Decision**: The system prompt includes an "Advisory Banners" section unconditionally, describing banner severity levels (critical/warning/info/positive) and encouraging trip-specific banners. Previously banners were mentioned only in conditional prompt sections (Cat C disruptions, Cat D accessibility).

**Rationale**: Banners are a primary passenger-facing information surface on a real kiosk — short, severity-tagged, visually prominent. Restricting prompt guidance to disruption cases meant models produced inconsistent banner use across categories: Beijing cases surfaced security/payment banners (via cultural_notes in the framebook) but other systems rarely did. Making the instruction always-visible yields more consistent banner quality across all case types. In a 5-case smoke test, GPT-5-mini produced 2–3 personalised banners per case covering operating hours, accessibility notes, payment methods, security/ID requirements, and policy exceptions. The change is neutral on Tier 1 and aligns with the human annotator's observation that banner content carries a majority of the information-conveying signal in their judgments.

## D21: PEFT evaluation requires n=2 independent training seeds (multi-seed CI)

**Decision**: Every reported PEFT delta is the mean across two independent training seeds (seed=42, seed=43). The two seeds vary LoRA initialization, optimizer randomness, and the train/val split simultaneously, capturing the full variance of the PEFT training procedure. Per-size seed-spread is reported alongside the mean. The `--seed` flag in `scripts/peft/train.py` propagates a single integer to all three downstream randomness sources.

**Rationale**: The original capacity-ceiling claim (single-seed: 27B regression of −1.6 tier-1 vs base) was at roughly 1.6 standard errors on a single-run noise model — directional but not statistical-significance-grade. A second seed converts the claim from "we observed a regression" to "the regression reproduces independently". The n=2 result holds: seed=42 −1.62 T1, seed=43 −1.68 T1, agreement within 0.06 T1 — refutes both single-seed-noise and (combined with the v17→v23 retraining experiment) distribution-shift explanations of the 27B regression.

**Secondary finding from the methodology**: seed-to-seed variance shrinks monotonically with model size (2B ±1.81, 4B ±0.77, 9B ±0.13, 27B ±0.06). PEFT training is a "seed lottery" at small scale and effectively deterministic at large scale. This pattern parallels the capacity-ceiling curve and is a unified statement about adapter dynamics — both "room for value" and "room to vary" vanish together as base capability grows. Multi-seed evaluation is non-optional at the small-model end, where seed=43 outperformed seed=42 on 2B by +1.82 T1 cross-system (5/6 systems improved); without n=2 we would have under-reported the 2B endpoint by 0.91 T1 in the cross-system mean.

**Cost**: ~24 hours of additional RTX 5090 time (4B+9B+27B+2B retrain + bench at seed=43). Acceptable given the load-bearing nature of the capacity-ceiling claim. Beyond n=2, marginal information per additional seed is lower than the wallclock cost on this benchmark, but paired McNemar / paired bootstrap on per-case correctness is the natural next significance test and is noted as future work.

## D22: Hardware deployment envelope is measured on Apple Silicon at canonical "lid-open AC-power" state

**Decision**: The §6 "Hardware deployment envelope" claim is grounded in two measurements: a 15-case stratified MARTA probe and a 45-minute sustained-load thermal characterisation on a MacBook Air M2 (Apple M2, 8 GPU cores, 16 GB unified memory, 100 GB/s, fanless), paired with the same probe set on a MacBook Pro M2 Max (96 GB, 400 GB/s, fan-cooled) baseline. Both run identical Q4_K_M GGUFs (`Qwen3.5-2B-metro-v24-Q4_K_M.gguf` from `continker/Qwen3.5-2B-metro-v24` on HF) via `llama.cpp` Metal backend at `ctx-size=32768`, `parallel=1`, `temperature=0`. The Air is measured with lid open and on AC power.

**Rationale**: The deployment claim is "model X runs adequately on hardware Y" — the data has to come from Y, not from extrapolation. Bandwidth-projected decode rates from the published spec (M2 Air at 100 GB/s should yield ~25-30 tok/s on a 1.2 GB Q4 weight) under-predicted the actual rate (39 tok/s sustained, 46 cold) by a meaningful margin, indicating Apple's effective bandwidth utilisation is better than the simple model. Conversely, TTFT predictions from prompt-eval-rate × prompt-length under-predicted observed TTFT (5.2 s observed vs ~1 s predicted) — the gap is HTTP overhead plus runner round-trips plus reasoning_content generation in the thinking-mode path. Only direct measurement captures the user-facing latency.

**Lid-open AC-power is the canonical state**: closing the lid (clamshell, external display) traps heat against the keyboard and induces 10-15 % more aggressive throttle than lid-open. Battery operation will throttle further as the SoC voltage limits drop. We measure the most-favourable realistic deployment configuration; clamshell or battery-only would worsen the curve. A future revision could measure clamshell as a pessimistic bound, but the lid-open AC number is the right anchor for "this is what you can rely on" deployment claims.

**Same-GGUF determinism**: Tier-1 87.9 reproduces exactly between M2 Max and M2 Air on the n=15 probe set. This is a useful methodological anchor — it means deployment selection is a pure throughput / TTFT / RAM trade-off, not a quality regression. Per-case raw outputs differ across hardware in subtle floating-point ways (different Metal kernel scheduling), but at temperature 0 and on a tier-1 deterministic schema, the aggregated score is identical. Published deployment numbers should report the same Tier-1 across all silicon variants and let the throughput table do the differentiation.
