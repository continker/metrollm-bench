import type { DashboardData } from "../types";
import { navigate } from "../router";
import { runColor, scoreColor } from "../utils/colors";
import { fmtPct, fmtMs } from "../utils/format";

const CAT_INFO: Record<string, { name: string; color: string; what: string; how: string; tools: string[]; components: string[][] }> = {
  A: {
    name: "Routing",
    color: "#388e3c",
    what: "Given an origin station and destination station, plan the best route through the MARTA rail network. The LLM receives structured events (station_selected, passenger_count_changed) and must call the route_planner tool, then format the result into a kiosk display.",
    how: "Cases range from single-line trips (Airport to Five Points, all on Red) to cross-network journeys requiring transfers at Five Points (e.g. Indian Creek to Sandy Springs: Blue line to Five Points, transfer to Red). The station graph has 38 stations across 4 lines with 35 edges.",
    tools: ["route_planner", "fare_calculator", "submit_response"],
    components: [
      ["route_correct", "15", "Transfers exact, distance within 2mi, line sequence matches"],
      ["fare_correct", "20", "Total within $0.50 (simple cases: 1 adult = $2.50)"],
      ["tool_calls_correct", "10", "Called route_planner (required)"],
      ["no_tool_hallucination", "10", "No fabricated tool names"],
      ["schema_validity", "5", "Valid JSON with reasoning + ui_updates fields"],
      ["framebook_conformance", "5", "Uses $ currency, mentions Breeze Card, no wrong transit terms"],
    ],
  },
  B: {
    name: "Fare calculation",
    color: "#1565c0",
    what: "Given a route and a passenger mix (adults, children, seniors, disabled passengers), calculate the correct total fare. MARTA uses a flat fare model: $2.50/adult, children under 5 ride free (max 2 per paying adult), seniors and disabled passengers pay $1.25.",
    how: "Cases test increasingly complex passenger combinations: single adult, families with free-riding children, groups mixing seniors and disabled passengers, and edge cases like 0 adults with 2 children (both free) or exceeding the 2-free-children-per-adult limit.",
    tools: ["route_planner", "fare_calculator", "submit_response"],
    components: [
      ["route_correct", "15", "Route still scored (cases include a route)"],
      ["fare_correct", "20", "Fare total within $0.50 tolerance; exact match = full marks"],
      ["tool_calls_correct", "10", "Called both route_planner and fare_calculator"],
      ["no_tool_hallucination", "10", "No fabricated tool names"],
      ["schema_validity", "5", "Valid JSON with reasoning + ui_updates"],
      ["framebook_conformance", "5", "Uses $, Breeze Card terminology"],
    ],
  },
  C: {
    name: "Disruption handling",
    color: "#e65100",
    what: "Route + fare with an active disruption. The system prompt mentions active disruptions, and the LLM must call disruption_feed to get details, then adapt: re-route around closures, issue advisories for maintenance, or declare full suspension during hurricanes.",
    how: "15 cases across 3 disruption types: station closures (5 cases, the model must re-route around the closed station), planned maintenance (5 cases, reduced service with advisories), and hurricane warnings (5 cases, full system suspension where no route/fare should be shown). The key test: does the model check for disruptions and act on them?",
    tools: ["route_planner", "disruption_feed", "fare_calculator", "submit_response"],
    components: [
      ["route_correct", "10", "Route adapted to disruption (or omitted for full suspension)"],
      ["fare_correct", "10", "Fare correct for disrupted route (or omitted)"],
      ["tool_calls_correct", "10", "Called disruption_feed + route_planner (or just disruption_feed for suspension)"],
      ["no_tool_hallucination", "10", "No fabricated tool names"],
      ["schema_validity", "5", "Valid JSON with reasoning + ui_updates"],
      ["framebook_conformance", "5", "Uses $, Breeze Card terminology"],
      ["disruption_detected", "15", "Called disruption_feed tool"],
      ["advisory_issued", "10", "Advisory banner with correct severity (warning/critical)"],
      ["advisory_content_correct", "10", "Advisory mentions required keywords (station name, closure reason, etc.)"],
    ],
  },
  D: {
    name: "Accessibility",
    color: "#7b1fa2",
    what: "Route + fare with a passenger who has an accessibility requirement (wheelchair, step-free access). The LLM must call station_info to check elevator/step-free status at each station on the route, and warn the passenger about any issues.",
    how: "15 cases in 3 tiers: 5 happy-path (all stations accessible), 5 pass-through (route crosses a station with elevator out of service), 5 destination-out (the destination itself has no elevator). The model must call station_info with query_type 'accessibility' and correctly report issues.",
    tools: ["route_planner", "station_info", "fare_calculator", "submit_response"],
    components: [
      ["route_correct", "15", "Transfers, distance, line sequence match ground truth"],
      ["fare_correct", "20", "Fare total within $0.50 tolerance"],
      ["tool_calls_correct", "10", "Called route_planner and station_info"],
      ["no_tool_hallucination", "10", "No fabricated tool names"],
      ["schema_validity", "5", "Valid JSON with reasoning + ui_updates"],
      ["framebook_conformance", "5", "Uses $, Breeze Card terminology"],
      ["accessibility_accuracy", "10", "Called station_info for accessibility + correctly identified issues (or lack thereof)"],
    ],
  },
};

export function renderCategory(
  container: HTMLElement,
  data: DashboardData,
  params: Record<string, string>,
): void {
  const cat = params.cat?.toUpperCase() || "A";
  const info = CAT_INFO[cat];
  if (!info) {
    container.textContent = `Unknown category: ${cat}`;
    return;
  }

  const runIds = Object.keys(data.runs);
  const caseIds = data.case_order.filter((id) => data.cases[id].category === cat);
  const maxPossible = info.components.reduce((sum, c) => sum + parseInt(c[1]), 0);

  // Build the page
  const wrapper = document.createElement("div");
  wrapper.className = "category-detail";

  // Header
  wrapper.innerHTML = `
    <div class="cat-detail-header">
      <a class="back-link" href="#/overview">&larr; Overview</a>
      <div class="cat-detail-title">
        <span class="cat-badge cat-${cat}" style="font-size: 0.9rem; padding: 0.2rem 0.6rem;">Cat ${cat}</span>
        <h1>${info.name}</h1>
      </div>
    </div>

    <section class="cat-section">
      <h2>What it tests</h2>
      <p>${info.what}</p>
    </section>

    <section class="cat-section">
      <h2>How cases vary</h2>
      <p>${info.how}</p>
    </section>

    <section class="cat-section">
      <h2>Expected tool sequence</h2>
      <div class="tool-sequence">
        ${info.tools.map((t) => `<span class="seq-node">${t}</span>`).join('<span class="seq-arrow">&rarr;</span>')}
      </div>
    </section>

    <section class="cat-section">
      <h2>Scoring rubric</h2>
      <p class="rubric-total">Maximum: ${maxPossible} points per case</p>
      <table class="summary-table">
        <thead><tr><th>Component</th><th>Max</th><th>Criteria</th></tr></thead>
        <tbody>
          ${info.components.map(([name, max, criteria]) => `
            <tr><td><code>${name}</code></td><td>${max}</td><td>${criteria}</td></tr>
          `).join("")}
        </tbody>
      </table>
    </section>

    <section class="cat-section">
      <h2>Results: ${caseIds.length} cases across ${runIds.length} runs</h2>
      <div class="case-results-table-wrap"></div>
    </section>
  `;

  container.appendChild(wrapper);

  // Build case results table
  const tableWrap = wrapper.querySelector(".case-results-table-wrap")!;
  const table = document.createElement("table");
  table.className = "summary-table case-results-table";

  // Header
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headerRow.innerHTML = `<th>Case</th><th>Title</th>`;
  for (const rid of runIds) {
    const run = data.runs[rid];
    const color = runColor(run.thinking);
    headerRow.innerHTML += `<th style="color:${color}">${run.label.replace("V3 ", "")}</th>`;
  }
  headerRow.innerHTML += `<th>Avg</th>`;
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Body
  const tbody = document.createElement("tbody");
  for (const caseId of caseIds) {
    const tr = document.createElement("tr");
    const caseMeta = data.cases[caseId];

    // Case ID cell — clickable
    const tdId = document.createElement("td");
    const idLink = document.createElement("a");
    idLink.href = `#/replay/${runIds[0]}/${caseId}`;
    idLink.textContent = caseId;
    idLink.style.color = "var(--accent)";
    idLink.style.textDecoration = "none";
    tdId.appendChild(idLink);
    tr.appendChild(tdId);

    // Title
    const tdTitle = document.createElement("td");
    tdTitle.textContent = caseMeta.title;
    tdTitle.style.maxWidth = "200px";
    tdTitle.style.overflow = "hidden";
    tdTitle.style.textOverflow = "ellipsis";
    tdTitle.style.whiteSpace = "nowrap";
    tr.appendChild(tdTitle);

    // Score per run
    const pcts: number[] = [];
    for (const rid of runIds) {
      const s = data.runs[rid].scores[caseId];
      const td = document.createElement("td");
      if (s && s.max_possible > 0) {
        const pct = (s.total / s.max_possible) * 100;
        pcts.push(pct);
        td.textContent = fmtPct(pct);
        td.style.color = scoreColor(pct / 100);
        td.style.cursor = "pointer";
        td.title = `${s.total}/${s.max_possible} — click for replay`;
        td.addEventListener("click", () => navigate(`/replay/${rid}/${caseId}`));
      } else {
        td.textContent = "—";
      }
      tr.appendChild(td);
    }

    // Average
    const tdAvg = document.createElement("td");
    if (pcts.length > 0) {
      const avg = pcts.reduce((a, b) => a + b, 0) / pcts.length;
      tdAvg.textContent = fmtPct(avg);
      tdAvg.style.fontWeight = "600";
      tdAvg.style.color = scoreColor(avg / 100);
    } else {
      tdAvg.textContent = "—";
    }
    tr.appendChild(tdAvg);

    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  tableWrap.appendChild(table);

  // Link row: view in heatmap
  const linkRow = document.createElement("div");
  linkRow.style.marginTop = "1rem";
  linkRow.style.display = "flex";
  linkRow.style.gap = "1rem";
  linkRow.innerHTML = `
    <a href="#/heatmap" class="cat-detail-link">View in Heatmap &rarr;</a>
    <a href="#/comparison" class="cat-detail-link">Compare runs &rarr;</a>
    <a href="#/toolflow" class="cat-detail-link">Tool call patterns &rarr;</a>
  `;
  tableWrap.appendChild(linkRow);
}
