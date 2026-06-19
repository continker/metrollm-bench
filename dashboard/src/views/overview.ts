import type { DashboardData } from "../types";
import { navigate } from "../router";
import { runColor } from "../utils/colors";
import { fmtPct, fmtMs, fmtTokens } from "../utils/format";

export function renderOverview(container: HTMLElement, data: DashboardData) {
  const d = data;
  const agg = d.aggregates;
  const tvn = agg.thinking_vs_non_thinking;

  const runIds = Object.keys(d.runs);
  const thinkingRuns = runIds.filter((r) => d.runs[r].thinking);
  const nonThinkingRuns = runIds.filter((r) => !d.runs[r].thinking);

  // Compute averages
  const thinkingPcts = thinkingRuns.map((r) => agg.by_run[r].overall_pct);
  const nonThinkingPcts = nonThinkingRuns.map((r) => agg.by_run[r].overall_pct);
  const thinkingAvg = thinkingPcts.reduce((a, b) => a + b, 0) / thinkingPcts.length;
  const nonThinkingAvg = nonThinkingPcts.reduce((a, b) => a + b, 0) / nonThinkingPcts.length;

  // Category averages
  const cats = ["A", "B", "C", "D"];
  const catLabels: Record<string, string> = {
    A: "Routing",
    B: "Fare calculation",
    C: "Disruption handling",
    D: "Accessibility",
  };
  const catThinking: Record<string, number> = {};
  const catNonThinking: Record<string, number> = {};
  for (const cat of cats) {
    const tVals = thinkingRuns.map((r) => agg.by_run[r].by_category[cat]?.mean_pct ?? 0);
    const ntVals = nonThinkingRuns.map((r) => agg.by_run[r].by_category[cat]?.mean_pct ?? 0);
    catThinking[cat] = tVals.reduce((a, b) => a + b, 0) / tVals.length;
    catNonThinking[cat] = ntVals.reduce((a, b) => a + b, 0) / ntVals.length;
  }

  // Latency
  const thinkingLatencies = thinkingRuns.map((r) => agg.by_run[r].latency?.mean ?? 0);
  const nonThinkingLatencies = nonThinkingRuns.map((r) => agg.by_run[r].latency?.mean ?? 0);
  const avgThinkingLatency = thinkingLatencies.reduce((a, b) => a + b, 0) / thinkingLatencies.length;
  const avgNonThinkingLatency = nonThinkingLatencies.reduce((a, b) => a + b, 0) / nonThinkingLatencies.length;

  // Failure count
  const totalCaseRuns = runIds.length * d.meta.cases_total;
  let failures = 0;
  for (const run of Object.values(d.runs)) {
    for (const caseId of d.case_order) {
      const exec = run.execution[caseId];
      if (exec?.error) failures++;
    }
  }

  container.innerHTML = `
    <div class="overview">
      <div class="overview-hero">
        <h1>MetroLLM-Bench</h1>
        <p class="overview-subtitle">Can natural language replace programming for transit kiosk operations?</p>
        <p class="overview-desc">
          This benchmark tests whether an LLM can serve as the intelligence behind a transit kiosk &mdash;
          routing passengers, calculating fares, and adapting to disruptions &mdash; all driven by natural
          language prompts and tool calls instead of hardcoded logic.
        </p>
      </div>

      <div class="overview-setup">
        <div class="setup-item">
          <span class="setup-label">Model</span>
          <span class="setup-value">${d.meta.model}</span>
        </div>
        <div class="setup-item">
          <span class="setup-label">System</span>
          <span class="setup-value">${d.meta.system.toUpperCase()}</span>
        </div>
        <div class="setup-item">
          <span class="setup-label">Test cases</span>
          <span class="setup-value">${d.meta.cases_total}</span>
        </div>
        <div class="setup-item">
          <span class="setup-label">Runs</span>
          <span class="setup-value">${d.meta.runs_total} (${thinkingRuns.length} thinking, ${nonThinkingRuns.length} non-thinking)</span>
        </div>
        <div class="setup-item">
          <span class="setup-label">Failures</span>
          <span class="setup-value">${failures} / ${totalCaseRuns} case-runs</span>
        </div>
      </div>

      <h2>Headline results</h2>
      <div class="overview-cards">
        <div class="result-card clickable" data-link="#/comparison">
          <div class="card-number" style="color: ${runColor(true)}">${fmtPct(thinkingAvg)}</div>
          <div class="card-label">Thinking mode</div>
          <div class="card-detail">avg across ${thinkingRuns.length} runs</div>
        </div>
        <div class="result-card clickable" data-link="#/comparison">
          <div class="card-number" style="color: ${runColor(false)}">${fmtPct(nonThinkingAvg)}</div>
          <div class="card-label">Non-thinking mode</div>
          <div class="card-detail">avg across ${nonThinkingRuns.length} runs</div>
        </div>
        <div class="result-card clickable" data-link="#/comparison">
          <div class="card-number">${fmtPct(thinkingAvg - nonThinkingAvg)}</div>
          <div class="card-label">Thinking advantage</div>
          <div class="card-detail">overall gap</div>
        </div>
        <div class="result-card clickable" data-link="#/comparison">
          <div class="card-number">${fmtMs(avgThinkingLatency)} / ${fmtMs(avgNonThinkingLatency)}</div>
          <div class="card-label">Avg latency</div>
          <div class="card-detail">thinking / non-thinking</div>
        </div>
      </div>

      <h2>By category</h2>
      <p class="overview-desc">The benchmark tests three types of tasks. Each case is scored on multiple components (route correctness, fare accuracy, tool usage, schema validity, etc.) for a maximum of 65&ndash;85 points.</p>
      <div class="category-breakdown">
        ${cats
          .map(
            (cat) => `
          <div class="category-row clickable" data-cat="${cat}">
            <div class="cat-header">
              <span class="cat-badge cat-${cat}">Cat ${cat}</span>
              <span class="cat-name">${catLabels[cat]}</span>
              <span class="cat-count">${agg.by_run[runIds[0]].by_category[cat]?.count ?? 0} cases</span>
              <span class="cat-arrow">&rarr;</span>
            </div>
            <div class="cat-bars">
              <div class="cat-bar-group">
                <div class="cat-bar" style="width: ${catThinking[cat]}%; background: ${runColor(true)}"></div>
                <span class="cat-bar-label">${fmtPct(catThinking[cat])} thinking</span>
              </div>
              <div class="cat-bar-group">
                <div class="cat-bar" style="width: ${catNonThinking[cat]}%; background: ${runColor(false)}"></div>
                <span class="cat-bar-label">${fmtPct(catNonThinking[cat])} non-thinking</span>
              </div>
            </div>
            <p class="cat-desc">${catDescription(cat)}</p>
          </div>
        `
          )
          .join("")}
      </div>

      <h2>Key findings</h2>
      <ul class="findings-list">
        <li><strong>Thinking helps most on disruptions.</strong> Cat C (disruption handling) shows the biggest gap between modes: ${fmtPct(catThinking["C"])} vs ${fmtPct(catNonThinking["C"])}. When the model needs to reason about re-routing around closures or issuing hurricane warnings, the extra reasoning step pays off.</li>
        <li><strong>Routing is essentially solved.</strong> Cat A (basic routing) scores ${fmtPct(catThinking["A"])} / ${fmtPct(catNonThinking["A"])} &mdash; the LLM reliably calls route_planner and formats the result.</li>
        <li><strong>Fare calculation is near-perfect.</strong> Cat B achieves ${fmtPct(catThinking["B"])} / ${fmtPct(catNonThinking["B"])}, though edge cases with complex passenger mixes occasionally trip up the model.</li>
        <li><strong>Token cost of thinking.</strong> Thinking mode takes ~${Math.round((avgThinkingLatency / avgNonThinkingLatency - 1) * 100)}% longer (${fmtMs(avgThinkingLatency)} vs ${fmtMs(avgNonThinkingLatency)}) for a ${fmtPct(thinkingAvg - nonThinkingAvg)} accuracy gain.</li>
      </ul>

      <h2>Explore the data</h2>
      <div class="explore-grid"></div>
    </div>
  `;

  // Category rows → category detail page
  container.querySelectorAll<HTMLElement>(".category-row[data-cat]").forEach((row) => {
    row.addEventListener("click", () => {
      navigate(`/category/${row.dataset.cat}`);
    });
  });

  // Headline cards → comparison
  container.querySelectorAll<HTMLElement>(".result-card[data-link]").forEach((card) => {
    card.addEventListener("click", () => {
      navigate(card.dataset.link!);
    });
  });

  // Add explore cards with click navigation
  const exploreGrid = container.querySelector(".explore-grid")!;
  const views = [
    {
      path: "#/heatmap",
      title: "Scoring Heatmap",
      desc: "See every case and scoring component in a color-coded grid. Green = full marks, red = failure. Gray cells mean that component doesn't apply to that case category.",
    },
    {
      path: "#/comparison",
      title: "Run Comparison",
      desc: "Compare all 6 runs side-by-side with box plots, category breakdowns, latency distributions, and per-case score overlays.",
    },
    {
      path: "#/toolflow",
      title: "Tool Call Flow",
      desc: "Sankey diagram showing which tools the LLM called and in what order. See how Cat C cases diverge from the standard route/fare/submit pattern.",
    },
    {
      path: "#/replay",
      title: "Conversation Replay",
      desc: "Read the full LLM conversation for any case: system prompt, user message, thinking traces, tool calls with arguments, tool results, and final submission.",
    },
  ];

  for (const v of views) {
    const card = document.createElement("div");
    card.className = "explore-card";
    card.innerHTML = `<h3>${v.title}</h3><p>${v.desc}</p>`;
    card.addEventListener("click", () => navigate(v.path));
    exploreGrid.appendChild(card);
  }
}

function catDescription(cat: string): string {
  switch (cat) {
    case "A":
      return "Given origin + destination, plan a route using the station graph. Scored on transfers, distance, and line sequence correctness.";
    case "B":
      return "Given a route and passenger mix (adults, children, seniors, disabled), calculate the fare. Flat $2.50/adult, children under 5 free, seniors/disabled $1.25.";
    case "C":
      return "Route + fare with an active disruption (station closure, maintenance, hurricane). Must call disruption_feed, issue an advisory, and re-route if needed.";
    case "D":
      return "Route + fare for a wheelchair user. Must call station_info to check elevator status at each station and warn about out-of-service elevators.";
    default:
      return "";
  }
}
