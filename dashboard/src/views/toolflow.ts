import Plotly from "plotly.js-dist-min";
import type { DashboardData } from "../types";
import { navigate } from "../router";
import { fmtMs } from "../utils/format";

export function renderToolflow(
  container: HTMLElement,
  data: DashboardData,
  params: Record<string, string>,
): void {
  const runIds = Object.keys(data.runs);
  const selectedRunId =
    params.runId && data.runs[params.runId] ? params.runId : runIds[0];
  const caseIds = data.case_order;
  const selectedCaseId =
    params.caseId && data.cases[params.caseId] ? params.caseId : caseIds[0];

  let activeTab: "sankey" | "timeline" = "sankey";

  // -- Tab toggle --
  const tabToggle = document.createElement("div");
  tabToggle.className = "tab-toggle";

  const sankeyBtn = document.createElement("button");
  sankeyBtn.className = "tab-btn active";
  sankeyBtn.textContent = "Aggregate Sankey";

  const timelineBtn = document.createElement("button");
  timelineBtn.className = "tab-btn";
  timelineBtn.textContent = "Per-Case Timeline";

  tabToggle.appendChild(sankeyBtn);
  tabToggle.appendChild(timelineBtn);
  container.appendChild(tabToggle);

  // -- Controls row --
  const controls = document.createElement("div");
  controls.className = "view-controls";

  const runLabel = document.createElement("label");
  runLabel.textContent = "Run: ";

  const runSelect = document.createElement("select");
  for (const rid of runIds) {
    const opt = document.createElement("option");
    opt.value = rid;
    opt.textContent = data.runs[rid].label;
    if (rid === selectedRunId) opt.selected = true;
    runSelect.appendChild(opt);
  }

  runLabel.appendChild(runSelect);
  controls.appendChild(runLabel);

  const caseLabel = document.createElement("label");
  caseLabel.textContent = "Case: ";

  const caseSelect = document.createElement("select");
  for (const cid of caseIds) {
    const opt = document.createElement("option");
    opt.value = cid;
    opt.textContent = `${cid} (${data.cases[cid].category})`;
    if (cid === selectedCaseId) opt.selected = true;
    caseSelect.appendChild(opt);
  }

  caseLabel.appendChild(caseSelect);
  controls.appendChild(caseLabel);

  container.appendChild(controls);

  // -- Content area --
  const content = document.createElement("div");
  container.appendChild(content);

  // Navigation on change
  runSelect.addEventListener("change", () => {
    navigate(`/toolflow/${runSelect.value}/${caseSelect.value}`);
  });
  caseSelect.addEventListener("change", () => {
    navigate(`/toolflow/${runSelect.value}/${caseSelect.value}`);
  });

  // Tab switching
  function setTab(tab: "sankey" | "timeline") {
    activeTab = tab;
    sankeyBtn.classList.toggle("active", tab === "sankey");
    timelineBtn.classList.toggle("active", tab === "timeline");
    caseLabel.style.display = tab === "timeline" ? "" : "none";
    renderContent();
  }

  sankeyBtn.addEventListener("click", () => setTab("sankey"));
  timelineBtn.addEventListener("click", () => setTab("timeline"));

  function renderContent() {
    content.innerHTML = "";
    if (activeTab === "sankey") {
      renderSankey(content, data, runSelect.value);
    } else {
      renderTimeline(content, data, runSelect.value, caseSelect.value);
    }
  }

  setTab("sankey");
}

// ---------- Aggregate Sankey ----------

function renderSankey(
  container: HTMLElement,
  data: DashboardData,
  runId: string,
): void {
  const patterns = data.aggregates.tool_patterns[runId];
  if (!patterns || Object.keys(patterns).length === 0) {
    container.textContent = "No tool pattern data available for this run.";
    return;
  }

  // Build position-prefixed nodes and links from pattern strings.
  // Pattern string example: "route_planner -> fare_calculator -> submit_response"
  const nodeIndex = new Map<string, number>(); // "position:tool" -> index
  const nodeLabels: string[] = [];
  const linkSource: number[] = [];
  const linkTarget: number[] = [];
  const linkValue: number[] = [];
  const linkLabels: string[] = [];

  // Accumulate link weights: key = "srcIdx->tgtIdx"
  const linkMap = new Map<string, number>();

  for (const [patternStr, count] of Object.entries(patterns)) {
    // Split on the arrow separator (handle both " -> " and " → ")
    const steps = patternStr
      .split(/\s*(?:->|→)\s*/)
      .map((s) => s.trim())
      .filter(Boolean);

    for (let i = 0; i < steps.length; i++) {
      const key = `${i}:${steps[i]}`;
      if (!nodeIndex.has(key)) {
        nodeIndex.set(key, nodeLabels.length);
        nodeLabels.push(`${i + 1}: ${steps[i]}`);
      }
    }

    for (let i = 0; i < steps.length - 1; i++) {
      const srcKey = `${i}:${steps[i]}`;
      const tgtKey = `${i + 1}:${steps[i + 1]}`;
      const srcIdx = nodeIndex.get(srcKey)!;
      const tgtIdx = nodeIndex.get(tgtKey)!;
      const edgeKey = `${srcIdx}->${tgtIdx}`;
      linkMap.set(edgeKey, (linkMap.get(edgeKey) || 0) + count);
    }
  }

  for (const [edgeKey, value] of linkMap) {
    const [src, tgt] = edgeKey.split("->").map(Number);
    linkSource.push(src);
    linkTarget.push(tgt);
    linkValue.push(value);
    linkLabels.push(`${value} case${value !== 1 ? "s" : ""}`);
  }

  // Assign colors per position depth
  const positionColors = [
    "#42a5f5", "#66bb6a", "#ffa726", "#ef5350", "#ab47bc",
    "#26c6da", "#d4e157", "#ec407a",
  ];
  const nodeColors = nodeLabels.map((label) => {
    const pos = parseInt(label.split(":")[0], 10) - 1;
    return positionColors[pos % positionColors.length];
  });

  const linkColors = linkSource.map((srcIdx) => {
    const color = nodeColors[srcIdx];
    // Make link colors semi-transparent
    return color + "66";
  });

  const chartDiv = document.createElement("div");
  chartDiv.id = "sankey-chart";
  container.appendChild(chartDiv);

  const trace: Partial<Plotly.PlotData> = {
    type: "sankey" as Plotly.PlotType,
    orientation: "h",
    node: {
      pad: 20,
      thickness: 20,
      label: nodeLabels,
      color: nodeColors,
      line: { color: "#333", width: 1 },
    },
    link: {
      source: linkSource,
      target: linkTarget,
      value: linkValue,
      label: linkLabels,
      color: linkColors,
    },
  };

  const layout: Partial<Plotly.Layout> = {
    title: {
      text: `Tool Call Flow Patterns - ${data.runs[runId].label}`,
      font: { color: "#e0e0e0" },
    },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#e0e0e0", size: 12 },
    height: 500,
    margin: { l: 20, r: 20, t: 50, b: 20 },
  };

  const config: Partial<Plotly.Config> = {
    responsive: true,
    displayModeBar: false,
  };

  Plotly.newPlot(chartDiv, [trace], layout, config);

  // Pattern summary table below the Sankey
  const tableWrap = document.createElement("div");
  tableWrap.style.marginTop = "1.5rem";

  const table = document.createElement("table");
  table.className = "summary-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const hdr of ["Pattern", "Count", "% of Cases"]) {
    const th = document.createElement("th");
    th.textContent = hdr;
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const totalCases = Object.values(patterns).reduce((a, b) => a + b, 0);
  const sorted = Object.entries(patterns).sort((a, b) => b[1] - a[1]);

  const tbody = document.createElement("tbody");
  for (const [pattern, count] of sorted) {
    const tr = document.createElement("tr");

    const tdPattern = document.createElement("td");
    tdPattern.style.fontFamily = '"SF Mono", "Fira Code", monospace';
    tdPattern.style.fontSize = "0.8rem";
    tdPattern.textContent = pattern;
    tr.appendChild(tdPattern);

    const tdCount = document.createElement("td");
    tdCount.textContent = String(count);
    tr.appendChild(tdCount);

    const tdPct = document.createElement("td");
    tdPct.textContent = `${((count / totalCases) * 100).toFixed(1)}%`;
    tr.appendChild(tdPct);

    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  tableWrap.appendChild(table);
  container.appendChild(tableWrap);
}

// ---------- Per-Case Timeline ----------

function renderTimeline(
  container: HTMLElement,
  data: DashboardData,
  runId: string,
  caseId: string,
): void {
  const run = data.runs[runId];
  if (!run) {
    container.textContent = "Run not found.";
    return;
  }

  const exec = run.execution[caseId];
  if (!exec) {
    container.textContent = "No execution data for this case.";
    return;
  }

  const toolCalls = exec.tool_calls;
  if (!toolCalls || toolCalls.length === 0) {
    container.textContent = "No tool calls recorded for this case.";
    return;
  }

  // Timeline visualization
  const timeline = document.createElement("div");
  timeline.className = "tool-timeline";

  for (let i = 0; i < toolCalls.length; i++) {
    const tc = toolCalls[i];

    // Arrow separator between nodes
    if (i > 0) {
      const arrow = document.createElement("span");
      arrow.className = "tool-arrow";
      arrow.textContent = "\u2192";
      timeline.appendChild(arrow);
    }

    const node = document.createElement("div");
    node.className = "tool-node";

    const nameSpan = document.createElement("span");
    nameSpan.textContent = tc.name;
    node.appendChild(nameSpan);

    const argsDiv = document.createElement("div");
    argsDiv.className = "tool-args";
    argsDiv.textContent = JSON.stringify(tc.arguments, null, 2);
    node.appendChild(argsDiv);

    node.addEventListener("click", () => {
      node.classList.toggle("expanded");
    });

    timeline.appendChild(node);
  }

  container.appendChild(timeline);

  // Execution stats
  const stats = document.createElement("div");
  stats.className = "chart-cell";
  stats.style.marginTop = "1rem";

  const statsTitle = document.createElement("h3");
  statsTitle.textContent = "Execution Stats";
  stats.appendChild(statsTitle);

  const statEntries: [string, string][] = [
    ["End-to-end", fmtMs(exec.e2e_ms)],
    ["Time to first token", fmtMs(exec.ttft_ms)],
    ["Input tokens", exec.input_tokens != null ? String(exec.input_tokens) : "\u2014"],
    ["Output tokens", exec.output_tokens != null ? String(exec.output_tokens) : "\u2014"],
    ["Tool calls", String(toolCalls.length)],
  ];

  if (exec.error) {
    statEntries.push(["Error", exec.error]);
  }

  const dl = document.createElement("dl");
  dl.style.display = "grid";
  dl.style.gridTemplateColumns = "auto 1fr";
  dl.style.gap = "0.25rem 1rem";
  dl.style.fontSize = "0.85rem";

  for (const [label, value] of statEntries) {
    const dt = document.createElement("dt");
    dt.style.color = "var(--text-muted)";
    dt.textContent = label;
    dl.appendChild(dt);

    const dd = document.createElement("dd");
    dd.textContent = value;
    dl.appendChild(dd);
  }

  stats.appendChild(dl);
  container.appendChild(stats);
}
