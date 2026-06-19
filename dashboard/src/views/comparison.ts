import Plotly from "plotly.js-dist-min";
import type { DashboardData } from "../types";
import { runColor, CATEGORY_COLORS } from "../utils/colors";
import { fmtMs, fmtPct, fmtTokens, shortRunId } from "../utils/format";

const DARK_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#e0e0e0" },
  margin: { l: 60, r: 30, t: 50, b: 60 },
};

export function renderComparison(
  container: HTMLElement,
  data: DashboardData,
  _params: Record<string, string>,
): void {
  const runIds = Object.keys(data.runs);

  // -- Chart grid --
  const grid = document.createElement("div");
  grid.className = "chart-grid";
  container.appendChild(grid);

  const cells = Array.from({ length: 4 }, () => {
    const cell = document.createElement("div");
    cell.className = "chart-cell";
    grid.appendChild(cell);
    return cell;
  });

  const boxDiv = document.createElement("div");
  boxDiv.id = "comparison-box";
  cells[0].appendChild(boxDiv);

  const barDiv = document.createElement("div");
  barDiv.id = "comparison-bar";
  cells[1].appendChild(barDiv);

  const violinDiv = document.createElement("div");
  violinDiv.id = "comparison-violin";
  cells[2].appendChild(violinDiv);

  const scatterDiv = document.createElement("div");
  scatterDiv.id = "comparison-scatter";
  cells[3].appendChild(scatterDiv);

  // ---- Chart 1: Box plot of per-case score % ----
  const boxTraces: Partial<Plotly.PlotData>[] = runIds.map((rid) => {
    const run = data.runs[rid];
    const pcts = data.case_order.map((cid) => {
      const s = run.scores[cid];
      return s && s.max_possible > 0
        ? (s.total / s.max_possible) * 100
        : 0;
    });
    return {
      y: pcts,
      name: run.label,
      type: "box",
      marker: { color: runColor(run.thinking) },
      boxpoints: "outliers" as const,
    };
  });

  Plotly.newPlot(
    boxDiv,
    boxTraces,
    {
      ...DARK_LAYOUT,
      title: { text: "Score % Distribution", font: { color: "#e0e0e0" } },
      yaxis: { title: { text: "Score %" }, range: [0, 105] },
      showlegend: false,
    },
    { responsive: true, displayModeBar: false },
  );

  // ---- Chart 2: Grouped bar by category ----
  const categories = [...new Set(data.case_order.map((c) => data.cases[c].category))].sort();
  const barTraces: Partial<Plotly.PlotData>[] = runIds.map((rid) => {
    const run = data.runs[rid];
    const agg = data.aggregates.by_run[rid];
    const yVals = categories.map((cat) =>
      agg?.by_category[cat]?.mean_pct ?? 0,
    );
    return {
      x: categories,
      y: yVals,
      name: run.label,
      type: "bar",
      marker: { color: runColor(run.thinking) },
      text: yVals.map((v) => fmtPct(v)),
      textposition: "outside" as const,
      textfont: { size: 10 },
    };
  });

  Plotly.newPlot(
    barDiv,
    barTraces,
    {
      ...DARK_LAYOUT,
      title: { text: "Category Breakdown", font: { color: "#e0e0e0" } },
      barmode: "group",
      xaxis: { title: { text: "Category" } },
      yaxis: { title: { text: "Mean %" }, range: [0, 105] },
      legend: { orientation: "h", y: -0.2 },
    },
    { responsive: true, displayModeBar: false },
  );

  // ---- Chart 3: Violin plot of latency ----
  const violinTraces: Partial<Plotly.PlotData>[] = runIds.map((rid) => {
    const run = data.runs[rid];
    const latencies = data.case_order
      .map((cid) => run.execution[cid]?.e2e_ms)
      .filter((v): v is number => v != null);
    return {
      y: latencies,
      name: run.label,
      type: "violin",
      box: { visible: true },
      meanline: { visible: true },
      line: { color: runColor(run.thinking) },
      fillcolor: runColor(run.thinking) + "40",
    };
  });

  Plotly.newPlot(
    violinDiv,
    violinTraces,
    {
      ...DARK_LAYOUT,
      title: { text: "Latency Distribution", font: { color: "#e0e0e0" } },
      yaxis: { title: { text: "E2E (ms)" } },
      showlegend: false,
    },
    { responsive: true, displayModeBar: false },
  );

  // ---- Chart 4: Scatter per-case scores across runs ----
  // Category boundary lines
  const scatterShapes: Partial<Plotly.Shape>[] = [];
  for (let i = 1; i < data.case_order.length; i++) {
    const prevCat = data.cases[data.case_order[i - 1]].category;
    const curCat = data.cases[data.case_order[i]].category;
    if (prevCat !== curCat) {
      scatterShapes.push({
        type: "line",
        x0: i - 0.5,
        x1: i - 0.5,
        y0: 0,
        y1: 105,
        line: { color: "#666", width: 1, dash: "dash" },
        xref: "x",
        yref: "y",
      });
    }
  }

  // Category annotation positions (middle of each category span)
  const catAnnotations: Partial<Plotly.Annotations>[] = [];
  let spanStart = 0;
  for (let i = 1; i <= data.case_order.length; i++) {
    const atEnd = i === data.case_order.length;
    const boundary = atEnd || data.cases[data.case_order[i]]?.category !== data.cases[data.case_order[i - 1]].category;
    if (boundary) {
      const cat = data.cases[data.case_order[i - 1]].category;
      catAnnotations.push({
        x: (spanStart + i - 1) / 2,
        y: 103,
        text: `Cat ${cat}`,
        showarrow: false,
        font: { color: CATEGORY_COLORS[cat] || "#e0e0e0", size: 11 },
        xref: "x",
        yref: "y",
      });
      spanStart = i;
    }
  }

  const scatterTraces: Partial<Plotly.PlotData>[] = runIds.map((rid) => {
    const run = data.runs[rid];
    const xs = data.case_order.map((_, idx) => idx);
    const ys = data.case_order.map((cid) => {
      const s = run.scores[cid];
      return s && s.max_possible > 0
        ? (s.total / s.max_possible) * 100
        : 0;
    });
    const texts = data.case_order.map((cid) => cid);
    return {
      x: xs,
      y: ys,
      name: run.label,
      type: "scatter",
      mode: "markers",
      marker: {
        color: runColor(run.thinking),
        size: 6,
        opacity: 0.7,
      },
      text: texts,
      hovertemplate: "%{text}<br>%{y:.1f}%<extra></extra>",
    };
  });

  Plotly.newPlot(
    scatterDiv,
    scatterTraces,
    {
      ...DARK_LAYOUT,
      title: { text: "Per-Case Scores", font: { color: "#e0e0e0" } },
      xaxis: {
        title: { text: "Case index" },
        range: [-1, data.case_order.length],
      },
      yaxis: { title: { text: "Score %" }, range: [0, 108] },
      shapes: scatterShapes,
      annotations: catAnnotations,
      legend: { orientation: "h", y: -0.2 },
    },
    { responsive: true, displayModeBar: false },
  );

  // ---- Summary table ----
  const table = document.createElement("table");
  table.className = "summary-table";

  const thead = document.createElement("thead");
  thead.innerHTML = `<tr>
    <th>Run</th>
    <th>Thinking</th>
    <th>Overall %</th>
    ${categories.map((c) => `<th>Cat ${c}</th>`).join("")}
    <th>Latency (mean)</th>
    <th>Latency (p95)</th>
    <th>Tokens (mean)</th>
  </tr>`;
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const rid of runIds) {
    const run = data.runs[rid];
    const agg = data.aggregates.by_run[rid];
    const catCells = categories.map((cat) => {
      const pct = agg?.by_category[cat]?.mean_pct;
      return `<td>${pct != null ? fmtPct(pct) : "—"}</td>`;
    }).join("");

    const totalTokenMean = agg?.tokens
      ? ((agg.tokens.input_mean ?? 0) + (agg.tokens.output_mean ?? 0))
      : null;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${shortRunId(rid)}</td>
      <td>${run.thinking ? "Yes" : "No"}</td>
      <td>${agg ? fmtPct(agg.overall_pct) : "—"}</td>
      ${catCells}
      <td>${agg ? fmtMs(agg.latency.mean) : "—"}</td>
      <td>${agg ? fmtMs(agg.latency.p95) : "—"}</td>
      <td>${fmtTokens(totalTokenMean)}</td>
    `;
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);
}
