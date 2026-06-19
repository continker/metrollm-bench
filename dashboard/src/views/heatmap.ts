import Plotly from "plotly.js-dist-min";
import type { DashboardData } from "../types";
import { navigate } from "../router";
import { SCORE_COLORSCALE } from "../utils/colors";

export function renderHeatmap(
  container: HTMLElement,
  data: DashboardData,
  params: Record<string, string>,
): void {
  const runIds = Object.keys(data.runs);
  const selectedRunId = params.runId && data.runs[params.runId]
    ? params.runId
    : runIds[0];

  // -- Run selector --
  const controls = document.createElement("div");
  controls.className = "view-controls";

  const label = document.createElement("label");
  label.textContent = "Run: ";
  label.style.marginRight = "0.5rem";

  const select = document.createElement("select");
  for (const rid of runIds) {
    const opt = document.createElement("option");
    opt.value = rid;
    opt.textContent = data.runs[rid].label;
    if (rid === selectedRunId) opt.selected = true;
    select.appendChild(opt);
  }
  select.addEventListener("change", () => {
    navigate(`/heatmap/${select.value}`);
  });

  label.appendChild(select);
  controls.appendChild(label);
  container.appendChild(controls);

  // -- Chart container --
  const chartDiv = document.createElement("div");
  chartDiv.id = "heatmap-chart";
  container.appendChild(chartDiv);

  const run = data.runs[selectedRunId];
  const caseIds = data.case_order;
  const components = data.scoring_components;
  const colLabels = [...components, "Total %"];

  // Build z-matrix (rows = cases, cols = components + total)
  const z: (number | null)[][] = [];
  const hoverTexts: string[][] = [];

  for (const caseId of caseIds) {
    const caseMeta = data.cases[caseId];
    const caseScore = run.scores[caseId];
    const row: (number | null)[] = [];
    const hoverRow: string[] = [];

    for (const comp of components) {
      if (!caseMeta.scoring_components.includes(comp)) {
        // Component not applicable to this case
        row.push(null);
        hoverRow.push(`${caseId}<br>${comp}<br>N/A`);
      } else if (caseScore?.breakdown[comp]) {
        const b = caseScore.breakdown[comp];
        const ratio = b.max > 0 ? b.score / b.max : 0;
        row.push(ratio);
        hoverRow.push(
          `${caseId}<br>${comp}<br>${b.score}/${b.max}<br>${b.reason}`,
        );
      } else {
        row.push(null);
        hoverRow.push(`${caseId}<br>${comp}<br>No data`);
      }
    }

    // Total % column
    if (caseScore) {
      const totalRatio = caseScore.max_possible > 0
        ? caseScore.total / caseScore.max_possible
        : 0;
      row.push(totalRatio);
      hoverRow.push(
        `${caseId}<br>Total<br>${caseScore.total}/${caseScore.max_possible}<br>${(totalRatio * 100).toFixed(1)}%`,
      );
    } else {
      row.push(null);
      hoverRow.push(`${caseId}<br>Total<br>No data`);
    }

    z.push(row);
    hoverTexts.push(hoverRow);
  }

  // Determine category boundaries for separator lines
  const shapes: Partial<Plotly.Shape>[] = [];
  for (let i = 1; i < caseIds.length; i++) {
    const prevCat = data.cases[caseIds[i - 1]].category;
    const curCat = data.cases[caseIds[i]].category;
    if (prevCat !== curCat) {
      shapes.push({
        type: "line",
        x0: -0.5,
        x1: colLabels.length - 0.5,
        y0: i - 0.5,
        y1: i - 0.5,
        line: { color: "#ffffff", width: 2 },
        xref: "x",
        yref: "y",
      });
    }
  }

  // Shorten component names for x-axis
  const shortNames = colLabels.map((name) =>
    name
      .replace("_correct", "")
      .replace("no_tool_", "no_")
      .replace("framebook_conformance", "framebook")
      .replace("schema_validity", "schema")
      .replace("disruption_detected", "disruption")
      .replace("reroute_quality", "reroute")
      .replace("suspension_compliance", "suspension"),
  );

  const trace: Partial<Plotly.PlotData> = {
    z,
    x: shortNames,
    y: caseIds,
    type: "heatmap",
    colorscale: SCORE_COLORSCALE as Plotly.ColorScale,
    zmin: 0,
    zmax: 1,
    hoverongaps: false,
    text: hoverTexts,
    hoverinfo: "text",
    colorbar: {
      title: { text: "Score ratio", side: "right" },
      tickformat: ".0%",
    },
  };

  const layout: Partial<Plotly.Layout> = {
    title: {
      text: `Score Heatmap - ${run.label}`,
      font: { color: "#e0e0e0" },
    },
    xaxis: {
      tickangle: -45,
      tickfont: { size: 11, color: "#e0e0e0" },
      side: "bottom",
    },
    yaxis: {
      autorange: "reversed" as const,
      tickfont: { size: 10, color: "#e0e0e0" },
      dtick: 1,
    },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#e0e0e0" },
    height: Math.max(600, caseIds.length * 18 + 150),
    margin: { l: 100, r: 80, t: 50, b: 100 },
    shapes,
  };

  const config: Partial<Plotly.Config> = {
    responsive: true,
    displayModeBar: false,
  };

  Plotly.newPlot(chartDiv, [trace], layout, config).then((gd) => {
    gd.on("plotly_click", (eventData: Plotly.PlotMouseEvent) => {
      if (eventData.points.length > 0) {
        const pt = eventData.points[0];
        const idx = Array.isArray(pt.pointIndex) ? pt.pointIndex[0] : pt.pointIndex;
        const caseId = caseIds[idx];
        navigate(`/replay/${selectedRunId}/${caseId}`);
      }
    });
  });
}
