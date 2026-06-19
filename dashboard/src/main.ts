import { loadData } from "./data";
import { renderLayout, updateActiveTab } from "./layout";
import { parseHash, onRouteChange } from "./router";
import { renderOverview } from "./views/overview";
import { renderCategory } from "./views/category";
import { renderHeatmap } from "./views/heatmap";
import { renderComparison } from "./views/comparison";
import { renderToolflow } from "./views/toolflow";
import { renderReplay } from "./views/replay";
import type { DashboardData } from "./types";

const views: Record<string, (el: HTMLElement, data: DashboardData, params: Record<string, string>) => void> = {
  overview: renderOverview,
  category: renderCategory,
  heatmap: renderHeatmap,
  comparison: renderComparison,
  toolflow: renderToolflow,
  replay: renderReplay,
};

async function render() {
  const data = await loadData();
  const container = document.getElementById("view-container")!;
  container.innerHTML = "";
  updateActiveTab();

  const route = parseHash();
  const viewFn = views[route.view];
  if (viewFn) {
    viewFn(container, data, route.params);
  } else {
    container.textContent = `Unknown view: ${route.view}`;
  }
}

async function init() {
  const app = document.getElementById("app")!;
  app.innerHTML = '<div class="loading">Loading data...</div>';

  try {
    await loadData();
    renderLayout(app);
    onRouteChange(render);
    render();
  } catch (err) {
    app.innerHTML = `<div class="error">Failed to load data: ${err}</div>`;
  }
}

init();
