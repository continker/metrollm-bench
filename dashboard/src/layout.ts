import { parseHash } from "./router";

const TABS = [
  { id: "overview", label: "Overview", path: "#/overview" },
  { id: "heatmap", label: "Heatmap", path: "#/heatmap" },
  { id: "comparison", label: "Comparison", path: "#/comparison" },
  { id: "toolflow", label: "Tool Flow", path: "#/toolflow" },
  { id: "replay", label: "Replay", path: "#/replay" },
];

export function renderLayout(root: HTMLElement): HTMLElement {
  root.innerHTML = "";

  const nav = document.createElement("nav");
  nav.className = "nav-bar";

  const title = document.createElement("a");
  title.className = "nav-title";
  title.textContent = "MetroLLM-Bench";
  title.href = "#/overview";
  title.style.textDecoration = "none";
  nav.appendChild(title);

  const tabContainer = document.createElement("div");
  tabContainer.className = "nav-tabs";
  for (const tab of TABS) {
    const a = document.createElement("a");
    a.href = tab.path;
    a.className = "nav-tab";
    a.dataset.view = tab.id;
    a.textContent = tab.label;
    tabContainer.appendChild(a);
  }
  nav.appendChild(tabContainer);
  root.appendChild(nav);

  const container = document.createElement("main");
  container.id = "view-container";
  root.appendChild(container);

  updateActiveTab();
  return container;
}

export function updateActiveTab() {
  const route = parseHash();
  document.querySelectorAll<HTMLAnchorElement>(".nav-tab").forEach((a) => {
    a.classList.toggle("active", a.dataset.view === route.view);
  });
}
