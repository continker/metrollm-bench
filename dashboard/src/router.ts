export interface Route {
  view: string;
  params: Record<string, string>;
}

export function parseHash(): Route {
  const hash = location.hash.slice(1) || "/overview";
  const parts = hash.split("/").filter(Boolean);
  const view = parts[0] || "overview";
  const params: Record<string, string> = {};

  if (view === "replay" || view === "toolflow") {
    if (parts[1]) params.runId = parts[1];
    if (parts[2]) params.caseId = parts[2];
  } else if (view === "heatmap") {
    if (parts[1]) params.runId = parts[1];
  } else if (view === "category") {
    if (parts[1]) params.cat = parts[1];
  }

  return { view, params };
}

export function navigate(path: string) {
  location.hash = path;
}

export function onRouteChange(cb: () => void) {
  window.addEventListener("hashchange", cb);
}
