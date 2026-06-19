export const THINKING_COLOR = "#1976d2";
export const NON_THINKING_COLOR = "#e65100";
export const CATEGORY_COLORS: Record<string, string> = {
  A: "#4caf50",
  B: "#2196f3",
  C: "#ff9800",
  D: "#7b1fa2",
};

export function runColor(thinking: boolean): string {
  return thinking ? THINKING_COLOR : NON_THINKING_COLOR;
}

export function scoreColor(ratio: number): string {
  if (ratio >= 0.95) return "#1b5e20";
  if (ratio >= 0.8) return "#388e3c";
  if (ratio >= 0.6) return "#f9a825";
  if (ratio >= 0.4) return "#e65100";
  return "#b71c1c";
}

/** Plotly-compatible colorscale for heatmap (0=red, 1=green) */
export const SCORE_COLORSCALE: [number, string][] = [
  [0, "#b71c1c"],
  [0.4, "#e65100"],
  [0.6, "#f9a825"],
  [0.8, "#388e3c"],
  [1, "#1b5e20"],
];
