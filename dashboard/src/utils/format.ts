export function fmtMs(ms: number | null): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function fmtPct(pct: number): string {
  return `${pct.toFixed(1)}%`;
}

export function fmtTokens(n: number | null): string {
  if (n == null) return "—";
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return `${n}`;
}

export function prettyJson(obj: unknown): string {
  return JSON.stringify(obj, null, 2);
}

export function shortRunId(runId: string): string {
  return runId.replace("v3_", "").replace(/_/g, " ");
}
