import type { DashboardData } from "./types";

let cached: DashboardData | null = null;

export async function loadData(): Promise<DashboardData> {
  if (cached) return cached;
  const resp = await fetch("/data.json");
  if (!resp.ok) throw new Error(`Failed to load data.json: ${resp.status}`);
  cached = (await resp.json()) as DashboardData;
  return cached;
}
