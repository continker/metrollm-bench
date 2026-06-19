export interface DashboardData {
  meta: Meta;
  cases: Record<string, CaseMeta>;
  case_order: string[];
  scoring_components: string[];
  runs: Record<string, Run>;
  aggregates: Aggregates;
}

export interface Meta {
  generated_at: string;
  model: string;
  system: string;
  cases_total: number;
  runs_total: number;
}

export interface CaseMeta {
  id: string;
  category: string;
  difficulty: string;
  title: string;
  max_possible: number;
  scoring_components: string[];
}

export interface Run {
  run_id: string;
  thinking: boolean;
  label: string;
  summary: RunSummary;
  scores: Record<string, CaseScore>;
  execution: Record<string, CaseExecution>;
  conversations: Record<string, CaseConversation>;
}

export interface RunSummary {
  cases_scored: number;
  total_points: number;
  max_points: number;
  average_score: number;
  average_pct: number;
  success_rate_pct: number;
}

export interface CaseScore {
  total: number;
  max_possible: number;
  breakdown: Record<string, ComponentScore>;
}

export interface ComponentScore {
  score: number;
  max: number;
  reason: string;
}

export interface CaseExecution {
  e2e_ms: number | null;
  ttft_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  error: string | null;
  tool_calls: ToolCall[];
}

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface CaseConversation {
  reasoning_content: string | null;
  messages: Message[];
}

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  reasoning_content?: string;
  tool_calls?: MessageToolCall[];
  tool_call_id?: string;
}

export interface MessageToolCall {
  type: string;
  function: { name: string; arguments: string };
  id: string;
}

export interface Aggregates {
  by_run: Record<string, RunAggregate>;
  thinking_vs_non_thinking: Record<string, { mean_pct: number; std_pct: number }>;
  tool_patterns: Record<string, Record<string, number>>;
}

export interface RunAggregate {
  overall_pct: number;
  by_category: Record<string, CategoryAggregate>;
  by_component: Record<string, { mean_pct: number; count: number }>;
  latency: LatencyStats;
  tokens: TokenStats;
}

export interface CategoryAggregate {
  mean_pct: number;
  count: number;
  total: number;
  max: number;
}

export interface LatencyStats {
  mean: number;
  median: number;
  p95: number;
  min: number;
  max: number;
}

export interface TokenStats {
  input_mean?: number;
  input_total?: number;
  output_mean?: number;
  output_total?: number;
}
