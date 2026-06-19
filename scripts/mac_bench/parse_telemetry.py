#!/usr/bin/env python3
"""Combine llama-server log, RSS samples, and bench results into one telemetry JSON.

Output schema (mac_bench/<chip>-<ram>gb-<size>/telemetry.json):

{
  "hardware": {"chip": "M2-Max", "ram_gb": 96, "fanless": false},
  "model": {"size": "2b", "repo": "continker/Qwen3.5-2B-metro-v24", "gguf_gb": 1.27},
  "eval": {"tier1_composite": 84.0, "metrollm_composite": 81.5, ...},
  "perf": {
    "decode_tok_s_median": 41.2, "decode_tok_s_p10": 38.0, "decode_tok_s_p90": 44.5,
    "decode_tok_s_n": 421,
    "ttft_ms_median": 287, "ttft_ms_p90": 540,
    "peak_rss_gb": 1.6,
    "runner_wallclock_s": 4520
  }
}

Stdin/stdout: pure JSON dump on success. Errors go to stderr; exit code is 0
unless required inputs missing.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

# llama.cpp 'eval time' line shapes vary across versions. Cover the ones we'll see.
# Examples:
#   eval time =     234.56 ms /    50 tokens (    4.69 ms per token,   213.42 tokens per second)
#   eval time =     234.56 ms /    50 runs   (    4.69 ms per token,   213.42 tokens per second)
EVAL_RE = re.compile(
    r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)\s*"
    r"\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
    re.IGNORECASE,
)

# Some builds use 'predicted' instead of 'eval':
PRED_RE = re.compile(
    r"predicted\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)\s*"
    r"\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
    re.IGNORECASE,
)


def parse_decode_tok_s(log_path: Path) -> list[float]:
    """Parse only DECODE eval lines (skip 'prompt eval' which is ~10x faster
    and would skew the median upward). The decode line is `eval time = ...`
    without the 'prompt' prefix. We require at least 8 tokens evaluated to
    skip 1-2 token completion bursts."""
    if not log_path.exists():
        return []
    rates: list[float] = []
    with log_path.open() as f:
        for line in f:
            # CRITICAL: skip prompt-eval lines (regex would match them otherwise).
            if "prompt eval time" in line:
                continue
            for rx in (EVAL_RE, PRED_RE):
                m = rx.search(line)
                if m:
                    n_tokens = int(m.group(2))
                    tok_s = float(m.group(3))
                    if n_tokens >= 8:
                        rates.append(tok_s)
                    break
    return rates


def parse_peak_rss_gb(rss_log: Path) -> float:
    if not rss_log.exists():
        return 0.0
    peak_kb = 0
    with rss_log.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                peak_kb = max(peak_kb, int(parts[1]))
    return peak_kb / 1024 / 1024  # KB → GB


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def parse_runner_ttft(raw_path: Path) -> list[float]:
    """Pull TTFT (ms) from runner output's per-case latency. Different runner versions
    expose this differently; we tolerate missing fields."""
    if not raw_path.exists():
        return []
    try:
        data = json.loads(raw_path.read_text())
    except json.JSONDecodeError:
        return []
    cases = data.get("cases") or data.get("results") or []
    out: list[float] = []
    for c in cases:
        # try common field names
        for key in ("ttft_ms", "first_token_ms", "first_round_latency_ms"):
            v = c.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
                break
        else:
            # fallback: nested under 'latency' or 'timing'
            timing = c.get("latency") or c.get("timing") or {}
            v = timing.get("ttft_ms") or timing.get("first_token_ms")
            if isinstance(v, (int, float)):
                out.append(float(v))
    return out


def load_metrics(scored_path: Path) -> dict:
    """Pull tier1, composite, and n_cases from the scored output. Field
    locations differ slightly from what the runner produces — we read both
    `metrics.tier1_composite` (the leaderboard number) and
    `summary.cases_scored` (the n)."""
    if not scored_path.exists():
        return {}
    try:
        d = json.loads(scored_path.read_text())
    except json.JSONDecodeError:
        return {}
    metrics = d.get("metrics", {}) or {}
    summary = d.get("summary", {}) or {}
    scores = d.get("scores", []) or []
    n_cases = summary.get("cases_scored") or len(scores) or None
    tier1_pct_values = [s.get("tier1_pct") for s in scores if isinstance(s, dict) and s.get("tier1_pct") is not None]
    tier1_pct_mean = (sum(tier1_pct_values) / len(tier1_pct_values)) if tier1_pct_values else None
    return {
        "tier1_composite": metrics.get("tier1_composite"),
        "metrollm_composite": metrics.get("metrollm_composite"),
        "tier1_pct_mean": tier1_pct_mean,
        "n_cases": n_cases,
    }


def fanless_for_chip(chip: str) -> bool:
    # Apple silicon fanless skus: MacBook Air (M1/M2/M3 base/Pro variants don't ship fanless),
    # only the **base** Air chips (M1, M2, M3, M4 Air) are fanless.
    # Pro/Max/Ultra are all fan-cooled. Match conservatively.
    fanless_chips = {"M1", "M2", "M3", "M4"}
    base = chip.replace("-", " ").strip()
    return base in fanless_chips


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama-log", required=True, type=Path)
    p.add_argument("--rss-log", required=True, type=Path)
    p.add_argument("--raw-results", required=True, type=Path)
    p.add_argument("--scored-results", required=True, type=Path)
    p.add_argument("--runner-wallclock", required=True, type=int)
    p.add_argument("--chip", required=True)
    p.add_argument("--ram-gb", required=True, type=int)
    p.add_argument("--size", required=True)
    p.add_argument("--ctx-size", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    rates = parse_decode_tok_s(args.llama_log)
    ttfts = parse_runner_ttft(args.raw_results)
    peak_rss = parse_peak_rss_gb(args.rss_log)
    metrics = load_metrics(args.scored_results)

    gguf_path = Path("data/mac_models") / f"Qwen3.5-{args.size.upper()}-metro-v24-Q4_K_M.gguf"
    gguf_gb = gguf_path.stat().st_size / 1e9 if gguf_path.exists() else 0.0

    out = {
        "hardware": {
            "chip": args.chip,
            "ram_gb": args.ram_gb,
            "fanless": fanless_for_chip(args.chip),
        },
        "model": {
            "size": args.size,
            "repo": f"continker/Qwen3.5-{args.size.upper()}-metro-v24",
            "gguf_gb": round(gguf_gb, 3),
            "ctx_size": args.ctx_size,
        },
        "eval": {
            "tier1_composite": metrics.get("tier1_composite"),
            "metrollm_composite": metrics.get("metrollm_composite"),
            "tier1_pct_mean": metrics.get("tier1_pct_mean"),
            "n_cases": metrics.get("n_cases"),
        },
        "perf": {
            "decode_tok_s_median": statistics.median(rates) if rates else 0.0,
            "decode_tok_s_p10": percentile(rates, 10),
            "decode_tok_s_p90": percentile(rates, 90),
            "decode_tok_s_n": len(rates),
            "ttft_ms_median": statistics.median(ttfts) if ttfts else 0.0,
            "ttft_ms_p90": percentile(ttfts, 90),
            "ttft_ms_n": len(ttfts),
            "peak_rss_gb": round(peak_rss, 3),
            "runner_wallclock_s": args.runner_wallclock,
        },
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
