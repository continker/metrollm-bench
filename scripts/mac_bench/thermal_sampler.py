#!/usr/bin/env python3
"""Real-time tok/s + RSS sampler for run_thermal.sh.

Polls llama-server.log incrementally and the RSS log every `interval` seconds,
appending one row per interval to thermal_curve.csv:

    t_sec, tok_s_mean, tok_s_p10, tok_s_n, rss_gb

Exits cleanly after `duration` seconds. Writes a final summary to
thermal_curve.json with cold/sustained/throttle stats.
"""
from __future__ import annotations
import argparse
import json
import re
import statistics
import time
from pathlib import Path

EVAL_RE = re.compile(
    r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)\s*"
    r"\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
    re.IGNORECASE,
)


def latest_rss_gb(rss_log: Path) -> float:
    if not rss_log.exists():
        return 0.0
    try:
        with rss_log.open() as f:
            tail = f.readlines()[-3:]
        for line in reversed(tail):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama-log", type=Path, required=True)
    p.add_argument("--rss-log", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--interval", type=int, default=30, help="seconds per sample window")
    p.add_argument("--duration", type=int, default=2700, help="seconds to sample (default 45 min)")
    p.add_argument("--min-tokens", type=int, default=8,
                   help="filter eval lines with fewer tokens than this (skip trivial bursts)")
    args = p.parse_args()

    # Wait until llama-server log exists
    deadline = time.time() + 60
    while not args.llama_log.exists() and time.time() < deadline:
        time.sleep(1)
    if not args.llama_log.exists():
        print(f"llama log never appeared at {args.llama_log}", flush=True)
        return

    last_pos = 0
    with args.llama_log.open() as f:
        f.seek(0, 2)  # skip startup lines (model load, etc.)
        last_pos = f.tell()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv = args.out_csv.open("w", buffering=1)
    csv.write("t_sec,tok_s_mean,tok_s_median,tok_s_p10,tok_s_n,rss_gb\n")

    rows: list[dict] = []
    start = time.time()
    next_sample = start + args.interval

    while time.time() - start < args.duration:
        sleep_for = max(0.5, next_sample - time.time())
        time.sleep(sleep_for)

        # Read all new content since last poll
        try:
            with args.llama_log.open() as f:
                f.seek(last_pos)
                chunk = f.read()
                last_pos = f.tell()
        except FileNotFoundError:
            chunk = ""

        rates = []
        # Process line-by-line to filter out prompt-eval lines (which would
        # otherwise inflate decode tok/s by ~10x).
        for line in chunk.splitlines():
            if "prompt eval time" in line:
                continue
            m = EVAL_RE.search(line)
            if m:
                n_tok = int(m.group(1))
                tok_s = float(m.group(2))
                if n_tok >= args.min_tokens:
                    rates.append(tok_s)

        rss = latest_rss_gb(args.rss_log)
        t = round(time.time() - start, 1)
        if rates:
            mean = statistics.mean(rates)
            med = statistics.median(rates)
            p10 = percentile(rates, 10)
        else:
            mean = med = p10 = 0.0
        csv.write(f"{t:.1f},{mean:.2f},{med:.2f},{p10:.2f},{len(rates)},{rss:.3f}\n")
        rows.append({"t_sec": t, "tok_s_mean": mean, "tok_s_median": med,
                     "tok_s_p10": p10, "tok_s_n": len(rates), "rss_gb": rss})
        next_sample += args.interval

    csv.close()

    # Summary stats
    early = [r for r in rows if r["t_sec"] <= 60 and r["tok_s_n"] > 0]
    late = [r for r in rows[-min(len(rows), 5):] if r["tok_s_n"] > 0]
    all_rates = [r["tok_s_mean"] for r in rows if r["tok_s_n"] > 0]
    cold = max((r["tok_s_mean"] for r in rows[:3] if r["tok_s_n"] > 0), default=0.0)
    sustained = statistics.median([r["tok_s_mean"] for r in late]) if late else 0.0
    overall = statistics.median(all_rates) if all_rates else 0.0
    throttle_pct = (1 - sustained / cold) * 100 if cold > 0 else 0.0
    peak_rss = max((r["rss_gb"] for r in rows), default=0.0)

    summary = {
        "duration_sec": args.duration,
        "interval_sec": args.interval,
        "n_samples": len(rows),
        "tok_s_cold": round(cold, 2),
        "tok_s_sustained_last5": round(sustained, 2),
        "tok_s_median_overall": round(overall, 2),
        "throttle_pct_cold_to_sustained": round(throttle_pct, 1),
        "peak_rss_gb": round(peak_rss, 3),
        "samples": rows,
    }
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_csv} and {args.out_json}")
    print(f"  cold:      {cold:.1f} tok/s")
    print(f"  sustained: {sustained:.1f} tok/s (last 5 samples)")
    print(f"  throttle:  {throttle_pct:+.1f}%  (cold → sustained)")
    print(f"  peak rss:  {peak_rss:.2f} GB")


if __name__ == "__main__":
    main()
