#!/usr/bin/env python3
"""Aggregate per-size telemetry files into one mac_<chip>-<ram>gb.json report.

Reads results/mac_bench/<chip>-<ram>gb-<size>/telemetry.json for every size
present, emits results/mac_bench/<chip>-<ram>gb.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SIZES = ["2b", "4b", "9b", "27b"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chip", required=True, help="e.g. M2-Max")
    p.add_argument("--ram-gb", required=True, type=int)
    p.add_argument("--out-dir", default="results/mac_bench")
    args = p.parse_args()

    base = Path(args.out_dir)
    prefix = f"{args.chip}-{args.ram_gb}gb"

    runs: list[dict] = []
    for size in SIZES:
        tel = base / f"{prefix}-{size}" / "telemetry.json"
        if tel.exists():
            runs.append(json.loads(tel.read_text()))

    if not runs:
        print(f"No telemetry files found for {prefix}", file=sys.stderr)
        sys.exit(1)

    hardware = runs[0]["hardware"]  # same across sizes on one Mac
    report = {
        "hardware": hardware,
        "runs": [{"model": r["model"], "eval": r["eval"], "perf": r["perf"]} for r in runs],
    }

    out_path = base / f"{prefix}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")

    # human-readable table
    print(f"\n{prefix}  ({hardware['chip']}, {hardware['ram_gb']} GB, fanless={hardware['fanless']})")
    print(f"{'size':<5} {'gguf':>6} {'tier1':>6} {'comp':>6} {'tok/s':>6} {'ttft':>5} {'rss':>5} {'time':>6}")
    print("-" * 56)
    for r in report["runs"]:
        m = r["model"]; e = r["eval"]; p = r["perf"]
        # `(x or 0)` because dict.get returns the explicit None when the key is
        # present-but-null (parse_telemetry.py emits null when scoring failed),
        # which would TypeError on `f"{None:>6.1f}"`.
        print(f"{m['size']:<5} {m['gguf_gb']:>6.2f} "
              f"{(e.get('tier1_composite') or 0):>6.1f} "
              f"{(e.get('metrollm_composite') or 0):>6.1f} "
              f"{p['decode_tok_s_median']:>6.1f} "
              f"{p['ttft_ms_median']:>5.0f} "
              f"{p['peak_rss_gb']:>5.2f} "
              f"{p['runner_wallclock_s']:>6}")


if __name__ == "__main__":
    main()
