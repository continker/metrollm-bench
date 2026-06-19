#!/usr/bin/env python3
"""Paired + single-model bootstrap CIs on the HELD-OUT partition (n=238) for
the rev3 paper §4 capacity-ceiling claims and companion §9.2.

Differences from compute_bootstrap_v23.py (which is full-matrix, non-clean):
  - PEFT rows use the clean students (results/v23_clean/ft_v23_clean_*).
  - GPT-5.4 full is the xhigh-effort run (results/v23_full_xhigh).
  - Per-case scores are filtered to the 238 held-out case IDs from
    data/splits/v23_holdout75_seed42.json before any resampling.

PEFT rows average the per-case score across the two seeds (s42, s43), matching
Table 1. Composite per case = total / max_possible * 100 (points-weighted,
consistent with compute_metrics aggregation).

Usage:
    uv run python scripts/compute_bootstrap_heldout.py [--seed 42] [--n-resamples 2000]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYSTEMS = ["marta", "bart", "cta", "doha", "taipei", "beijing"]
SPLIT = REPO / "data/splits/v23_holdout75_seed42.json"

MODELS: dict[str, list[str]] = {
    "Qwen 27B base":     ["results/v23_scaling/base_27b_{sys}_scored.json"],
    "Qwen 35B-A3B base": ["results/v23_scaling/base_35b_{sys}_scored.json"],
    "Qwen 4B base":      ["results/v23_scaling/base_4b_{sys}_scored.json"],
    "Qwen 4B+PEFT (n=2)": [
        "results/v23_clean/ft_v23_clean_4b_{sys}_scored.json",
        "results/v23_clean/ft_v23_clean_4b_s2_{sys}_scored.json",
    ],
    "Qwen 9B+PEFT (n=2)": [
        "results/v23_clean/ft_v23_clean_9b_{sys}_scored.json",
        "results/v23_clean/ft_v23_clean_9b_s2_{sys}_scored.json",
    ],
    "Qwen 27B+PEFT (n=2)": [
        "results/v23_clean/ft_v23_clean_27b_{sys}_scored.json",
        "results/v23_clean/ft_v23_clean_27b_s2_{sys}_scored.json",
    ],
    "GPT-5.4 full xhigh": ["results/v23_full_xhigh/{sys}_scored.json"],
}

PAIRED: list[tuple[str, str]] = [
    ("Qwen 4B+PEFT (n=2)",  "GPT-5.4 full xhigh"),   # practitioner headline (T1)
    ("Qwen 4B+PEFT (n=2)",  "Qwen 4B base"),         # PEFT delta at 4B
    ("Qwen 27B+PEFT (n=2)", "Qwen 27B base"),        # capacity-ceiling regression
    ("Qwen 9B+PEFT (n=2)",  "Qwen 35B-A3B base"),    # 9B+PEFT vs 35B-A3B
]


def load_per_case(template: str, holdout: set[str]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for sys in SYSTEMS:
        path = REPO / template.format(sys=sys)
        if not path.exists():
            print(f"  WARN missing {path}")
            continue
        d = json.loads(path.read_text())
        for s in d.get("scores", []):
            cid = s.get("case_id")
            if cid not in holdout:
                continue
            t1 = s.get("tier1_pct")
            total = s.get("total")
            mx = s.get("max_possible")
            if cid is None or t1 is None or total is None or mx in (None, 0):
                continue
            out[cid] = (float(t1), total / mx * 100.0)
    return out


def merge_seeds(templates: list[str], holdout: set[str]) -> dict[str, tuple[float, float]]:
    if len(templates) == 1:
        return load_per_case(templates[0], holdout)
    per = [load_per_case(t, holdout) for t in templates]
    common = set(per[0])
    for p in per[1:]:
        common &= set(p)
    return {cid: (sum(p[cid][0] for p in per) / len(per),
                  sum(p[cid][1] for p in per) / len(per)) for cid in common}


def pctile(sv: list[float], p: float) -> float:
    if not sv:
        return 0.0
    return sv[max(0, min(len(sv) - 1, int(round(p / 100.0 * (len(sv) - 1)))))]


def boot(values: list[float], n: int, rng: random.Random) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    k = len(values)
    means = [sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n)]
    means.sort()
    return statistics.mean(values), pctile(means, 2.5), pctile(means, 97.5)


def paired_boot(a: list[float], b: list[float], n: int, rng: random.Random):
    k = len(a)
    if k == 0:
        return 0.0, 0.0, 0.0
    diffs = [ai - bi for ai, bi in zip(a, b)]
    means = [sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n)]
    means.sort()
    return statistics.mean(diffs), pctile(means, 2.5), pctile(means, 97.5)


def sig(lo: float, hi: float) -> str:
    if lo > 0 or hi < 0:
        return "excludes 0"
    return "includes 0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--partition", choices=["holdout", "train", "full"], default="holdout")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    split = json.loads(SPLIT.read_text())
    if args.partition == "holdout":
        holdout = set(split["holdout_ids"])
    elif args.partition == "train":
        holdout = set(split["train_ids"])
    else:
        holdout = set(split["holdout_ids"]) | set(split["train_ids"])
    print(f"# {args.partition} bootstrap (n_cases={len(holdout)}, resamples={args.n_resamples}, seed={args.seed})\n")

    per_model = {name: merge_seeds(t, holdout) for name, t in MODELS.items()}

    print("## single-model held-out CIs")
    print(f"{'Model':<22} {'n':>4} {'T1':>7} {'T1 95% CI':>18} {'Comp':>7} {'Comp 95% CI':>18}")
    for name in MODELS:
        pc = per_model[name]
        t1s = [v[0] for v in pc.values()]
        cs = [v[1] for v in pc.values()]
        m1, lo1, hi1 = boot(t1s, args.n_resamples, rng)
        mc, loc, hic = boot(cs, args.n_resamples, rng)
        print(f"{name:<22} {len(pc):>4} {m1:>7.2f} {f'[{lo1:.2f}, {hi1:.2f}]':>18} {mc:>7.2f} {f'[{loc:.2f}, {hic:.2f}]':>18}")

    print("\n## paired held-out CIs (Δ = A − B)")
    for a, b in PAIRED:
        pa, pb = per_model[a], per_model[b]
        common = sorted(set(pa) & set(pb))
        a1 = [pa[c][0] for c in common]; b1 = [pb[c][0] for c in common]
        ac = [pa[c][1] for c in common]; bc = [pb[c][1] for c in common]
        d1, l1, h1 = paired_boot(a1, b1, args.n_resamples, rng)
        dc, lc, hc = paired_boot(ac, bc, args.n_resamples, rng)
        print(f"\n  {a}  vs  {b}  (n={len(common)})")
        print(f"    ΔT1   {d1:+.2f}  CI [{l1:+.2f}, {h1:+.2f}]  {sig(l1, h1)}")
        print(f"    ΔComp {dc:+.2f}  CI [{lc:+.2f}, {hc:+.2f}]  {sig(lc, hc)}")


if __name__ == "__main__":
    main()
