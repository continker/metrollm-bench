#!/usr/bin/env python3
"""Compute single-model 95% bootstrap CIs and paired-bootstrap CIs on v23 scored data
for the research companion §9.1 + §9.2 tables.

Output schema:
    §9.1 single-model: per-model T1 mean + 95% CI, Composite mean + 95% CI (n=2000 resamples, n_cases=955)
    §9.2 paired:        per-pair ΔT1 mean + 95% CI, ΔComp mean + 95% CI, significance
                        flag (✓ if 0 outside CI, ◐ if borderline, ✗ if 0 in CI)

PEFT rows use the per-case mean across n=2 seeds (seed=42 + seed=43) at the score
level — i.e., for each case ID, the model's score is (s42_score + s43_score) / 2.
This matches paper §3 Table 1 PEFT rows which are reported as mean-of-n=2.

Usage:
    uv run python scripts/compute_bootstrap_v23.py [--seed 42] [--n-resamples 2000]

Output: prints two markdown tables and a summary of which cases were dropped (if any).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

SYSTEMS = ["marta", "bart", "cta", "doha", "taipei", "beijing"]

# (display_name, [(seed_label, scored_path_template), ...])
# scored_path_template is formatted with {sys}.
# When two paths are listed for a model, the per-case score is averaged across them.
MODELS: dict[str, list[tuple[str, str]]] = {
    "Qwen 27B base":            [("seed=na", "results/v23_scaling/base_27b_{sys}_scored.json")],
    "Qwen 35B-A3B base":        [("seed=na", "results/v23_scaling/base_35b_{sys}_scored.json")],
    "Qwen 9B base":             [("seed=na", "results/v23_scaling/base_9b_{sys}_scored.json")],
    "Qwen 4B base":             [("seed=na", "results/v23_scaling/base_4b_{sys}_scored.json")],
    "Qwen 9B + metro-v24 LoRA (n=2)":  [
        ("seed=42", "results/v23_peft/ft_v23_9b_{sys}_scored.json"),
        ("seed=43", "results/v23_peft/ft_v23_9b_s2_{sys}_scored.json"),
    ],
    "Qwen 4B + metro-v24 LoRA (n=2)":  [
        ("seed=42", "results/v23_peft/ft_v23_4b_{sys}_scored.json"),
        ("seed=43", "results/v23_peft/ft_v23_4b_s2_{sys}_scored.json"),
    ],
    "Qwen 27B + metro-v24 LoRA (n=2)": [
        ("seed=42", "results/v23_peft/ft_v23_27b_{sys}_scored.json"),
        ("seed=43", "results/v23_peft/ft_v23_27b_s2_{sys}_scored.json"),
    ],
    "GPT-5.4 full":             [("seed=na", "results/v23_full/{sys}_scored.json")],
    "Mistral Small 2603":       [("seed=na", "results/v23_mistral_small4/{sys}_scored.json")],
}

# Paired comparisons for §9.2 (paper-load-bearing only; smaller table avoids multiple-comparison concerns)
PAIRED: list[tuple[str, str]] = [
    ("Qwen 4B + metro-v24 LoRA (n=2)",  "GPT-5.4 full"),                         # practitioner headline (T1)
    ("Qwen 4B + metro-v24 LoRA (n=2)",  "Qwen 4B base"),                         # PEFT delta at 4B
    ("Qwen 27B + metro-v24 LoRA (n=2)", "Qwen 27B base"),                        # capacity-ceiling regression
    ("Qwen 9B + metro-v24 LoRA (n=2)",  "Qwen 35B-A3B base"),                    # 9B+PEFT ties 35B headline
]


def load_per_case(path_template: str) -> dict[str, tuple[float, float]]:
    """Returns {case_id: (tier1_pct, composite_pct)} from one model's scored runs across all 6 systems."""
    out: dict[str, tuple[float, float]] = {}
    for sys in SYSTEMS:
        path = Path(path_template.format(sys=sys))
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"  WARN: bad JSON {path}: {e}")
            continue
        for s in d.get("scores", []):
            cid = s.get("case_id")
            tier1_pct = s.get("tier1_pct")
            total = s.get("total")
            mx = s.get("max_possible")
            if cid is None or tier1_pct is None or total is None or mx in (None, 0):
                continue
            comp_pct = (total / mx) * 100.0
            out[cid] = (float(tier1_pct), float(comp_pct))
    return out


def merge_seeds(name: str, seeds: list[tuple[str, str]]) -> dict[str, tuple[float, float]]:
    """For multi-seed PEFT models, average per-case scores across seeds.
    Drops cases where any seed is missing (so n_cases is consistent across seeds)."""
    if len(seeds) == 1:
        per_case = load_per_case(seeds[0][1])
        return per_case
    per_seed = [load_per_case(p) for _, p in seeds]
    common = set(per_seed[0].keys())
    for ps in per_seed[1:]:
        common &= set(ps.keys())
    dropped = (len(per_seed[0]) - len(common)) + (len(per_seed[1]) - len(common))
    if dropped:
        print(f"  {name}: dropped {dropped} cases missing in one of n=2 seeds (n_used = {len(common)})")
    out: dict[str, tuple[float, float]] = {}
    for cid in common:
        t1s = [ps[cid][0] for ps in per_seed]
        cs = [ps[cid][1] for ps in per_seed]
        out[cid] = (sum(t1s) / len(t1s), sum(cs) / len(cs))
    return out


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(round((p / 100.0) * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def bootstrap_ci(values: list[float], n_resamples: int, rng: random.Random) -> tuple[float, float, float]:
    """Returns (mean, lo, hi) with 95% CI from n_resamples paired draws of len(values) with replacement."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return statistics.mean(values), percentile(means, 2.5), percentile(means, 97.5)


def paired_bootstrap_ci(a: list[float], b: list[float], n_resamples: int, rng: random.Random) -> tuple[float, float, float]:
    """Paired bootstrap on (a-b) — preserves case-level pairing across resamples."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = [ai - bi for ai, bi in zip(a, b)]
    means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return statistics.mean(diffs), percentile(means, 2.5), percentile(means, 97.5)


def sig_flag(lo: float, hi: float) -> str:
    if lo > 0 or hi < 0:
        return "✓"
    if lo > -0.5 or hi < 0.5:  # "borderline" marker if CI brushes zero closely
        return "◐"
    return "✗"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-resamples", type=int, default=2000)
    args = p.parse_args()

    rng = random.Random(args.seed)
    print(f"# v23 bootstrap CIs (n_resamples={args.n_resamples}, deterministic seed={args.seed})\n")

    # --- Load per-case data per model ---
    print("Loading per-case scores:")
    per_model: dict[str, dict[str, tuple[float, float]]] = {}
    for name, seeds in MODELS.items():
        print(f"  {name}")
        per_model[name] = merge_seeds(name, seeds)

    # --- §9.1 Single-model CIs ---
    headline = [
        "Qwen 27B base",
        "Qwen 35B-A3B base",
        "Qwen 9B + metro-v24 LoRA (n=2)",
        "Qwen 27B + metro-v24 LoRA (n=2)",
        "Qwen 4B + metro-v24 LoRA (n=2)",
        "GPT-5.4 full",
        "Mistral Small 2603",
    ]

    print(f"\n## §9.1 Single-model bootstrap CIs (2000 resamples)\n")
    print("| Model | n | T1 mean | T1 95% CI | Composite mean | Composite 95% CI |")
    print("|---|---:|---:|---|---:|---|")
    for name in headline:
        per_case = per_model.get(name, {})
        if not per_case:
            print(f"| {name} | 0 | -- | (no data) | -- | -- |")
            continue
        t1s = [v[0] for v in per_case.values()]
        cs = [v[1] for v in per_case.values()]
        t1_m, t1_lo, t1_hi = bootstrap_ci(t1s, args.n_resamples, rng)
        c_m, c_lo, c_hi = bootstrap_ci(cs, args.n_resamples, rng)
        print(f"| {name} | {len(per_case)} | {t1_m:.2f} | [{t1_lo:.2f}, {t1_hi:.2f}] | {c_m:.2f} | [{c_lo:.2f}, {c_hi:.2f}] |")

    # --- §9.2 Paired bootstrap ---
    print(f"\n## §9.2 Paired bootstrap on identical case set (2000 resamples)\n")
    print("Δ = A − B; ✓ = 95% CI excludes 0; ◐ = 95% CI within ±0.5 of 0; ✗ = CI brackets 0 widely.\n")
    print("| A | B | n | ΔT1 mean | ΔT1 95% CI | ΔComp mean | ΔComp 95% CI | Sig. T1 / Comp |")
    print("|---|---|---:|---:|---|---:|---|---|")
    for a_name, b_name in PAIRED:
        pa = per_model.get(a_name, {})
        pb = per_model.get(b_name, {})
        common = sorted(set(pa.keys()) & set(pb.keys()))
        if not common:
            print(f"| {a_name} | {b_name} | 0 | -- | (no data) | -- | -- | -- / -- |")
            continue
        a_t1 = [pa[c][0] for c in common]
        b_t1 = [pb[c][0] for c in common]
        a_c = [pa[c][1] for c in common]
        b_c = [pb[c][1] for c in common]
        dt1_m, dt1_lo, dt1_hi = paired_bootstrap_ci(a_t1, b_t1, args.n_resamples, rng)
        dc_m, dc_lo, dc_hi = paired_bootstrap_ci(a_c, b_c, args.n_resamples, rng)
        sig_t1 = sig_flag(dt1_lo, dt1_hi)
        sig_c = sig_flag(dc_lo, dc_hi)
        print(f"| {a_name} | {b_name} | {len(common)} | {dt1_m:+.2f} | [{dt1_lo:+.2f}, {dt1_hi:+.2f}] | {dc_m:+.2f} | [{dc_lo:+.2f}, {dc_hi:+.2f}] | {sig_t1} / {sig_c} |")


if __name__ == "__main__":
    main()
