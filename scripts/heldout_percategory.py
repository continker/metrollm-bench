#!/usr/bin/env python3
"""Per-category held-out scores for the rev3 Figure 3 heatmap.

Filters each model's per-system scored files to the 238 held-out case IDs
(data/splits/v23_holdout75_seed42.json), derives the category from the
case_id middle token (e.g. MARTA-A-001 -> A), and reports per-category
composite + tier-1 as the mean across the six systems. PEFT rows are the
mean across the two training seeds (per-seed per-category mean, then
averaged), matching how Table 1 reports the n=2 leaderboard rows.

Also prints the overall held-out tier-1/composite per model as a sanity
check against the Table 1 leaderboard.

Usage:
    uv run python scripts/heldout_percategory.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYSTEMS = ["marta", "bart", "cta", "doha", "taipei", "beijing"]
CATEGORIES = list("ABCDEFGHIJK")
SPLIT = REPO / "data/splits/v23_holdout75_seed42.json"

# Each model maps to one or more "seed groups". Each seed group is
# (result_dir, filename_pattern). PEFT models have two groups (seed 42, 43);
# everything else has one.
MODELS: list[tuple[str, list[tuple[Path, str]]]] = [
    ("Qwen 27B base",       [(REPO / "results/v23_scaling", "base_27b_{sys}_scored.json")]),
    ("GPT-5.4 full xhigh",  [(REPO / "results/v23_full_xhigh", "{sys}_scored.json")]),
    ("Qwen 35B-A3B base",   [(REPO / "results/v23_scaling", "base_35b_{sys}_scored.json")]),
    ("Qwen 27B+PEFT",       [(REPO / "results/v23_clean", "ft_v23_clean_27b_{sys}_scored.json"),
                             (REPO / "results/v23_clean", "ft_v23_clean_27b_s2_{sys}_scored.json")]),
    ("Qwen 4B+PEFT",        [(REPO / "results/v23_clean", "ft_v23_clean_4b_{sys}_scored.json"),
                             (REPO / "results/v23_clean", "ft_v23_clean_4b_s2_{sys}_scored.json")]),
    ("Qwen 9B+PEFT",        [(REPO / "results/v23_clean", "ft_v23_clean_9b_{sys}_scored.json"),
                             (REPO / "results/v23_clean", "ft_v23_clean_9b_s2_{sys}_scored.json")]),
]


def cat_of(case_id: str) -> str | None:
    parts = case_id.split("-")
    return parts[1] if len(parts) >= 3 else None


def seed_group_stats(result_dir: Path, pattern: str, holdout: set[str]):
    """Return (per_cat: {cat: (tier1, comp, n)}, overall: (tier1, comp, n))
    for a single seed group across all six systems."""
    cat_t1: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    cat_co: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    all_t1: list[float] = []
    all_co: list[float] = []
    for sys in SYSTEMS:
        path = result_dir / pattern.format(sys=sys)
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for s in data.get("scores", []):
            cid = s.get("case_id")
            if cid not in holdout:
                continue
            cat = cat_of(cid)
            if cat is None or cat not in cat_t1:
                continue
            t1 = s.get("tier1_pct", 0.0)
            co = s.get("pct", 0.0)
            cat_t1[cat].append(t1)
            cat_co[cat].append(co)
            all_t1.append(t1)
            all_co.append(co)
    per_cat = {}
    for c in CATEGORIES:
        if cat_t1[c]:
            per_cat[c] = (sum(cat_t1[c]) / len(cat_t1[c]),
                          sum(cat_co[c]) / len(cat_co[c]),
                          len(cat_t1[c]))
    overall = (sum(all_t1) / len(all_t1) if all_t1 else 0.0,
               sum(all_co) / len(all_co) if all_co else 0.0,
               len(all_t1))
    return per_cat, overall


def model_stats(groups: list[tuple[Path, str]], holdout: set[str]):
    """Average per-category and overall across seed groups."""
    group_results = [seed_group_stats(d, p, holdout) for d, p in groups]
    # per-category: average the seed-group means
    per_cat = {}
    for c in CATEGORIES:
        t1s = [pc[c][0] for pc, _ in group_results if c in pc]
        cos = [pc[c][1] for pc, _ in group_results if c in pc]
        ns = [pc[c][2] for pc, _ in group_results if c in pc]
        if t1s:
            per_cat[c] = (sum(t1s) / len(t1s), sum(cos) / len(cos), ns[0])
    o_t1 = sum(o[0] for _, o in group_results) / len(group_results)
    o_co = sum(o[1] for _, o in group_results) / len(group_results)
    o_n = group_results[0][1][2] if group_results else 0
    return per_cat, (o_t1, o_co, o_n)


def main() -> None:
    split = json.loads(SPLIT.read_text())
    holdout = set(split["holdout_ids"])
    print(f"held-out n = {len(holdout)}\n")

    # Overall sanity check
    print("=== OVERALL held-out (sanity vs Table 1) ===")
    print(f"{'Model':<22} {'n':>4} {'Tier1':>7} {'Comp':>7}")
    print("-" * 44)
    model_percat = {}
    for name, groups in MODELS:
        per_cat, (t1, co, n) = model_stats(groups, holdout)
        model_percat[name] = per_cat
        print(f"{name:<22} {n:>4} {t1:>7.2f} {co:>7.2f}")

    # Per-category COMPOSITE matrix (Figure 3 source)
    print("\n=== PER-CATEGORY COMPOSITE (rows=cat, cols=model) ===")
    header = f"{'Cat':<4}" + "".join(f"{name.split()[0]+name.split()[1][:3]:>10}"
                                     if len(name.split()) > 1 else f"{name:>10}"
                                     for name, _ in MODELS)
    names = [n for n, _ in MODELS]
    print(f"{'Cat':<4}" + "".join(f"{i+1:>10}" for i in range(len(names))))
    for i, n in enumerate(names):
        print(f"  col {i+1} = {n}")
    print("-" * (4 + 10 * len(names)))
    for c in CATEGORIES:
        row = f"{c:<4}"
        for name in names:
            v = model_percat[name].get(c)
            row += f"{v[1]:>10.1f}" if v else f"{'--':>10}"
        print(row)

    # Per-category TIER-1 matrix
    print("\n=== PER-CATEGORY TIER-1 (rows=cat, cols=model) ===")
    for c in CATEGORIES:
        row = f"{c:<4}"
        for name in names:
            v = model_percat[name].get(c)
            row += f"{v[0]:>10.1f}" if v else f"{'--':>10}"
        print(row)


if __name__ == "__main__":
    main()
