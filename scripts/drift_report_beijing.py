#!/usr/bin/env python3
"""Beijing drift analysis after framebook additions (v23).

Compares the current full 162-case Beijing run against expectations:
- overall tier1 / composite
- per-category distribution
- per-case outliers (anything below a reasonable floor)
- specific checks on the two cases driven by new framebook notes:
  BJM-A-021 (Line 10 circular direction), BJM-B-016 (Airport Express window)

Reports:
- Aggregate averages
- Bottom-10 cases by composite
- Any case where tier1 < 50% (red flags)
- Per-category counts + averages
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

scored_path = Path("results/beijing_drift_check/beijing_scored.json")
if not scored_path.exists():
    print(f"Missing {scored_path}")
    sys.exit(1)

data = json.loads(scored_path.read_text())
scores = data.get("scores", [])
if not scores:
    print("No scores found")
    sys.exit(1)

# Per-category aggregates
by_cat = defaultdict(list)
for s in scores:
    cat = s["case_id"].split("-")[-2] if "-" in s["case_id"] else "?"
    by_cat[cat].append(s)

print(f"=== Beijing drift check — {len(scores)} cases on gpt-5.4-mini ===\n")

# Overall
t1 = [s["tier1_pct"] for s in scores if s.get("tier1_pct") is not None]
comp = [s["pct"] for s in scores if s.get("pct") is not None]
print(f"Overall tier1:     {sum(t1)/len(t1):.1f}%  (n={len(t1)})")
print(f"Overall composite: {sum(comp)/len(comp):.1f}%  (n={len(comp)})")
print()

# Per-category
print(f"{'cat':<4} {'n':>4} {'tier1%':>8} {'comp%':>8}")
print("-" * 30)
for cat in sorted(by_cat):
    cs = by_cat[cat]
    ct1 = [s["tier1_pct"] for s in cs if s.get("tier1_pct") is not None]
    cc = [s["pct"] for s in cs if s.get("pct") is not None]
    if not ct1:
        continue
    print(f"{cat:<4} {len(cs):>4} {sum(ct1)/len(ct1):>7.1f}% {sum(cc)/len(cc):>7.1f}%")

# Red flags: tier1 < 50%
red = [s for s in scores if (s.get("tier1_pct") or 100) < 50]
print(f"\n=== Red flags (tier1 < 50%): {len(red)} ===")
for s in sorted(red, key=lambda x: x["tier1_pct"])[:10]:
    print(f"  {s['case_id']}: tier1 {s['tier1_pct']:.1f}%  composite {s['pct']:.1f}%")

# Bottom 10 by composite
print(f"\n=== Bottom 10 by composite ===")
for s in sorted(scores, key=lambda x: x["pct"])[:10]:
    print(f"  {s['case_id']}: comp {s['pct']:.1f}%  t1 {s['tier1_pct']:.1f}%")

# Specific v23 targets
print("\n=== v23 targets (framebook-driven) ===")
for cid in ("BJM-A-021", "BJM-B-016", "BJM-C-022"):
    s = next((x for x in scores if x["case_id"] == cid), None)
    if not s:
        print(f"  {cid}: not found")
        continue
    adv = (s["breakdown"].get("advisory_content_correct") or {}).get("score", "-")
    print(f"  {cid}: comp {s['pct']:.1f}%  t1 {s['tier1_pct']:.1f}%  advisory {adv}/10")

# Distribution histogram
print("\n=== Composite distribution ===")
buckets = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 85), (85, 90), (90, 95), (95, 101)]
for lo, hi in buckets:
    n = sum(1 for s in scores if lo <= (s.get("pct") or 0) < hi)
    bar = "█" * n
    print(f"  {lo:>3}-{hi-1 if hi<101 else 100:<3}: {n:>3}  {bar}")
