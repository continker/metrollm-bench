"""Build a deterministic 75/25 train/holdout split of the v23 benchmark.

Stratifies by system (not by system × category — cells like Cat K at 5
cases/system are too small for stable cell-level stratification). Pins the
15 gap-audit cases into held-out before the random pass.

Output: a single JSON file listing the train and holdout case IDs plus
per-system, per-category, and per-cell summary statistics, and a
template-overlap report quantifying how many training cases share an
(origin, destination) pair with each held-out case.

Usage:
    uv run python scripts/build_holdout_split.py
    uv run python scripts/build_holdout_split.py --seed 42 --frac 0.25
    uv run python scripts/build_holdout_split.py --dry-run

Invariants are checked before write. On any hard-fail, the file is not
written and the script exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = REPO_ROOT / "cases"
DEFAULT_OUTPUT = REPO_ROOT / "data/splits/v23_holdout75_seed42.json"

SYSTEMS = ["marta", "doha", "bart", "taipei", "cta", "beijing"]

# Pinned to held-out. These 15 cases were authored post-PEFT-training as a
# gap-audit; preserving them in held-out gives the partition a named
# strictly-OOD sub-cluster.
PINNED_GAP_AUDIT = {
    "BART-B-016", "BART-F-016",
    "BJM-A-021", "BJM-B-016", "BJM-C-022",
    "CTA-C-018", "CTA-F-016",
    "DOHA-C-016", "DOHA-E-006", "DOHA-E-007",
    "MARTA-D-016", "MARTA-F-016", "MARTA-F-017",
    "TRTC-B-016", "TRTC-C-018",
}


def load_cases() -> list[dict]:
    """Read all 6 case files into a single flat list with case_id, system,
    category, and the (origin_id, dest_id) station-pair for the overlap
    report. Order is system → case-file-order."""
    out: list[dict] = []
    for system in SYSTEMS:
        path = CASES_DIR / f"{system}_cases.json"
        if not path.exists():
            sys.exit(f"FATAL: missing case file {path}")
        cases = json.loads(path.read_text())
        for c in cases:
            ev = c.get("events") or []
            origin = next(
                (e.get("station_id") for e in ev
                 if e.get("type") == "station_selected" and e.get("field") == "origin"),
                None,
            )
            dest = next(
                (e.get("station_id") for e in ev
                 if e.get("type") == "station_selected" and e.get("field") == "destination"),
                None,
            )
            out.append({
                "case_id": c["id"],
                "system": c["system"],
                "category": c["category"],
                "od": (origin, dest) if origin and dest else None,
            })
    return out


def partition(cases: list[dict], seed: int, frac_holdout: float) -> tuple[set[str], set[str]]:
    """Pin gap-audit IDs to held-out, then system-stratified random split of
    the remainder. Per-system target is round(frac × system_total) minus
    pinned-in-system. Returns (train_ids, holdout_ids)."""
    rng = random.Random(seed)

    # Verify all pinned IDs exist in the loaded set (rev1 case-set drift
    # would surface as a missing pin and should fail loud, not silently).
    all_ids = {c["case_id"] for c in cases}
    missing_pins = PINNED_GAP_AUDIT - all_ids
    if missing_pins:
        sys.exit(
            f"FATAL: {len(missing_pins)} pinned gap-audit case IDs "
            f"not present in current case files: {sorted(missing_pins)}\n"
            f"Case set has drifted since rev1; investigate before splitting."
        )

    holdout: set[str] = set(PINNED_GAP_AUDIT)
    train: set[str] = set()

    for system in SYSTEMS:
        sys_cases = [c for c in cases if c["system"] == system]
        sys_total = len(sys_cases)
        sys_pinned = [c for c in sys_cases if c["case_id"] in PINNED_GAP_AUDIT]
        sys_unpinned = [c for c in sys_cases if c["case_id"] not in PINNED_GAP_AUDIT]

        target_holdout = round(frac_holdout * sys_total)
        random_holdout_n = max(0, target_holdout - len(sys_pinned))

        # Sort unpinned by case_id first (deterministic input order),
        # then shuffle with the seeded RNG, then take the first N.
        sys_unpinned.sort(key=lambda c: c["case_id"])
        rng.shuffle(sys_unpinned)
        for c in sys_unpinned[:random_holdout_n]:
            holdout.add(c["case_id"])
        for c in sys_unpinned[random_holdout_n:]:
            train.add(c["case_id"])

    return train, holdout


def check_invariants(
    cases: list[dict],
    train: set[str],
    holdout: set[str],
    frac_holdout: float,
) -> list[str]:
    """Return list of hard-fail messages. Empty list means all invariants
    passed."""
    fails: list[str] = []

    all_ids = {c["case_id"] for c in cases}
    n_all = len(all_ids)
    n_train = len(train)
    n_holdout = len(holdout)

    if n_all != 955:
        fails.append(f"total cases != 955 (got {n_all})")
    if n_train + n_holdout != n_all:
        fails.append(f"train + holdout != all ({n_train} + {n_holdout} = {n_train + n_holdout}, expected {n_all})")
    if train & holdout:
        fails.append(f"train ∩ holdout is non-empty ({len(train & holdout)} overlap)")
    if (train | holdout) != all_ids:
        fails.append(f"train ∪ holdout != all (missing {len(all_ids - (train | holdout))} ids)")

    actual_frac = n_holdout / n_all if n_all else 0
    if not (0.24 <= actual_frac <= 0.27):
        fails.append(f"holdout fraction {actual_frac:.3f} outside [0.24, 0.27]")

    missing_pins = PINNED_GAP_AUDIT - holdout
    if missing_pins:
        fails.append(f"{len(missing_pins)} pinned gap-audit IDs not in holdout: {sorted(missing_pins)}")

    for system in SYSTEMS:
        sys_cases = [c["case_id"] for c in cases if c["system"] == system]
        sys_total = len(sys_cases)
        sys_holdout = sum(1 for cid in sys_cases if cid in holdout)
        sys_frac = sys_holdout / sys_total if sys_total else 0
        if not (0.22 <= sys_frac <= 0.28):
            fails.append(
                f"per-system holdout fraction for {system} = {sys_frac:.3f} "
                f"(holdout {sys_holdout} of {sys_total}); outside [0.22, 0.28]"
            )

    return fails


def soft_warnings(cases: list[dict], holdout: set[str]) -> list[str]:
    """Per-cell holdout-fraction checks. Cells with <5 cases are listed
    for inspection but don't block. Cells with ≥5 cases are flagged if
    holdout fraction is outside [0.10, 0.50]."""
    warnings: list[str] = []
    by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in cases:
        by_cell[(c["system"], c["category"])].append(c["case_id"])

    for (system, cat), cell_ids in sorted(by_cell.items()):
        n = len(cell_ids)
        n_hold = sum(1 for cid in cell_ids if cid in holdout)
        frac = n_hold / n if n else 0
        if n < 5:
            warnings.append(
                f"  small cell {system}/{cat}: n={n}, holdout={n_hold} "
                f"(fraction={frac:.2f}, not strictly enforced)"
            )
        elif not (0.10 <= frac <= 0.50):
            warnings.append(
                f"  cell {system}/{cat} holdout fraction {frac:.2f} "
                f"outside [0.10, 0.50] (n={n}, holdout={n_hold})"
            )
    return warnings


def template_overlap(cases: list[dict], train: set[str], holdout: set[str]) -> dict:
    """For each held-out case with a known (origin, destination) pair,
    count how many training cases share the same pair. Quantifies the
    in-distribution residual the agent's protocol-validation pass flagged:
    held-out is not strictly OOD because station-pair templates persist
    across the partition by construction."""
    train_od_counts: dict[tuple[str, str], int] = defaultdict(int)
    train_od_by_system: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    for c in cases:
        if c["case_id"] in train and c["od"]:
            train_od_counts[c["od"]] += 1
            train_od_by_system[c["system"]][c["od"]] += 1

    per_holdout_neighbors: list[int] = []
    per_system_neighbors: dict[str, list[int]] = defaultdict(list)
    holdout_no_od = 0
    for c in cases:
        if c["case_id"] not in holdout:
            continue
        if not c["od"]:
            holdout_no_od += 1
            continue
        n_neighbors = train_od_by_system[c["system"]].get(c["od"], 0)
        per_holdout_neighbors.append(n_neighbors)
        per_system_neighbors[c["system"]].append(n_neighbors)

    def stats(xs: list[int]) -> dict:
        if not xs:
            return {"n": 0}
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        return {
            "n": n,
            "with_zero_neighbors": sum(1 for x in xs_sorted if x == 0),
            "with_one_or_more": sum(1 for x in xs_sorted if x >= 1),
            "median": xs_sorted[n // 2],
            "p90": xs_sorted[min(n - 1, int(n * 0.9))],
            "max": xs_sorted[-1],
        }

    return {
        "summary": {
            "holdout_cases_without_OD_pair": holdout_no_od,
            **stats(per_holdout_neighbors),
        },
        "by_system": {sys: stats(xs) for sys, xs in per_system_neighbors.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frac", type=float, default=0.25, help="target held-out fraction")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="don't write the file; just print stats")
    args = parser.parse_args()

    cases = load_cases()
    print(f"Loaded {len(cases)} cases from {len(SYSTEMS)} systems.", file=sys.stderr)

    train, holdout = partition(cases, args.seed, args.frac)

    print("\nInvariant checks:", file=sys.stderr)
    fails = check_invariants(cases, train, holdout, args.frac)
    if fails:
        print("  HARD-FAIL invariants:", file=sys.stderr)
        for f in fails:
            print(f"    {f}", file=sys.stderr)
        sys.exit(1)
    print("  all hard invariants passed.", file=sys.stderr)

    warnings = soft_warnings(cases, holdout)
    if warnings:
        print("\nSoft-warn per-cell checks:", file=sys.stderr)
        for w in warnings:
            print(w, file=sys.stderr)

    overlap = template_overlap(cases, train, holdout)
    s = overlap["summary"]
    print("\nTemplate-overlap report:", file=sys.stderr)
    print(f"  held-out cases with no train OD-neighbor: {s.get('with_zero_neighbors', 0)} / {s.get('n', 0)}",
          file=sys.stderr)
    print(f"  per-holdout train-neighbors: median={s.get('median')}, p90={s.get('p90')}, max={s.get('max')}",
          file=sys.stderr)

    per_system_summary = {}
    per_category_summary: dict[str, dict] = defaultdict(lambda: {"all": 0, "train": 0, "holdout": 0})
    per_cell: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"train": 0, "holdout": 0}))

    for c in cases:
        sys_, cat, cid = c["system"], c["category"], c["case_id"]
        per_category_summary[cat]["all"] += 1
        if cid in train:
            per_cell[sys_][cat]["train"] += 1
            per_category_summary[cat]["train"] += 1
        else:
            per_cell[sys_][cat]["holdout"] += 1
            per_category_summary[cat]["holdout"] += 1

    for system in SYSTEMS:
        sys_cases = [c for c in cases if c["system"] == system]
        sys_total = len(sys_cases)
        sys_hold = sum(1 for c in sys_cases if c["case_id"] in holdout)
        per_system_summary[system] = {
            "all": sys_total,
            "train": sys_total - sys_hold,
            "holdout": sys_hold,
            "pct_holdout": round(100 * sys_hold / sys_total, 2),
        }

    for cat, d in per_category_summary.items():
        d["pct_holdout"] = round(100 * d["holdout"] / d["all"], 2) if d["all"] else 0

    print("\nPer-system breakdown:", file=sys.stderr)
    for system, s in per_system_summary.items():
        print(f"  {system:<8} all={s['all']:3d}  train={s['train']:3d}  holdout={s['holdout']:3d}  ({s['pct_holdout']}%)",
              file=sys.stderr)

    print("\nPer-category breakdown:", file=sys.stderr)
    for cat in sorted(per_category_summary):
        d = per_category_summary[cat]
        print(f"  Cat {cat}  all={d['all']:3d}  train={d['train']:3d}  holdout={d['holdout']:3d}  ({d['pct_holdout']}%)",
              file=sys.stderr)

    payload = {
        "spec_version": "v23-holdout75-seed42",
        "seed": args.seed,
        "frac_holdout": args.frac,
        "stratify_by": "system",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totals": {
            "all": len(cases),
            "train": len(train),
            "holdout": len(holdout),
        },
        "per_system": per_system_summary,
        "per_category": dict(per_category_summary),
        "per_cell": {sys_: dict(cat_d) for sys_, cat_d in per_cell.items()},
        "pinned_gap_audit": sorted(PINNED_GAP_AUDIT),
        "train_ids": sorted(train),
        "holdout_ids": sorted(holdout),
        "template_overlap_report": overlap,
    }

    if args.dry_run:
        print(f"\n--dry-run; not writing {args.output}", file=sys.stderr)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=False))
    size_kb = args.output.stat().st_size // 1024
    print(f"\nWrote {args.output} ({size_kb} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
