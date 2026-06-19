"""Filter scored.json files by partition (train/holdout/full) and recompute
cross-system metrics. Used for rev3 paper tables.

Reads the split spec from `data/splits/v23_holdout75_seed42.json` and for
each (model_tag, system) scored file in the given results directory,
filters the per-case scores to the requested partition and recomputes
tier1_composite + metrollm_composite using harness.scorer.compute_metrics.

Usage:
  uv run python scripts/score_split.py \\
      --results-dir results/v23_clean \\
      --tags ft_v23_clean_2b,ft_v23_clean_2b_s2,ft_v23_clean_4b,... \\
      --partition holdout

Prints a markdown table grouped by tag, plus the cross-system averages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ["marta", "bart", "cta", "doha", "taipei", "beijing"]
DEFAULT_SPLIT = REPO_ROOT / "data/splits/v23_holdout75_seed42.json"

# Make harness importable so we can reuse compute_metrics.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.scorer import compute_metrics  # noqa: E402


def load_split(split_path: Path) -> tuple[set[str], set[str]]:
    """Return (train_ids, holdout_ids) sets."""
    data = json.loads(split_path.read_text())
    return set(data["train_ids"]), set(data["holdout_ids"])


def filter_and_compute(scored_path: Path, partition_ids: set[str] | None) -> dict | None:
    """Load a scored.json, filter scores+results to partition (or keep all
    if partition_ids is None for full-matrix), recompute metrics."""
    if not scored_path.exists():
        return None
    data = json.loads(scored_path.read_text())
    all_scores = data.get("scores") or []
    all_results = data.get("results") or []
    # results may also be embedded under a different key; we don't strictly
    # need them for tier1_composite, but compute_metrics signature requires
    # them. Fall back to empty list if missing.

    if partition_ids is None:
        scores = all_scores
        results = all_results
    else:
        scores = [s for s in all_scores if s.get("case_id") in partition_ids]
        results = [r for r in all_results if r.get("case_id") in partition_ids]

    if not scores:
        return None

    metrics = compute_metrics(scores, results)
    metrics["_n"] = len(scores)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--tags",
        required=True,
        help="comma-separated model tag prefixes (e.g. ft_v23_clean_2b,...)",
    )
    parser.add_argument(
        "--partition",
        choices=["train", "holdout", "full"],
        default="holdout",
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    args = parser.parse_args()

    train_ids, holdout_ids = load_split(args.split)
    if args.partition == "train":
        partition_ids = train_ids
    elif args.partition == "holdout":
        partition_ids = holdout_ids
    else:
        partition_ids = None

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    print(
        f"# Cross-system summary on partition='{args.partition}' "
        f"(n_cases={len(partition_ids) if partition_ids else 955})",
        file=sys.stderr,
    )

    print(f"{'tag':<24} {'sys':<10} {'n':>4} {'T1':>7} {'Comp':>7}")
    print("-" * 56)

    per_tag_t1 = {}
    per_tag_comp = {}
    for tag in tags:
        t1s, cs, ns = [], [], []
        for sys_ in SYSTEMS:
            scored_path = args.results_dir / f"{tag}_{sys_}_scored.json"
            metrics = filter_and_compute(scored_path, partition_ids)
            if metrics is None:
                print(f"{tag:<24} {sys_:<10}    MISS")
                continue
            t1 = metrics.get("tier1_composite", 0)
            c = metrics.get("metrollm_composite", 0)
            n = metrics.get("_n", 0)
            print(f"{tag:<24} {sys_:<10} {n:>4} {t1:>7.2f} {c:>7.2f}")
            t1s.append(t1)
            cs.append(c)
            ns.append(n)
        if t1s:
            avg_t1 = sum(t1s) / len(t1s)
            avg_c = sum(cs) / len(cs)
            per_tag_t1[tag] = avg_t1
            per_tag_comp[tag] = avg_c
            print(f"{tag:<24} {'AVG':<10} {sum(ns):>4} {avg_t1:>7.2f} {avg_c:>7.2f}")
        print()

    # n=2 means per size
    print(f"# n=2 means per size on partition='{args.partition}'")
    sizes_seen = set()
    for tag in tags:
        for size in ["2b", "4b", "9b", "27b"]:
            if not tag.endswith(f"_{size}") and not tag.endswith(f"_{size}_s2"):
                continue
            sizes_seen.add(size)
    for size in sorted(sizes_seen, key=lambda s: ["2b", "4b", "9b", "27b"].index(s)):
        t42 = per_tag_t1.get(f"ft_v23_clean_{size}")
        t43 = per_tag_t1.get(f"ft_v23_clean_{size}_s2")
        if t42 is None or t43 is None:
            continue
        mean = (t42 + t43) / 2
        spread = abs(t42 - t43) / 2
        print(f"  {size:<4}  s42={t42:>6.2f}  s43={t43:>6.2f}  mean={mean:>6.2f}  spread=±{spread:.2f}")


if __name__ == "__main__":
    main()
