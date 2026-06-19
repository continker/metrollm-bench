"""Slice the v23 case files by the train/holdout split.

Given the split JSON produced by build_holdout_split.py, write per-system
case files containing only the cases in the requested partition. The
output files have the same shape as cases/{system}_cases.json so existing
runners and scorers can consume them with --cases <sliced-path> and no
other change.

Outputs default to:
  cases/{system}_cases_train_split75.json
  cases/{system}_cases_holdout_split75.json

Usage:
    uv run python scripts/slice_cases_by_split.py --partition train
    uv run python scripts/slice_cases_by_split.py --partition holdout
    uv run python scripts/slice_cases_by_split.py --partition both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = REPO_ROOT / "cases"
SYSTEMS = ["marta", "doha", "bart", "taipei", "cta", "beijing"]
DEFAULT_SPLIT = REPO_ROOT / "data/splits/v23_holdout75_seed42.json"


def slice_one(system: str, partition_ids: set[str], suffix: str) -> tuple[Path, int]:
    src = CASES_DIR / f"{system}_cases.json"
    if not src.exists():
        sys.exit(f"FATAL: missing {src}")
    cases = json.loads(src.read_text())
    sliced = [c for c in cases if c["id"] in partition_ids]
    dst = CASES_DIR / f"{system}_cases_{suffix}.json"
    dst.write_text(json.dumps(sliced, indent=2))
    return dst, len(sliced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--partition",
        choices=["train", "holdout", "both"],
        default="train",
        help="which partition to write (default: train)",
    )
    parser.add_argument(
        "--suffix-train",
        default="train_split75",
        help="output filename suffix for the train partition",
    )
    parser.add_argument(
        "--suffix-holdout",
        default="holdout_split75",
        help="output filename suffix for the holdout partition",
    )
    args = parser.parse_args()

    if not args.split.exists():
        sys.exit(f"FATAL: split file not found at {args.split}")
    data = json.loads(args.split.read_text())
    train_ids = set(data["train_ids"])
    holdout_ids = set(data["holdout_ids"])
    print(f"Split spec: {data['spec_version']}", file=sys.stderr)
    print(f"  train: {len(train_ids)}  holdout: {len(holdout_ids)}", file=sys.stderr)

    do_train = args.partition in ("train", "both")
    do_holdout = args.partition in ("holdout", "both")

    print("\nSlicing:", file=sys.stderr)
    for system in SYSTEMS:
        if do_train:
            dst, n = slice_one(system, train_ids, args.suffix_train)
            print(f"  {system:<8} train   → {dst.relative_to(REPO_ROOT)}  ({n} cases)", file=sys.stderr)
        if do_holdout:
            dst, n = slice_one(system, holdout_ids, args.suffix_holdout)
            print(f"  {system:<8} holdout → {dst.relative_to(REPO_ROOT)}  ({n} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
