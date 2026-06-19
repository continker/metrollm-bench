#!/usr/bin/env python3
"""Compute Table 3 (held-out pilot) averages by filtering the 15 v23 gap-audit
case IDs from the full-run scored files.

Usage:
    uv run python scripts/compute_heldout.py

Source files are the per-system scored JSONs under `results/v23_*`. The script
emits the Tier-1 and Composite percentages as cross-system means across the
15 filtered cases for each of the 5 headline models in paper_v3 Table 3.
"""

from __future__ import annotations

import json
from pathlib import Path

HELDOUT_IDS: dict[str, list[str]] = {
    "bart": ["BART-B-016", "BART-F-016"],
    "beijing": ["BJM-A-021", "BJM-B-016", "BJM-C-022"],
    "cta": ["CTA-C-018", "CTA-F-016"],
    "doha": ["DOHA-C-016", "DOHA-E-006", "DOHA-E-007"],
    "marta": ["MARTA-D-016", "MARTA-F-016", "MARTA-F-017"],
    "taipei": ["TRTC-B-016", "TRTC-C-018"],
}

MODELS: list[tuple[str, Path, str]] = [
    # (display name, result dir, per-system scored filename pattern)
    ("Qwen 27B base",            Path("results/v23_scaling"), "base_27b_{sys}_scored.json"),
    ("Qwen 9B + metro-v24 LoRA", Path("results/v23_peft"),    "ft_v23_9b_{sys}_scored.json"),
    ("Qwen 27B + metro-v24 LoRA",Path("results/v23_peft"),    "ft_v23_27b_{sys}_scored.json"),
    ("Qwen 4B + metro-v24 LoRA", Path("results/v23_peft"),    "ft_v23_4b_{sys}_scored.json"),
    ("Qwen 35B-A3B base",        Path("results/v23_scaling"), "base_35b_{sys}_scored.json"),
    ("Qwen 9B base",             Path("results/v23_scaling"), "base_9b_{sys}_scored.json"),
    ("Qwen 4B base",             Path("results/v23_scaling"), "base_4b_{sys}_scored.json"),
    ("GPT-5.4 full",             Path("results/v23_full"),    "{sys}_scored.json"),
    ("Mistral Small 2603",       Path("results/v23_mistral_small4"), "{sys}_scored.json"),
]


def heldout_averages(base_dir: Path, pattern: str) -> tuple[int, float, float]:
    rows: list[tuple[float, float]] = []
    for system, case_ids in HELDOUT_IDS.items():
        path = base_dir / pattern.format(sys=system)
        if not path.exists():
            continue
        scored = json.loads(path.read_text())
        wanted = set(case_ids)
        for entry in scored.get("scores", []):
            if entry.get("case_id") in wanted:
                rows.append((entry.get("tier1_pct", 0.0), entry.get("pct", 0.0)))
    if not rows:
        return 0, 0.0, 0.0
    tier1 = sum(r[0] for r in rows) / len(rows)
    composite = sum(r[1] for r in rows) / len(rows)
    return len(rows), tier1, composite


def main() -> None:
    print(f"{'Model':<34} {'n':>3}  {'Tier-1 %':>9}  {'Composite %':>11}")
    print("-" * 62)
    for name, base_dir, pattern in MODELS:
        n, tier1, composite = heldout_averages(base_dir, pattern)
        if n:
            print(f"{name:<34} {n:>3}  {tier1:>9.1f}  {composite:>11.1f}")
        else:
            print(f"{name:<34} {'MISS':>3}")


if __name__ == "__main__":
    main()
