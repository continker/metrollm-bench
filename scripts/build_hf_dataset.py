#!/usr/bin/env python3
"""Build the MetroLLM-Bench cases as a Hugging Face dataset folder (mirror of this repo).

The six case files under cases/ stay the source of truth. This script flattens
them into one row per case, assigns the train/holdout partition from the split
file, and writes two JSON Lines files plus a dataset card into a local folder.
Uploading that folder to the Hub is a separate step.

    uv run python scripts/build_hf_dataset.py            # -> results/hf_dataset/

Row layout: eleven plain columns (id, system, category, difficulty,
interaction_mode, title, partition, expected_outcome, expected_kiosk_action,
fare_total, fare_currency) and six JSON-string columns (events, system_context,
ground_truth, scoring, tolerances, extras) that carry the complete case verbatim.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYSTEMS = ["marta", "doha", "bart", "taipei", "cta", "beijing"]
SYSTEM_NAMES = {
    "marta": "MARTA (Atlanta)", "doha": "Doha Metro", "bart": "BART (San Francisco Bay Area)",
    "taipei": "Taipei MRT", "cta": "CTA L (Chicago)", "beijing": "Beijing Subway",
}
CATEGORY_NAMES = {
    "A": "Routing", "B": "Fare calculation", "C": "Disruptions", "D": "Accessibility",
    "E": "Cultural and multilingual", "F": "Policy change", "G": "Multi-turn",
    "H": "Adversarial", "I": "Temporal reasoning", "J": "Tool hallucination", "K": "Compound stress",
}
PLAIN = ["id", "system", "category", "difficulty", "interaction_mode", "title"]
JSON_COLS = ["events", "system_context", "ground_truth", "scoring", "tolerances"]


def flatten(case: dict, partition: str) -> dict:
    gt = case["ground_truth"]
    fare = gt.get("fare") or {}
    row = {k: case.get(k) for k in PLAIN}
    row["partition"] = partition
    row["expected_outcome"] = gt.get("expected_outcome")
    row["expected_kiosk_action"] = gt.get("expected_kiosk_action")
    row["fare_total"] = float(fare["total"]) if isinstance(fare, dict) and fare.get("total") is not None else None
    row["fare_currency"] = fare.get("currency") if isinstance(fare, dict) else None
    for k in JSON_COLS:
        row[k] = json.dumps(case[k], ensure_ascii=False, sort_keys=True)
    extras = {k: v for k, v in case.items() if k not in PLAIN and k not in JSON_COLS}
    row["extras"] = json.dumps(extras, ensure_ascii=False, sort_keys=True)
    return row


def build(cases_dir: Path, split_file: Path) -> tuple[list[dict], list[dict], dict]:
    split = json.loads(split_file.read_text())
    train_ids, holdout_ids = set(split["train_ids"]), set(split["holdout_ids"])
    assert not (train_ids & holdout_ids), "split lists overlap"
    train, holdout = [], []
    seen = set()
    for system in SYSTEMS:
        for case in json.loads((cases_dir / f"{system}_cases.json").read_text()):
            cid = case["id"]
            assert cid not in seen, f"duplicate id {cid}"
            seen.add(cid)
            if cid in train_ids:
                train.append(flatten(case, "train"))
            elif cid in holdout_ids:
                holdout.append(flatten(case, "holdout"))
            else:
                raise SystemExit(f"{cid} is in neither split list")
    missing = (train_ids | holdout_ids) - seen
    assert not missing, f"split ids without a case: {sorted(missing)[:5]}"
    return train, holdout, split


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def card(train: list[dict], holdout: list[dict], split: dict, repo_id: str) -> str:
    rows = train + holdout
    by_system = collections.Counter(r["system"] for r in rows)
    by_cat = collections.Counter(r["category"] for r in rows)
    by_system_split = collections.Counter((r["system"], r["partition"]) for r in rows)
    system_table = "\n".join(
        f"| {SYSTEM_NAMES[s]} | `{s}` | {by_system[s]} | {by_system_split[(s, 'train')]} | {by_system_split[(s, 'holdout')]} |"
        for s in SYSTEMS)
    cat_table = "\n".join(f"| {c} | {CATEGORY_NAMES[c]} | {by_cat[c]} |" for c in sorted(by_cat))
    return f"""---
license: apache-2.0
pretty_name: MetroLLM-Bench
language:
- en
task_categories:
- text-generation
tags:
- benchmark
- tool-calling
- agents
- transit
- kiosk
- llm-evaluation
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.jsonl
  - split: holdout
    path: data/holdout-*.jsonl
---

# MetroLLM-Bench

The {len(rows)} cases of **MetroLLM-Bench**, a benchmark for language models as the policy layer of a
transit kiosk. The model receives kiosk events, calls structured tools (route planner, fare
calculator, station info, disruption feed, knowledge base) and submits a terminal state: outcome,
fare quote where applicable, kiosk action.

Paper: [MetroLLM-Bench: Evaluating Language Models as Transit Kiosk Runtimes](https://arxiv.org/abs/2609.10016)
(arXiv:2609.10016, [HF paper page](https://huggingface.co/papers/2609.10016)). Code, harness and ground-truth generator:
[github.com/continker/metrollm-bench](https://github.com/continker/metrollm-bench).
Fine-tuned students: the [MetroLLM-Bench v24 collection](https://huggingface.co/collections/continker/metrollm-bench-v24-6a35b586a11068e1b1ba3d47).
Live HF Space demo: [remcohendriks/metrollm](https://huggingface.co/spaces/remcohendriks/metrollm).

## Case details

One row per case. Each row holds the kiosk event sequence (station selected, passenger count
changed, disruption update, free-text input), the system context (current time, active
disruptions, framebook) and the ground truth used for scoring (expected route, fare, outcome,
kiosk action, and category-specific expectations such as traps or policies).

Scoring requires the harness in the repository: the mock tool server backed by the six system
datasets, the runner, and the 22-component scorer with its language-model judge. See
`REPRODUCING.md` there.

## Systems

| System | `system` | Cases | Train | Held-out |
|---|---|---:|---:|---:|
{system_table}

## Categories

| `category` | Name | Cases |
|---|---|---:|
{cat_table}

## Splits

`train` ({len(train)} cases) and `holdout` ({len(holdout)} cases) form the paper's stratified 75/25
case-level partition (seed {split["seed"]}, stratified by {split["stratify_by"]}). Training data for
the students came from `train` only; all held-out numbers in the paper are computed on `holdout`.

## Row schema

| Column | Type | Content |
|---|---|---|
| `id` | string | case id, e.g. `MARTA-A-001` |
| `system` | string | one of the six system keys |
| `category` | string | `A` to `K` |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `interaction_mode` | string | `structured`, `freetext`, `multi_turn`, `adversarial`, `hallucination_probe`, `compound` |
| `title` | string | short scenario title |
| `partition` | string | `train` or `holdout` (same as the split) |
| `expected_outcome` | string | one of `route_and_fare_ready`, `advisory_only`, `service_unavailable`, `request_declined`, `policy_answer_only` |
| `expected_kiosk_action` | string | the kiosk action the terminal state must carry |
| `fare_total` | float or null | expected fare; null where no fare applies |
| `fare_currency` | string or null | `USD`, `QAR`, `TWD`, `CNY` |
| `events` | string (JSON) | the kiosk event sequence, verbatim |
| `system_context` | string (JSON) | current time, active disruptions, framebook, feature toggles |
| `ground_truth` | string (JSON) | the complete ground truth, all keys |
| `scoring` | string (JSON) | point weights of the scoring components that apply to this case |
| `tolerances` | string (JSON) | numeric tolerances used by the scorer |
| `extras` | string (JSON) | category-specific fields (scenario, trap, policy, temporal, accessibility, cultural ids) |

The JSON columns hold the original case objects verbatim; `json.loads` returns the same objects as
`cases/<system>_cases.json`.

## Loading

```python
from datasets import load_dataset
import json

ds = load_dataset("{repo_id}")
row = ds["holdout"][0]
events = json.loads(row["events"])
ground_truth = json.loads(row["ground_truth"])
```

## Citation

```bibtex
@techreport{{hendriks2026metrollm,
  title       = {{MetroLLM-Bench: Evaluating Language Models as Transit Kiosk Runtimes}},
  author      = {{Hendriks, Remco}},
  institution = {{Continker}},
  type        = {{Technical report}},
  number      = {{v1.2}},
  year        = {{2026}},
  month       = {{9}},
  doi         = {{10.5281/zenodo.21893944}},
  eprint      = {{2609.10016}},
  archiveprefix = {{arXiv}},
  primaryclass = {{cs.LG}},
  url         = {{https://arxiv.org/abs/2609.10016}}
}}
```

## License

Apache License 2.0, as the code, ground truth and framebooks in the repository.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="continker/metrollm-bench", help="dataset id used in the card's loading example")
    ap.add_argument("--cases-dir", type=Path, default=REPO / "cases")
    ap.add_argument("--split-file", type=Path, default=REPO / "data/splits/v23_holdout75_seed42.json")
    ap.add_argument("--out", type=Path, default=REPO / "results/hf_dataset", help="local build folder (gitignored)")
    args = ap.parse_args()

    train, holdout, split = build(args.cases_dir, args.split_file)
    out = args.out
    write_jsonl(out / "data/train-00000-of-00001.jsonl", train)
    write_jsonl(out / "data/holdout-00000-of-00001.jsonl", holdout)
    (out / "README.md").write_text(card(train, holdout, split, args.repo_id), encoding="utf-8")
    print(f"built {out}: train={len(train)} holdout={len(holdout)} columns={len(train[0])}")


if __name__ == "__main__":
    main()
