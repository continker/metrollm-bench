#!/usr/bin/env python3
"""
Upload a single metro-v24 PEFT variant to continker/ on HuggingFace.

Per size, creates `continker/Qwen3.5-{N}B-metro-v24` and uploads:
  - Qwen3.5-{N}B-metro-v24-Q4_K_M.gguf  (runtime artifact for llama.cpp)
  - adapter/ (LoRA weights + tokenizer)
  - README.md (model card)
  - training_summary.json (hyperparams, seed, dataset version)

Usage:
    HF_TOKEN=hf_xxx python3 scripts/peft/upload_hf.py --size 2b
    HF_TOKEN=hf_xxx python3 scripts/peft/upload_hf.py --size 27b --include-seed-2

Set --dry-run to print what would be uploaded without touching HF.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EVAL = {
    "2b":  {"params_b": 2,  "gguf_gb": 1.2,  "tier1": 81.5, "comp": 80.1, "delta_t1": +6.2, "delta_c": +6.9},
    "4b":  {"params_b": 4,  "gguf_gb": 2.6,  "tier1": 91.4, "comp": 88.8, "delta_t1": +2.2, "delta_c": +1.8},
    "9b":  {"params_b": 9,  "gguf_gb": 5.3,  "tier1": 92.4, "comp": 90.0, "delta_t1": +2.2, "delta_c": +1.4},
    "27b": {"params_b": 27, "gguf_gb": 16,   "tier1": 91.0, "comp": 89.4, "delta_t1": -1.6, "delta_c": -1.2},
}

PER_SYSTEM_T1 = {
    "2b":  {"marta": 84.0, "bart": 80.7, "cta": 82.7, "doha": 81.0, "taipei": 80.5, "beijing": 80.1},
    "4b":  {"marta": 93.1, "bart": 91.4, "cta": 91.9, "doha": 92.2, "taipei": 91.6, "beijing": 88.3},
    "9b":  {"marta": 94.0, "bart": 90.7, "cta": 93.4, "doha": 93.1, "taipei": 92.6, "beijing": 90.7},
    "27b": {"marta": 92.0, "bart": 91.1, "cta": 92.5, "doha": 89.6, "taipei": 91.1, "beijing": 89.6},
}


def make_model_card(size: str) -> str:
    e = EVAL[size]
    sys_table = "\n".join(
        f"| {sys.upper()} | {t1:.1f} |"
        for sys, t1 in PER_SYSTEM_T1[size].items()
    )
    sign = "+" if e["delta_t1"] >= 0 else ""
    sign_c = "+" if e["delta_c"] >= 0 else ""
    return f"""---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen3.5-{size.upper()}
tags:
- transit
- kiosk
- tool-use
- agent
- metrollm-bench
- qwen
- lora
- gguf
- quantized
library_name: peft
pipeline_tag: text-generation
---

# Qwen3.5-{size.upper()} + metro-v24 LoRA

Domain-specialised tool-using agent for transit-kiosk tasks: routing, fare calculation,
disruption advisories, accessibility, multilingual cultural notes, multi-turn context tracking,
and policy adaptation across 6 metro systems (MARTA, BART, CTA, Doha, Taipei MRT, Beijing
Subway).

QLoRA r=16 fine-tune of `Qwen/Qwen3.5-{size.upper()}` on 790 distilled traces from
Qwen3.5-27B and Qwen3.5-35B-A3B teachers (filtered to tier1 ≥ 90% per case, deduplicated
by case_id, evaluated on the [MetroLLM-Bench](https://github.com/...) v23 harness).

## Files

| File | Purpose |
|---|---|
| `Qwen3.5-{size.upper()}-metro-v24-Q4_K_M.gguf` ({e['gguf_gb']:.1f} GB) | Runtime artifact for llama.cpp / Ollama |
| `adapter/` | Raw LoRA adapter (use with PEFT + base Qwen3.5-{size.upper()}) |
| `training_summary.json` | Hyperparameters, seed, dataset version |

## Eval (v23, 6 systems, Haiku judge for Tier 2)

**Cross-system average**: Tier-1 {e['tier1']:.1f}, Composite {e['comp']:.1f}
({sign}{e['delta_t1']:.1f} T1 / {sign_c}{e['delta_c']:.1f} Comp vs base Qwen3.5-{size.upper()})

| System | Tier-1 % |
|---|---:|
{sys_table}

## Quickstart (llama.cpp)

```bash
huggingface-cli download continker/Qwen3.5-{size.upper()}-metro-v24 \\
  Qwen3.5-{size.upper()}-metro-v24-Q4_K_M.gguf --local-dir ./models

llama-server -m ./models/Qwen3.5-{size.upper()}-metro-v24-Q4_K_M.gguf \\
  --port 8080 --ctx-size 32768 --n-gpu-layers 999
```

## Quickstart (PEFT adapter, Python)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-{size.upper()}", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "continker/Qwen3.5-{size.upper()}-metro-v24", subfolder="adapter")
tokenizer = AutoTokenizer.from_pretrained("continker/Qwen3.5-{size.upper()}-metro-v24", subfolder="adapter")
```

## Training

- Base: `Qwen/Qwen3.5-{size.upper()}`
- Method: QLoRA, rank=16, alpha=32, dropout=0.05
- Targets: q/k/v/o + gate/up/down projections
- Optimizer: AdamW, lr=2e-4, cosine, warmup 5%
- Epochs: 3, effective batch 8 (per_device_train_batch_size=2 × grad_accum=4)
- Max sequence length: {2048 if size == '27b' else 4096}
- Seed: 42 (default; multi-seed CI in progress for 27B)
- Dataset: 790 distilled examples, see [continker/metrollm-bench-train-data-v23](https://huggingface.co/datasets/continker/metrollm-bench-train-data-v23)

## Limitations

- Trained on 6 metro systems; generalisation to other systems untested.
- Tool-use schema is specific to the MetroLLM-Bench mock server (route_planner,
  fare_calculator, station_info, disruption_feed, knowledge_base,
  submit_assistant_state).
- Quantised to 4-bit (Q4_K_M); for full-precision behaviour use the adapter on
  bf16 base weights.

## Citation

```
@misc{{metrollm-bench-2026,
  title={{MetroLLM-Bench: Evaluating LLMs as Prompt-Driven Transit Kiosk Agents}},
  author={{Hendriks, Remco and contributors}},
  year={{2026}},
  publisher={{HuggingFace}},
  howpublished={{\\url{{https://huggingface.co/continker}}}}
}}
```
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--size", required=True, choices=["2b", "4b", "9b", "27b"])
    p.add_argument("--org", default="continker", help="HF org (default: continker)")
    p.add_argument("--gguf-dir", default=os.path.expanduser("~/models"),
                   help="Directory containing GGUF (default: ~/models)")
    p.add_argument("--peft-dir", default=os.path.expanduser("~/metrollm-peft"),
                   help="PEFT working dir containing lora-v23-{size}/adapter")
    p.add_argument("--include-seed-2", action="store_true",
                   help="Also upload seed-2 GGUF + adapter under seed-43/ subdir (27B only)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--public", action="store_true", default=True,
                   help="Create public repo (default; required for free HF org bandwidth)")
    p.add_argument("--private", action="store_true", help="Override: create private repo")
    return p.parse_args()


def main():
    args = parse_args()
    size_up = args.size.upper()
    repo_id = f"{args.org}/Qwen3.5-{size_up}-metro-v24"

    gguf_dir = Path(args.gguf_dir)
    peft_dir = Path(args.peft_dir)
    gguf = gguf_dir / f"Qwen3.5-{size_up}-metro-v24-Q4_K_M.gguf"
    adapter_dir = peft_dir / f"lora-v23-{args.size}" / "adapter"
    summary_json = peft_dir / f"lora-v23-{args.size}" / "training_summary.json"

    seed_2_gguf = gguf_dir / f"Qwen3.5-{size_up}-metro-v24-s2-Q4_K_M.gguf"
    seed_2_adapter = peft_dir / f"lora-v23-{args.size}-s2" / "adapter"
    seed_2_summary = peft_dir / f"lora-v23-{args.size}-s2" / "training_summary.json"

    # Preflight
    missing = []
    for p in [gguf, adapter_dir, summary_json]:
        if not p.exists():
            missing.append(str(p))
    if args.include_seed_2:
        if args.size != "27b":
            print("ERROR: --include-seed-2 only valid for --size 27b", file=sys.stderr)
            sys.exit(2)
        for p in [seed_2_gguf, seed_2_adapter, seed_2_summary]:
            if not p.exists():
                missing.append(str(p))
    if missing:
        print("ERROR: missing artifacts:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    print(f"Repo: https://huggingface.co/{repo_id}")
    print(f"GGUF: {gguf} ({gguf.stat().st_size / 1e9:.2f} GB)")
    print(f"Adapter: {adapter_dir}")
    if args.include_seed_2:
        print(f"Seed-2 GGUF: {seed_2_gguf} ({seed_2_gguf.stat().st_size / 1e9:.2f} GB)")
        print(f"Seed-2 adapter: {seed_2_adapter}")

    if args.dry_run:
        print("\n[dry-run] Skipping upload.")
        print("\n--- Model card preview (first 30 lines) ---")
        for line in make_model_card(args.size).splitlines()[:30]:
            print(line)
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(2)

    from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

    api = HfApi(token=token)
    private = args.private  # default public

    print(f"\nCreating/verifying repo {repo_id} (private={private})...")
    create_repo(repo_id, token=token, private=private, exist_ok=True, repo_type="model")

    # 1) Model card
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=make_model_card(args.size).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
        commit_message=f"Add model card for {repo_id}",
    )

    # 2) Training summary
    print("Uploading training_summary.json...")
    api.upload_file(
        path_or_fileobj=str(summary_json),
        path_in_repo="training_summary.json",
        repo_id=repo_id,
        token=token,
        commit_message="Add training_summary.json",
    )

    # 3) GGUF (large file, LFS auto-handled)
    print(f"Uploading GGUF ({gguf.stat().st_size / 1e9:.2f} GB)...")
    api.upload_file(
        path_or_fileobj=str(gguf),
        path_in_repo=gguf.name,
        repo_id=repo_id,
        token=token,
        commit_message=f"Add Q4_K_M GGUF",
    )

    # 4) Adapter folder
    print("Uploading adapter/...")
    api.upload_folder(
        folder_path=str(adapter_dir),
        path_in_repo="adapter",
        repo_id=repo_id,
        token=token,
        commit_message="Add LoRA adapter + tokenizer",
        ignore_patterns=["checkpoints/*", "*.bin"],
    )

    # 5) Optional seed-2 artifacts (27B only)
    if args.include_seed_2:
        print(f"Uploading seed-2 GGUF ({seed_2_gguf.stat().st_size / 1e9:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=str(seed_2_gguf),
            path_in_repo=f"seed-43/{seed_2_gguf.name}",
            repo_id=repo_id,
            token=token,
            commit_message="Add seed=43 GGUF for multi-seed CI",
        )
        print("Uploading seed-2 adapter/...")
        api.upload_folder(
            folder_path=str(seed_2_adapter),
            path_in_repo="seed-43/adapter",
            repo_id=repo_id,
            token=token,
            commit_message="Add seed=43 LoRA adapter",
            ignore_patterns=["checkpoints/*"],
        )
        api.upload_file(
            path_or_fileobj=str(seed_2_summary),
            path_in_repo="seed-43/training_summary.json",
            repo_id=repo_id,
            token=token,
        )

    print(f"\nDone. https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
