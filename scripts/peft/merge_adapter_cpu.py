#!/usr/bin/env python3
"""
CPU-only merge for LoRA adapter → bf16 safetensors.

Use this for models too large to fit in one GPU (e.g., 27B at bf16 needs
~54 GB, which overflows 32 GB VRAM and triggers accelerate's balanced-memory
path — which has a version-mismatch bug: `TypeError: unhashable type: 'set'`
in get_balanced_memory).

Bypasses unsloth entirely. Loads base + adapter on CPU, merges on CPU,
saves bf16 safetensors.
"""

import argparse
import gc
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
                        help="HF repo name (e.g., Qwen/Qwen3.5-27B) or local dir")
    parser.add_argument("--adapter", required=True,
                        help="Path to LoRA adapter dir (contains adapter_config.json)")
    parser.add_argument("--output", required=True,
                        help="Output dir for merged safetensors")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_name = args.base
    adapter_path = os.path.expanduser(args.adapter)
    output_path = Path(os.path.expanduser(args.output))
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base {base_name} on CPU (bf16)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Attaching adapter {adapter_path}...")
    model = PeftModel.from_pretrained(base, adapter_path, device_map="cpu")

    print("Merging adapter into base (CPU)...")
    merged = model.merge_and_unload()

    del base, model
    gc.collect()

    print(f"Saving merged model to {output_path}...")
    merged.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="5GB",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    tokenizer.save_pretrained(str(output_path))

    print("Merge complete.")


if __name__ == "__main__":
    main()
