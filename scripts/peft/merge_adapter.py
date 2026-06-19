#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and save as HF safetensors only.

Separate from export_gguf.py because unsloth's save_pretrained_gguf tries to
build llama.cpp from source and triggers sudo prompts on some boxes. This
script does *only* the merge; use llama.cpp's convert_hf_to_gguf.py +
llama-quantize to produce the GGUF.
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True,
                        help="Path to LoRA adapter dir (contains adapter_config.json)")
    parser.add_argument("--output", required=True,
                        help="Output dir for merged safetensors")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    import torch
    from unsloth import FastLanguageModel

    adapter_path = os.path.expanduser(args.adapter)
    output_path = Path(os.path.expanduser(args.output))
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base + adapter from {adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    print(f"Merging and saving to {output_path}...")
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )
    print("Merge complete.")


if __name__ == "__main__":
    main()
