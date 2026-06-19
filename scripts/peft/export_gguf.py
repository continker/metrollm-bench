#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and export to GGUF Q4_K_M.

The merged model is exported to GGUF Q4_K_M (e.g.
Qwen3.5-2B-metro-v24-Q4_K_M.gguf) for serving with llama.cpp / llama-server.

Usage:
    python export_gguf.py \
        --adapter ./lora-adapter/adapter \
        --base    ./Qwen3.5-2B \
        --output  ./merged \
        --gguf-name Qwen3.5-2B-metro-v24
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=os.path.expanduser(
        "~/metrollm-peft/lora-adapter/adapter"))
    parser.add_argument("--base", default=os.path.expanduser(
        "~/metrollm-peft/Qwen3.5-2B"))
    parser.add_argument("--output", default=os.path.expanduser(
        "~/metrollm-peft/merged"))
    parser.add_argument("--gguf-name", default="Qwen3.5-2B-metro-v24",
                        help="Base name for the output GGUF (without extension)")
    parser.add_argument("--quant", default="q4_k_m",
                        help="Quantization method: q4_k_m, q6_k, f16 (default: q4_k_m)")
    parser.add_argument("--models-dir", default=os.path.expanduser("~/models"),
                        help="Where to copy the final GGUF for llama-server")
    args = parser.parse_args()

    import torch
    from unsloth import FastLanguageModel

    print(f"Loading base + adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    merged_dir = Path(args.output)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Save merged model (safetensors)
    print(f"Saving merged model to {merged_dir} ...")
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    # Export to GGUF
    gguf_dir = Path(args.models_dir)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = gguf_dir / f"{args.gguf_name}-{args.quant.upper()}.gguf"

    print(f"Exporting to GGUF ({args.quant}) -> {gguf_path} ...")
    model.save_pretrained_gguf(
        str(merged_dir / args.gguf_name),
        tokenizer,
        quantization_method=args.quant,
    )

    # Unsloth puts the file next to the output path — move to models dir
    candidate = Path(str(merged_dir / args.gguf_name) + f"-{args.quant.upper()}.gguf")
    if candidate.exists() and not gguf_path.exists():
        import shutil
        shutil.move(str(candidate), str(gguf_path))
        print(f"Moved GGUF to {gguf_path}")
    elif gguf_path.exists():
        print(f"GGUF already at {gguf_path}")
    else:
        # Try to find it
        gguf_candidates = list(merged_dir.glob("*.gguf"))
        if gguf_candidates:
            import shutil
            shutil.move(str(gguf_candidates[0]), str(gguf_path))
            print(f"Moved GGUF from {gguf_candidates[0]} to {gguf_path}")
        else:
            print(f"WARNING: could not locate output GGUF. Check {merged_dir}")
            return

    print(f"\nDone. GGUF at: {gguf_path}")
    print(f"\nAdd a preset to your llama-server config:")
    preset_id = args.gguf_name.lower().replace(".", "").replace("-", "_")
    print(f"""
[{preset_id}]
model = {gguf_path}
n-gpu-layers = 999
ctx-size = 262144
parallel = 2
flash-attn = on
kv-unified = true
no-mmap = true
""")


if __name__ == "__main__":
    main()
