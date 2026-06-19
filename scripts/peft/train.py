#!/usr/bin/env python3
"""
LoRA fine-tuning of Qwen3.5-2B on MetroLLM-Bench SFT data.

Usage:
    python train.py --data scripts/peft/train_data.jsonl \
                    --model ~/metrollm-peft/Qwen3.5-2B \
                    --output ~/metrollm-peft/lora-adapter

Config targets a smoke test: rank=16, 3 epochs, ~5GB VRAM.
Scale up rank/epochs for full run.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="scripts/peft/train_data.jsonl")
    parser.add_argument("--model", default=os.path.expanduser("~/metrollm-peft/Qwen3.5-2B"))
    parser.add_argument("--output", default=os.path.expanduser("~/metrollm-peft/lora-adapter"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (16=smoke test, 64=full)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4, help="Effective batch = batch_size * grad_accum")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max tokens per example")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit training examples (smoke test)")
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA (saves VRAM for larger models)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for LoRA init, trainer, and train/val split (vary for multi-seed CI)")
    args = parser.parse_args()

    # --- Imports (heavy, do after arg parse) ---
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Load model ---
    print(f"\nLoading base model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=args.qlora,
    )

    # --- Apply LoRA ---
    print(f"Applying LoRA (rank={args.rank})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.print_trainable_parameters()

    # --- Load data ---
    print(f"\nLoading training data: {args.data}")
    raw = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    if args.max_examples:
        raw = raw[:args.max_examples]
    print(f"  {len(raw)} examples loaded")

    # Apply chat template to each example
    def format_example(ex):
        messages = ex["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    formatted = [format_example(ex) for ex in raw]
    dataset = Dataset.from_list(formatted)

    # Train/val split (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # --- Trainer ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            bf16=True,
            fp16=False,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.save_steps,
            eval_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            dataloader_num_workers=2,
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            packing=False,
            seed=args.seed,
            data_seed=args.seed,
        ),
    )

    # --- Train ---
    print(f"\nStarting training (epochs={args.epochs}, effective_batch={args.batch_size * args.grad_accum})")
    trainer.train()

    # --- Save adapter ---
    adapter_path = str(output_dir / "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to: {adapter_path}")

    # Save training summary
    summary = {
        "base_model": args.model,
        "lora_rank": args.rank,
        "epochs": args.epochs,
        "seed": args.seed,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "data_source": args.data,
        "adapter_path": adapter_path,
    }
    with open(str(output_dir / "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Training complete.")


if __name__ == "__main__":
    main()
