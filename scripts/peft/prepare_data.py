#!/usr/bin/env python3
"""
Prepare SFT training data from v17 benchmark results.

Reads raw runner output (v17_qwen27b_*.json, v17_qwen35b_*.json),
joins with scored files to filter by tier1_pct, converts messages
to Qwen3.5 chat format, and writes JSONL for training.

Output format: one JSON object per line
  {"messages": [{"role": ..., "content": ...}, ...]}

Tool calls are serialized to <tool_call>...</tool_call> in assistant content.
Tool results stay as role=tool content.
Thinking tokens (reasoning_content) are stripped.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def convert_messages(messages: list[dict]) -> list[dict] | None:
    """
    Convert OpenAI-format messages (with tool_calls dicts) to flat text format
    that Qwen3.5 chat template expects.

    Returns None if conversion fails or conversation is malformed.
    """
    out = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls", [])

        if role == "system":
            out.append({"role": "system", "content": content})

        elif role == "user":
            out.append({"role": "user", "content": content})

        elif role == "assistant":
            if tool_calls:
                # Serialize each tool call to <tool_call> XML
                parts = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = args_raw
                    call_obj = {"name": name, "arguments": args}
                    parts.append(f"<tool_call>\n{json.dumps(call_obj)}\n</tool_call>")
                text = "\n".join(parts)
                out.append({"role": "assistant", "content": text})
            elif content:
                out.append({"role": "assistant", "content": content})
            else:
                # Empty assistant turn (shouldn't happen in good traces)
                return None

        elif role == "tool":
            # Tool results: keep as-is
            out.append({"role": "tool", "content": content})

        else:
            # Unknown role — skip but don't fail
            continue

    return out


def load_scores(scored_path: Path) -> dict[str, float]:
    """Load case_id -> tier1_pct mapping from a scored JSON file."""
    with open(scored_path) as f:
        d = json.load(f)
    scores = d.get("scores", [])
    if isinstance(scores, list):
        return {s["case_id"]: s.get("tier1_pct", 0.0) for s in scores}
    elif isinstance(scores, dict):
        return {k: v.get("tier1_pct", 0.0) for k, v in scores.items()}
    return {}


def process_file(raw_path: Path, scored_path: Path, min_tier1: float) -> list[dict]:
    """Process one (raw, scored) pair. Returns list of training examples."""
    scores = load_scores(scored_path)

    with open(raw_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    examples = []

    for r in results:
        case_id = r.get("case_id", "")
        if r.get("error"):
            continue

        tier1 = scores.get(case_id, 0.0)
        if tier1 < min_tier1:
            continue

        messages = r.get("messages", [])
        if not messages:
            continue

        converted = convert_messages(messages)
        if converted is None:
            continue

        # Basic sanity: must have at least system + user + one assistant
        roles = [m["role"] for m in converted]
        if "system" not in roles or "user" not in roles or "assistant" not in roles:
            continue

        examples.append({
            "messages": converted,
            "_meta": {
                "case_id": case_id,
                "tier1_pct": tier1,
                "source": raw_path.name,
            }
        })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data from benchmark results")
    parser.add_argument("--results-dir", default="results", help="Directory with result JSON files")
    parser.add_argument("--output", default="scripts/peft/train_data.jsonl", help="Output JSONL path")
    parser.add_argument("--min-tier1", type=float, default=90.0, help="Minimum tier1_pct to include (default: 90)")
    parser.add_argument("--file-prefix", default="v17_",
                        help="Prefix in result filenames. v17 results: 'v17_', v23 scaling: '' (default: v17_)")
    parser.add_argument("--models", nargs="+", default=["qwen27b", "qwen35b"],
                        help="Model tags to use as teacher traces (default: qwen27b qwen35b)")
    parser.add_argument("--systems", nargs="+",
                        default=["marta", "taipei", "beijing", "doha", "bart", "cta"],
                        help="Systems to include")
    parser.add_argument("--strip-meta", action="store_true",
                        help="Strip _meta fields from output (for training use)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_examples = []
    stats = {"files_processed": 0, "files_missing": 0, "included": 0, "filtered": 0}

    for model_tag in args.models:
        for system in args.systems:
            raw_path = results_dir / f"{args.file_prefix}{model_tag}_{system}.json"
            scored_path = results_dir / f"{args.file_prefix}{model_tag}_{system}_scored.json"

            if not raw_path.exists() or not scored_path.exists():
                print(f"  MISSING: {raw_path.name}", file=sys.stderr)
                stats["files_missing"] += 1
                continue

            examples = process_file(raw_path, scored_path, args.min_tier1)
            stats["files_processed"] += 1
            stats["included"] += len(examples)
            all_examples.extend(examples)
            print(f"  {raw_path.name}: {len(examples)} examples (tier1>={args.min_tier1:.0f}%)")

    # Deduplicate by case_id (prefer highest tier1)
    seen: dict[str, dict] = {}
    for ex in all_examples:
        cid = ex["_meta"]["case_id"]
        if cid not in seen or ex["_meta"]["tier1_pct"] > seen[cid]["_meta"]["tier1_pct"]:
            seen[cid] = ex
    deduped = list(seen.values())

    # Write JSONL
    written = 0
    with open(output_path, "w") as f:
        for ex in deduped:
            if args.strip_meta:
                ex = {k: v for k, v in ex.items() if k != "_meta"}
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nTotal included: {stats['included']}, after dedup: {written}")
    print(f"Written to: {output_path}")

    # Per-category breakdown
    by_cat: dict[str, int] = {}
    for ex in deduped:
        cat = ex["_meta"]["case_id"].split("-")[1] if "-" in ex["_meta"]["case_id"] else "?"
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print("\nPer-category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  Cat {cat}: {count}")


if __name__ == "__main__":
    main()
