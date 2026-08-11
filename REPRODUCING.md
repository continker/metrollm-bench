# Reproducing MetroLLM-Bench

This is the step-by-step reproduction path for the paper's results: evaluating models on the benchmark, the rule-based baseline, the train/held-out partition, the PEFT distillation pipeline, the partition-filtered statistics, and the Apple Silicon measurements. Every command below uses an entry point that ships in this repository.

The paper's local runs used a single NVIDIA RTX 5090 (32 GB VRAM, Ubuntu 24.04); any CUDA or Apple Silicon machine with enough memory for the chosen model works. API keys are needed only for hosted models (Azure OpenAI, Mistral) and for the Tier-2 judge (`ANTHROPIC_API_KEY`, see `.env.example`). Deterministic Tier-1 scoring needs no keys.

## 1. Setup

```bash
git clone https://github.com/continker/metrollm-bench && cd metrollm-bench
uv sync
# for local models:
brew install llama.cpp        # macOS; on Linux, build llama.cpp with CUDA
```

## 2. Evaluate a model

Three processes: the mock tool server, the runner, the scorer.

```bash
# terminal 1: serve a model (example: the 4B PEFT student)
uv run --with huggingface_hub huggingface-cli download \
  continker/Qwen3.5-4B-metro-v24 Qwen3.5-4B-metro-v24-Q4_K_M.gguf --local-dir models
llama-server --model models/Qwen3.5-4B-metro-v24-Q4_K_M.gguf \
  --port 8080 --ctx-size 16384 -ngl 999 --alias metro-v24

# terminal 2: tool server for one system
uv run python -m harness.mock_server --system marta --port 8100

# terminal 3: run + score
uv run python -m harness.runner \
  --cases cases/marta_cases.json --system marta \
  --llm-url http://localhost:8080/v1 --llm-model metro-v24 \
  --output results/run_marta.json
uv run python -m harness.scorer --system marta \
  --results results/run_marta.json --output results/run_marta_scored.json --no-judge
```

Repeat per system (`marta doha bart taipei cta beijing`). Drop `--no-judge` to add the six judged Tier-2 rubrics (needs `ANTHROPIC_API_KEY`; judgments are disk-cached). For hosted models point `--llm-url` at the provider's OpenAI-compatible endpoint; the runner handles the family-specific settings from the paper (temperature 0.0 locally and for Mistral; the GPT-5 family requires temperature 1.0, with `reasoning_effort` passed through).

## 3. Rule-based baseline

```bash
uv run python -m harness.rule_agent \
  --cases cases/marta_cases.json --system marta \
  --mock-url http://localhost:8100 --output results/rule_marta.json
uv run python -m harness.scorer --system marta \
  --results results/rule_marta.json --output results/rule_marta_scored.json
```

## 4. Train/held-out partition

The canonical split ships at `data/splits/v23_holdout75_seed42.json` (717 train / 238 held-out, system-stratified, seed=42), together with pre-sliced per-system case files (`cases/{system}_cases_train_split75.json` and `..._holdout_split75.json`). To regenerate both from scratch:

```bash
uv run python scripts/build_holdout_split.py
uv run python scripts/slice_cases_by_split.py --split data/splits/v23_holdout75_seed42.json
```

## 5. PEFT pipeline

**5a. Teacher traces** — run the two teachers (base Qwen 3.5 27B and 35B-A3B, served as in step 2) over the six `*_train_split75.json` case files only, and score the outputs. No held-out case may enter the teacher pool.

**5b. Training set** — filter to traces with Tier-1 ≥ 90, deduplicate by case keeping the higher-scoring teacher:

```bash
uv run python scripts/peft/prepare_data.py \
  --models 27b 35b --min-tier1 90 --strip-meta \
  --output scripts/peft/train_data.jsonl
# see --help for --results-dir / --file-prefix conventions matching your scored filenames
```

This reproduces the paper's 600-example set (540 from 27B, 60 from 35B).

**5c. Train** — one run per size per seed (the paper uses seeds 42 and 43):

```bash
uv run python scripts/peft/train.py \
  --model <hf-base-id, e.g. Qwen/Qwen3.5-4B> \
  --data scripts/peft/train_data.jsonl \
  --qlora --rank 16 --epochs 3 --seed 42 \
  --max-seq-len 4096 \        # 2048 at 27B (32 GB VRAM ceiling)
  --output lora-4b-s42
```

Training stack used for the paper: Unsloth 2026.4.2, PyTorch 2.10, peft 0.18.1, trl 0.24. Per-seed wallclock on the RTX 5090: 27 min (2B), 63 min (4B), 74 min (9B), 103 min (27B).

**5d. Merge + quantise**:

```bash
uv run python scripts/peft/merge_adapter.py --adapter lora-4b-s42 ...
uv run python scripts/peft/export_gguf.py ...   # convert_hf_to_gguf + llama-quantize to Q4_K_M
```

Note: at 27B use `scripts/peft/merge_adapter_cpu.py`. The GPU merge path hits an `accelerate` version-mismatch bug once the model overflows VRAM and CPU offload engages; the CPU fallback avoids it.

**5e. Bench the students** — step 2, pointed at each student GGUF, over all six full case files.

## 6. Partition metrics and statistics

```bash
# held-out / full-matrix aggregation of scored files
uv run python scripts/score_split.py --results-dir results/ --partition holdout

# bootstrap CIs + the four paired headline comparisons (5,000 resamples)
uv run python scripts/compute_bootstrap_heldout.py --partition holdout
uv run python scripts/compute_bootstrap_heldout.py --partition full

# per-category held-out matrix (paper Figure 3)
uv run python scripts/heldout_percategory.py
```

## 7. Apple Silicon envelope

The paper's exploratory Mac measurements (Appendix B.6) use two scripts:

```bash
bash scripts/mac_bench/run_probe.sh 2b                    # ~30 min: 15 stratified MARTA cases
bash scripts/mac_bench/run_thermal.sh 2b --duration 45m   # cold / sustained / throttle curve
```

- **Probe**: 15 cases (one per category, weighted toward multi-round C/K) via `llama-server` with full Metal offload, `parallel=1`, model-specific context size. Records Tier-1, decode tok/s, TTFT, peak RSS.
- **Thermal**: replays the full MARTA set for 45 minutes, sampling the server log and RSS every 30 s. Only informative on fanless hardware; fan-cooled Macs give a flat curve.

Measured for the paper (lid open, on AC — the most favourable realistic state; clamshell or battery throttles 10–15% harder):

| Hardware | Model | Decode | Basis |
|---|---|---:|---|
| MacBook Air M2 (16 GB, fanless) | 2B Q4_K_M | 39 tok/s sustained | measured |
| MacBook Pro M2 Max (96 GB) | 2B Q4_K_M | 108 tok/s | measured |
| MacBook Air M2 | 4B Q4_K_M | ~18 tok/s | bandwidth-projected, not measured (swap-pressure risk at 16 GB) |

The standalone Mac package is [`continker/metrollm-bench-mac`](https://huggingface.co/continker/metrollm-bench-mac) on Hugging Face.

## A note on the paper's batch drivers

The overnight batch wrappers used to sequence the paper's runs (per-system loops around the commands above, plus Azure/Mistral sweep scripts) are environment-specific and not included. They contain no logic beyond looping the entry points documented here.
