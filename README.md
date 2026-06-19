# MetroLLM-Bench

**A transit kiosk should run its own intelligence. Put a small language model in a box the operator owns, keep it off the cloud. Adjust it in plain language, not code: MetroLLM-Bench tests whether that is real yet.**

It is a 955-case benchmark across six metro systems. It asks whether a language model can run a transit kiosk from a prose prompt, handling routing, fares, and disruptions through tool calls, with no code changes. The headline finding is a 2.6 GB fine-tuned model that runs offline on commodity hardware and matches a frontier cloud API on the task.

[![Demo](https://img.shields.io/badge/Demo-Live-FF4B4B?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/remcohendriks/metrollm)
[![Models](https://img.shields.io/badge/Models-continker-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/continker)
[![Research notes](https://img.shields.io/badge/Research-Continker-1f6feb)](https://continker.ai/research)
[![Paper](https://img.shields.io/badge/Paper-under%20review-B31B1B)](#citation)
[![License](https://img.shields.io/badge/License-Apache%202.0-3DA639?logo=apache&logoColor=white)](LICENSE)

This is the case for sovereignty. The policy lives in prose the operator writes. The model lives in a box the operator owns. Neither has to reach the cloud. For public infrastructure that cannot send rider data to someone else's servers, a kiosk that needs no network is not a limitation, it is the point.

## Quickstart

Three ways in, from zero effort to full reproduction.

### 1. See it work (no install)

Open the [live demo](https://huggingface.co/spaces/remcohendriks/metrollm). It runs the fine-tuned 9B student in your browser. Pick a system and a scenario, then watch the model handle a disruption and produce a kiosk ticket.

### 2. Reproduce it locally (macOS, Linux, or Windows, no API keys)

The headline claim is that a small fine-tuned model does this on commodity hardware. Serve a GGUF with [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` and point the harness at it. It runs cross-platform and picks the Metal, CUDA, or CPU backend for you. No accounts needed.

```bash
git clone https://github.com/continker/metrollm-bench && cd metrollm-bench

# 1. fetch the fine-tuned student (2.6 GB) and serve it (OpenAI-compatible API)
uv run --with huggingface_hub huggingface-cli download \
  continker/Qwen3.5-4B-metro-v24 Qwen3.5-4B-metro-v24-Q4_K_M.gguf --local-dir models
llama-server --model models/Qwen3.5-4B-metro-v24-Q4_K_M.gguf \
  --port 8080 --ctx-size 16384 -ngl 999 --alias metro-v24 &

# 2. tool server plus a few cases, scored Tier-1 (no API key)
uv run python -m harness.mock_server --system marta --port 8100 &
uv run python -m harness.runner \
  --cases cases/marta_cases.json --system marta \
  --llm-url http://localhost:8080/v1 --llm-model metro-v24 \
  --case-ids MARTA-A-001,MARTA-C-001,MARTA-K-001 --output results/run.json
uv run python -m harness.scorer \
  --system marta --results results/run.json --output results/scored.json --no-judge
```

`--no-judge` keeps it key-free. You get deterministic Tier-1, which covers route, fare, and tool-call correctness. On an M2 Max the 15-case probe runs at about 62 tok/s and 7 GB of RAM, with clean multi-tool calls on the compound-stress cases and Tier-1 in the low 90s. The same works on a Linux or Windows box with a GPU. It also runs on CPU alone, just slower.

**Choosing a model.** Every size ships as a Q4_K_M GGUF. Pick by GPU VRAM, or by unified memory on a Mac.

| Model | size | runs on |
|---|---|---|
| 2B | 1.2 GB | anything, including CPU-only |
| **4B** | 2.6 GB | **8 GB VRAM / 16 GB Mac, the validated default** |
| 9B | 5.3 GB | 12 GB VRAM / 24 GB Mac |
| 27B | 16 GB | 24 GB+ VRAM / 32 GB+ Mac (the card flags a regression at 27B) |

Prefer one command? Ollama also serves these. Run `ollama run hf.co/continker/Qwen3.5-4B-metro-v24` and point the harness at the OpenAI API on `:11434`. Its macOS app needs a first-run launch. To reproduce the Apple-Silicon throughput and thermal envelope from the paper, the `scripts/mac_bench/` package has `run_probe.sh` and `run_thermal.sh`.

### 3. Run the benchmark yourself

The same commands run any model through the full suite. Point `--llm-url` at any OpenAI-compatible endpoint, whether a local llama-server or a hosted API like OpenAI, Azure, or Mistral (see [`docs/usage.md`](docs/usage.md)). Drop `--case-ids` to run all 156 MARTA cases. Then pick a scoring tier.

```bash
# Tier-1 only, deterministic, no API key
uv run python -m harness.scorer --system marta \
  --results results/run.json --output results/scored.json --no-judge

# Full 22-component composite, adds the Tier-2 Haiku judge (needs ANTHROPIC_API_KEY)
uv run python -m harness.scorer --system marta \
  --results results/run.json --output results/scored.json
```

Tier-1 is the key-free signal that carries the comparative load, covering route, fare, and tool correctness. Tier-2 adds eight semantic components scored by an LLM judge. They need an `ANTHROPIC_API_KEY` (see [`.env.example`](.env.example)), and they are the part most people can skip.

```bash
# Regenerate a system's cases from ground truth
uv run python -m cases.generator --system marta

# Run the test suite (no network, no keys)
uv run pytest tests/ -q
```

## What it covers

Six real metro systems span three fare models and a 37-to-414 station range.

| System | City | Stations | Fare model | Cases |
|--------|------|---------:|------------|------:|
| MARTA | Atlanta | 38 | flat | 156 |
| Doha Metro | Doha | 37 | flat | 156 |
| BART | San Francisco | 50 | distance | 157 |
| Taipei MRT | Taipei | 107 | distance | 167 |
| CTA | Chicago | 142 | flat with exceptions | 157 |
| Beijing Subway | Beijing | 414 | distance + airport flat | 162 |

Each system carries a framebook that sets its terminology, currency, operating hours, and cultural conventions, injected as the system prompt at runtime. The 955 cases span eleven categories: A Routing, B Fare, C Disruption, D Accessibility, E Cultural, F Policy, G Multi-turn, H Adversarial, I Temporal, J Tool-Hallucination, K Compound Stress.

The model works through a mock tool server. It can plan routes, calculate fares, look up station and line info, read a disruption feed, query a policy knowledge base, and submit a terminal kiosk state. Every case ends in a structured state that the scorer checks against ground truth.

## Scoring

Twenty-two components split into two tiers.

- **Tier 1 (14 components, deterministic)**: route correctness, fare arithmetic and breakdown, tool-call correctness, hallucination resistance, outcome and purchase-gate correctness, disruption detection, advisory issuance, context tracking, re-planning efficiency, and a cultural-accuracy keyword check. This tier is clean enough to use as a fine-tuning reward signal.
- **Tier 2 (8 components, semantic)**: framebook conformance, advisory content, policy acknowledgement, temporal accuracy, safety-response quality, data fabrication, accessibility accuracy, scope adherence. Scored by a language model judge (Claude Haiku).

The composite combines both tiers. The judge was calibrated against a 100-case human study. It reaches 82% exact agreement and 97% within one point, with a Cohen's κ of 0.53 (moderate on the Landis and Koch scale). The high raw agreement next to a moderate κ is the imbalanced-class paradox, since most rubric cases score 2/2. Comparative claims here lean on the deterministic Tier-1 and composite signals, not on Tier-2 distinctions alone.

## Headline results

A stratified 75/25 case-level partition (seed=42) governs the fine-tuning evaluation. 717 cases generate training data, and 238 are strictly held out for reporting. The split spec ships in [`data/splits/`](data/splits/).

On the 238-case held-out partition, six models cluster within 1.5 composite points, from 89.1 to 90.6. They are Qwen 27B base (90.60), GPT-5.4 full at maximum reasoning effort (90.45), Qwen 35B-A3B base (89.90), and the 27B, 4B, and 9B PEFT students (89.72, 89.12, 88.85). A 4B distilled student in 2.6 GB of Q4_K_M lands within 0.05 Tier-1 points of GPT-5.4 full at maximum effort, and it beats that model at standard effort. A 2.6 GB model is small enough to run in the kiosk and own outright, which is what makes the offline, sovereign deployment viable. A rule-based deterministic baseline reaches 84.6 Tier-1. The language model's advantage concentrates in temporal reasoning, policy adaptation, and disruption-advisory composition.

The PEFT-versus-base delta decays monotonically with base capability and turns negative at 27B (held-out Tier-1, mean of two training seeds):

| Size | Base | seed=42 | seed=43 | Mean | Δ vs base | seed spread | GGUF |
|------|-----:|--------:|--------:|-----:|----------:|------------:|-----:|
| 2B  | 74.17 | 76.80 | 82.07 | 79.43 | **+5.26** | ±2.63 | 1.2 GB |
| 4B  | 89.32 | 91.82 | 90.83 | 91.32 | **+2.00** | ±0.49 | 2.6 GB |
| 9B  | 89.38 | 90.53 | 91.53 | 91.03 | **+1.65** | ±0.50 | 5.3 GB |
| 27B | 92.32 | 91.93 | 90.88 | 91.41 | **−0.91** | ±0.53 | 16 GB |

At n=238 the paired-bootstrap CIs include zero. The full 955-case matrix has the power to certify the 4B gain (+1.72, CI [+0.72, +2.74]) and the 27B regression (−1.09, CI [−1.82, −0.38]). The reading is that base competence and task ceiling leave room for an adapter to help at small scale, and that room closes as the base model grows.

## Open weights

The four distilled students are public under [`continker/`](https://huggingface.co/continker) on HuggingFace. Each repo carries a Q4_K_M GGUF, the LoRA adapter, and a model card.

- `continker/Qwen3.5-2B-metro-v24` (1.2 GB)
- `continker/Qwen3.5-4B-metro-v24` (2.6 GB, best size and quality tradeoff)
- `continker/Qwen3.5-9B-metro-v24` (5.3 GB)
- `continker/Qwen3.5-27B-metro-v24` (16 GB, the card flags the base-vs-PEFT regression at this size)

An Apple Silicon replication package is at [`continker/metrollm-bench-mac`](https://huggingface.co/continker).

## Layout

```
harness/      tool server, runner, scorer, judge, graph + fare engines, rule-baseline agent
cases/        case generator and the per-system case files (+ the 75/25 split slices)
data/systems/ the six metro systems (framebook, fares, stations, lines, events)
data/splits/  the pre-registered held-out partition spec (seed=42)
scripts/      PEFT training + GGUF export, Mac-bench tooling, analysis scripts
dashboard/    results dashboard, calibration annotator, kiosk simulator (static front-ends)
tests/        ~800 fast tests, no network
docs/         spec, decisions, references, usage
```

## Reproducing the results

The held-out partition is fixed in `data/splits/v23_holdout75_seed42.json`. Training data is regenerated from the 717 training cases via `scripts/peft/prepare_data.py`. `scripts/peft/train.py` runs the QLoRA fit (rank 16, three epochs), and `scripts/peft/export_gguf.py` produces the served GGUF. Each model is run with `harness.runner` and scored with `harness.scorer` across the six systems. `scripts/score_split.py` then filters metrics to the held-out partition, and the bootstrap analyses live in `scripts/compute_heldout.py`, `scripts/heldout_percategory.py`, and `scripts/compute_bootstrap_heldout.py`.

## Citation

A paper describing the benchmark is under review. This section will be updated with the preprint citation once it is available. For now, cite the repository.

```bibtex
@misc{hendriks2026metrollm,
  title  = {MetroLLM-Bench: Evaluating Language Models as Transit Kiosk Runtimes},
  author = {Hendriks, Remco},
  year   = {2026},
  note   = {https://github.com/continker/metrollm-bench}
}
```

## License

The code in this repository is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE). The benchmark cases, ground truth, and framebooks are released under the same terms.

The station, line, and coordinate data under `data/` is derived from public transit-agency GTFS feeds and from Wikipedia. Wikipedia-derived content is CC BY-SA 4.0 by its authors and is redistributed here under CC BY-SA 4.0 (not Apache-2.0), with attribution. See [`NOTICE`](NOTICE) for the data sources.
