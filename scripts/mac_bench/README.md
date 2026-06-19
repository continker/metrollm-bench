# Mac M-series PEFT Bench

Self-contained per-Mac bench package for the metro-v24 PEFT variants. Pulls
GGUF + adapter from `continker/` on HuggingFace, runs `llama-server` locally,
benches one metro system (MARTA, ~150 cases) per model size, captures
deterministic eval metrics + decode tok/s + TTFT + peak RAM.

Output lands in `results/mac_bench/<chip>-<ram>gb-<size>/` per run, then
`scripts/mac_bench/aggregate.py` rolls them up to `<chip>-<ram>gb.json`.

## Per-Mac scope

| Mac | Unified RAM | Sizes to run |
|---|---:|---|
| MacBook Air M2 | 16 GB | 2b, 4b |
| MacBook Pro M4 Pro | 24 GB | 2b, 4b, 9b |
| MacBook Pro M2 Max | 96 GB | 2b, 4b, 9b |

27B is not benched on Macs (16 GB GGUF + decode buffers exceeds practical
headroom on the smaller Macs; M2 Max can run it but the fairness comparison
across the three machines requires the same model set). The 27B+PEFT model
is published separately to `continker/Qwen3.5-27B-metro-v24` on HuggingFace
for users with sufficient hardware (≥48 GB unified RAM or a workstation GPU).

## Context-size requirements

`llama.cpp` allocates the **full KV cache upfront** at server start — context
size is a flat memory tax, not a "max possible" knob. Measured final-conversation
token counts across 8 Qwen3.5 models on MARTA (n ≈ 156 cases each):

| Model | p50 | p99 | max | Default ctx | Reason |
|---|---:|---:|---:|---:|---|
| 2B FT | 3,367 (v17) | 17,652 (v17) | v17 hit 18,824 | **32,768** | v17 PEFT chained long; v23 PEFT distribution similar; 32K is the safest default |
| 4B FT | 3,183 | 8,747 | 10,283 | **16,384** | 6K headroom over max |
| 9B FT | 3,109 | 7,841 | 8,225 | **16,384** | 8K headroom over max |
| 27B FT | 4,069 | 10,327 | 11,512 | **16,384** | not run on Macs by default |

**Memory cost of the KV cache** (fp16, GQA architectures):

| Size | KV / token | KV @ 8K | KV @ 16K | KV @ 32K |
|---|---:|---:|---:|---:|
| 2B | 36 KB | 0.30 GB | 0.60 GB | 1.21 GB |
| 4B / 9B | 144 KB | 1.21 GB | 2.42 GB | 4.83 GB |
| 27B | 256 KB | 2.15 GB | 4.29 GB | 8.59 GB |

**Total RAM (weights + KV @ default ctx + ~1.5 GB Metal/buffers)**:

| Size × Mac | M2 Air 16 GB | M4 Pro 24 GB | M2 Max 96 GB |
|---|---:|---:|---:|
| 2B @ 32K | 4.0 GB ✓ | 4.0 GB ✓ | 4.0 GB ✓ |
| 4B @ 16K | 6.5 GB ✓ | 6.5 GB ✓ | 6.5 GB ✓ |
| 9B @ 16K | 9.2 GB tight | 9.2 GB ✓ | 9.2 GB ✓ |

Override the default with `--ctx N` if you need to fit a tight RAM envelope:

```bash
bash scripts/mac_bench/run_bench.sh 9b --ctx 8192   # KV halves to 1.21 GB
```

Going below 8K risks "context full" mid-bench failures on long Cat K compound
cases (which can chain 6+ tool calls).

## One-time setup per Mac

```bash
# 1. Install runtime deps (Apple Silicon Homebrew)
brew install llama.cpp uv

# 2. Clone the bench repo
git clone <repo-url> ~/metrollm-bench
cd ~/metrollm-bench
uv sync

# 3. Provide an Anthropic key for the Tier-2 judge (optional — without it,
#    Tier-2 components fall back to keyword scoring)
cp .env.example .env
# edit .env: ANTHROPIC_API_KEY=sk-ant-...
```

The first `run_bench.sh` invocation downloads the GGUF (~1–6 GB depending
on size) into `data/mac_models/` and caches it for subsequent runs.

## Running

One model size at a time. Sequential is fine — each run takes between
~30 minutes (2B) and ~3 hours (9B) on M-series.

```bash
bash scripts/mac_bench/run_bench.sh 2b
bash scripts/mac_bench/run_bench.sh 4b
bash scripts/mac_bench/run_bench.sh 9b
```

After all sizes complete:

```bash
# detect chip + ram for the aggregate filename
CHIP=$(sysctl -n machdep.cpu.brand_string | sed 's/Apple //; s/ /-/g')
RAM=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1024/1024/1024}')
uv run python scripts/mac_bench/aggregate.py --chip "$CHIP" --ram-gb "$RAM"
```

This emits `results/mac_bench/<chip>-<ram>gb.json` plus a human-readable
table.

## What gets captured

Per (Mac × size):

- `tier1_composite` and `metrollm_composite` — same scorer used in the paper
- `decode_tok_s_median` / `_p10` / `_p90` — single-stream decode throughput
- `ttft_ms_median` / `_p90` — first-token latency end-to-end (HTTP + decode)
- `peak_rss_gb` — max RSS of the `llama-server` process during decode
- `runner_wallclock_s` — wall time for the full MARTA bench
- `gguf_gb` — on-disk model file size

## Three bench modes

| Script | Purpose | Wallclock | Run on |
|---|---|---|---|
| `run_bench.sh <size>` | Full 156-case accuracy bench. Published-code replication target. | 1-8 h | one Mac (typically the fastest) |
| `run_probe.sh <size>` | 15-case stratified probe — TTFT, decode tok/s, peak RAM, T1 spot-check. | 15-30 min | every Mac in the comparison |
| `run_thermal.sh <size> --duration 45m` | Sustained realistic-workload run with 30 s tok/s + RSS sampling. Captures cold → sustained → throttle curve. | configurable | **fanless silicon only** (M-base Air / M mini base) — fan-cooled Macs produce flat curves |

`run_bench.sh` is for the published artefact (others can replicate exactly).
`run_probe.sh` and `run_thermal.sh` are the paper-§6 hardware-envelope tools.

## Files

- `run_bench.sh` — full 156-case accuracy bench
- `run_probe.sh` — 15-case stratified probe (cross-Mac comparison table)
- `run_thermal.sh` — N-minute sustained-load runner with parallel sampler
- `thermal_sampler.py` — polls llama-server log + RSS log every 30 s, writes thermal_curve.csv/json
- `parse_telemetry.py` — turns llama-server log + RSS samples into telemetry.json (used by run_bench / run_probe)
- `aggregate.py` — rolls up per-size telemetry into one Mac-level report
- `README.md` — this file

## Why no 27B on Macs

The 27B GGUF is 16 GB on disk; with decode KV buffers it consumes 18–22 GB
of unified memory under load. M2 Air and M4 Pro at 16/24 GB will swap or
OOM. M2 Max at 96 GB can run it cleanly but a same-model-set comparison
across the Mac tier requires omitting it. The 27B variant is published to
`continker/Qwen3.5-27B-metro-v24` for users with sufficient hardware.
