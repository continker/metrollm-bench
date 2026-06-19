#!/usr/bin/env bash
# Mac M-series PEFT bench. One model size per invocation.
#
# What it captures: tier1 / composite (existing scorer) + decode tok/s + TTFT
# + peak RAM + chip/RAM/fanless metadata. Single-system MARTA bench (~150 cases).
#
# Pull artefacts from continker/ HF org. No teacher box / network back to LAN.
#
# Prereqs (one-time per Mac):
#   - macOS 14+ (Apple Silicon)
#   - Homebrew
#   - llama.cpp:  brew install llama.cpp
#   - uv:         brew install uv  (or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
#   - Repo cloned + `uv sync` in the repo root
#   - .env with ANTHROPIC_API_KEY (for Tier 2 judge)
#
# Run:
#   bash scripts/mac_bench/run_bench.sh 2b               # default ctx=32768 (2B may chain long)
#   bash scripts/mac_bench/run_bench.sh 4b               # default ctx=16384
#   bash scripts/mac_bench/run_bench.sh 9b               # default ctx=16384
#   bash scripts/mac_bench/run_bench.sh 9b --ctx 8192    # tighter ctx for low-RAM Macs
#   (skip 27b on Macs <48 GB unified RAM)
#
# Context-size requirements (fp16 KV cache, --parallel 1, see docs in README.md):
#   p99 final-conversation tokens, measured across 8 Qwen3.5 PEFT/base models on MARTA:
#     2B FT: not yet measured (v17 2B PEFT hit 18.8K → 32K default for safety)
#     4B FT: 8.7K  (16K default → 7.3K headroom for next response)
#     9B FT: 7.8K  (16K default → 8.2K headroom)
#     27B FT: 9.6K (16K default → 6.4K headroom)
#   llama.cpp allocates the full KV cache UPFRONT at server start.
#   Reducing ctx-size below the defaults risks "context full" mid-bench failures.

set -u
cd "$(dirname "$0")/../.." || exit 1

# ---- arg parse ----
SIZE=""
CTX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ctx)        CTX="$2"; shift 2 ;;
    --ctx=*)      CTX="${1#--ctx=}"; shift ;;
    -h|--help)
      grep -E '^# (Run:|  bash|  -|  $)' "$0" | sed 's/^# *//'
      exit 0
      ;;
    *)            SIZE="$1"; shift ;;
  esac
done
if [[ -z "$SIZE" ]]; then
  echo "Usage: $0 {2b|4b|9b|27b} [--ctx N]" >&2
  exit 2
fi
case "$SIZE" in
  2b|4b|9b|27b) ;;
  *) echo "Bad size: $SIZE" >&2; exit 2 ;;
esac
SIZE_UP=$(echo "$SIZE" | tr '[:lower:]' '[:upper:]')

# Default ctx-size per model (rounded to powers of 2 covering measured p99 + ~6K headroom).
# Override with --ctx for tight-RAM Macs; below 8192 risks bench failures on long chains.
if [[ -z "$CTX" ]]; then
  case "$SIZE" in
    2b)  CTX=32768 ;;   # 2B may retry more, KV cost is small (~1.2 GB) so 32K is cheap
    4b)  CTX=16384 ;;   # measured max 10.3K, 16K covers comfortably
    9b)  CTX=16384 ;;   # measured max 8.2K
    27b) CTX=16384 ;;   # measured max 11.5K (not run on Mac in default flow)
  esac
fi

REPO_ID="continker/Qwen3.5-${SIZE_UP}-metro-v24"
GGUF_NAME="Qwen3.5-${SIZE_UP}-metro-v24-Q4_K_M.gguf"
LOCAL_GGUF_DIR="data/mac_models"
LOCAL_GGUF="$LOCAL_GGUF_DIR/$GGUF_NAME"

CHIP=$(sysctl -n machdep.cpu.brand_string | sed 's/Apple //; s/ /-/g')
RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1024/1024/1024}')
RUN_TAG="${CHIP}-${RAM_GB}gb-${SIZE}"
OUT_DIR="results/mac_bench/${RUN_TAG}"
mkdir -p "$OUT_DIR"

LLAMA_PORT=8081  # different from box-bench default 8080 to avoid clash
MOCK_PORT=8102   # different from box-bench default 8100 — both mocks may run concurrently on the same Mac
LLAMA_LOG="$OUT_DIR/llama_server.log"
RSS_LOG="$OUT_DIR/llama_rss.log"
MOCK_LOG="$OUT_DIR/mock_server.log"
RAW_RESULTS="$OUT_DIR/marta_raw.json"
SCORED_RESULTS="$OUT_DIR/marta_scored.json"
TELEMETRY_JSON="$OUT_DIR/telemetry.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---- prereq checks ----
if ! command -v llama-server >/dev/null 2>&1; then
  log "ERROR: llama-server not on PATH. brew install llama.cpp"
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  log "ERROR: uv not on PATH. brew install uv"
  exit 1
fi
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  log "Note: 'hf' CLI not found; will use uv-managed huggingface_hub Python lib for download."
fi

# ---- download GGUF if missing ----
mkdir -p "$LOCAL_GGUF_DIR"
if [[ -f "$LOCAL_GGUF" ]]; then
  log "GGUF cached: $LOCAL_GGUF ($(du -h "$LOCAL_GGUF" | awk '{print $1}'))"
else
  log "Downloading $REPO_ID/$GGUF_NAME -> $LOCAL_GGUF"
  uv run --with huggingface_hub python - <<PY
from huggingface_hub import hf_hub_download
import os
os.makedirs("$LOCAL_GGUF_DIR", exist_ok=True)
path = hf_hub_download(
    repo_id="$REPO_ID",
    filename="$GGUF_NAME",
    local_dir="$LOCAL_GGUF_DIR",
)
print("downloaded:", path)
PY
fi

if [[ ! -f "$LOCAL_GGUF" ]]; then
  log "ERROR: download failed"
  exit 1
fi

# ---- kill anything on llama port + mock port ----
kill_port() {
  local port=$1
  local pids
  pids=$(lsof -t -i :${port} -P -n 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
    sleep 1
    pids=$(lsof -t -i :${port} -P -n 2>/dev/null || true)
    [[ -n "$pids" ]] && { kill -9 $pids 2>/dev/null || true; sleep 1; }
  fi
}
kill_port $LLAMA_PORT
kill_port $MOCK_PORT

# ---- estimated RAM check ----
# Rough KV cost (fp16, GQA): 2B = 36 KB/tok, 4B/9B = 144 KB/tok, 27B = 256 KB/tok
case "$SIZE" in
  2b)  KV_PER_TOK_KB=36;  WEIGHTS_GB=1.2 ;;
  4b)  KV_PER_TOK_KB=144; WEIGHTS_GB=2.6 ;;
  9b)  KV_PER_TOK_KB=144; WEIGHTS_GB=5.3 ;;
  27b) KV_PER_TOK_KB=256; WEIGHTS_GB=16.0 ;;
esac
KV_GB=$(awk "BEGIN {printf \"%.2f\", $KV_PER_TOK_KB * $CTX / 1024 / 1024}")
EST_GB=$(awk "BEGIN {printf \"%.1f\", $WEIGHTS_GB + $KV_GB + 1.5}")  # +1.5 for Metal/buffers
log "Mem estimate: weights $WEIGHTS_GB GB + KV@${CTX} $KV_GB GB + overhead 1.5 GB = ${EST_GB} GB total."
log "Available: ${RAM_GB} GB unified. (macOS + apps typically reserve 4-6 GB.)"

# ---- start llama-server ----
log "Starting llama-server on :$LLAMA_PORT (Metal full-offload, parallel=1, ctx=$CTX)"
llama-server \
  --model "$LOCAL_GGUF" \
  --port $LLAMA_PORT \
  --n-gpu-layers 999 \
  --ctx-size "$CTX" \
  --parallel 1 \
  --flash-attn on \
  --alias "${SIZE}-metro-v24" \
  --no-mmap \
  > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
log "llama-server PID=$LLAMA_PID"

# wait for ready
log "Waiting for llama-server health..."
until curl -sf "http://localhost:${LLAMA_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    log "ERROR: llama-server died during startup. Last 30 lines:"
    tail -30 "$LLAMA_LOG"
    exit 1
  fi
  sleep 2
done
log "llama-server ready"

# ---- start RSS sampler (1s cadence) ----
(
  while kill -0 "$LLAMA_PID" 2>/dev/null; do
    rss_kb=$(ps -o rss= -p "$LLAMA_PID" 2>/dev/null | tr -d ' ')
    [[ -n "$rss_kb" ]] && echo "$(date +%s) $rss_kb"
    sleep 1
  done
) > "$RSS_LOG" 2>&1 &
RSS_PID=$!

# ---- start mock_server ----
log "Starting mock_server on :$MOCK_PORT (system=marta)"
uv run python -m harness.mock_server --system marta --port $MOCK_PORT \
  > "$MOCK_LOG" 2>&1 &
MOCK_PID=$!
until curl -sf "http://localhost:${MOCK_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "$MOCK_PID" 2>/dev/null; then
    log "ERROR: mock_server died. Last 30 lines:"
    tail -30 "$MOCK_LOG"
    kill "$LLAMA_PID" "$RSS_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

# ---- run bench ----
RUN_START=$(date +%s)
log "Running runner (MARTA, parallel=1, thinking on)..."
if ! uv run python -m harness.runner \
    --cases "cases/marta_cases.json" --system marta \
    --llm-url "http://localhost:${LLAMA_PORT}/v1" \
    --llm-key "sk-mac-bench" \
    --llm-model "${SIZE}-metro-v24" \
    --thinking --parallel 1 \
    --mock-url "http://localhost:${MOCK_PORT}" \
    --output "$RAW_RESULTS" 2>&1 | tee "$OUT_DIR/runner.log" | tail -5; then
  log "WARN: runner returned non-zero; will still attempt scoring"
fi
RUN_END=$(date +%s)
log "Runner wallclock: $((RUN_END - RUN_START))s"

# ---- shutdown llama + mock + rss in correct order ----
log "Stopping mock_server..."
kill "$MOCK_PID" 2>/dev/null || true
log "Stopping llama-server..."
kill "$LLAMA_PID" 2>/dev/null || true
sleep 2
kill -9 "$LLAMA_PID" 2>/dev/null || true
kill "$RSS_PID" 2>/dev/null || true
wait 2>/dev/null || true

# ---- score (scorer always uses LLM judge; needs ANTHROPIC_API_KEY in .env) ----
if [[ -f "$RAW_RESULTS" ]]; then
  log "Scoring (Claude Haiku judge)..."
  uv run python -m harness.scorer \
    --system marta --results "$RAW_RESULTS" --output "$SCORED_RESULTS" \
    2>&1 | tail -5
else
  log "WARN: no raw results to score"
fi

# ---- parse telemetry ----
log "Parsing telemetry..."
uv run python scripts/mac_bench/parse_telemetry.py \
  --llama-log "$LLAMA_LOG" \
  --rss-log "$RSS_LOG" \
  --raw-results "$RAW_RESULTS" \
  --scored-results "$SCORED_RESULTS" \
  --runner-wallclock $((RUN_END - RUN_START)) \
  --chip "$CHIP" --ram-gb "$RAM_GB" --size "$SIZE" \
  --ctx-size "$CTX" \
  --output "$TELEMETRY_JSON"

log "Done. Output: $OUT_DIR"
log ""
log "Telemetry summary:"
uv run python -c "
import json
t = json.loads(open('$TELEMETRY_JSON').read())
print(f\"  chip:           {t['hardware']['chip']} ({t['hardware']['ram_gb']} GB)\")
print(f\"  model:          {t['model']['size']} ({t['model']['gguf_gb']:.2f} GB GGUF)\")
print(f\"  tier1:          {t['eval'].get('tier1_composite', 'n/a')}\")
print(f\"  composite:      {t['eval'].get('metrollm_composite', 'n/a')}\")
print(f\"  decode tok/s:   {t['perf']['decode_tok_s_median']:.1f} median, {t['perf']['decode_tok_s_p10']:.1f} p10\")
print(f\"  ttft ms:        {t['perf']['ttft_ms_median']:.0f} median\")
print(f\"  peak rss:       {t['perf']['peak_rss_gb']:.2f} GB\")
print(f\"  wallclock:      {t['perf']['runner_wallclock_s']}s\")
"
