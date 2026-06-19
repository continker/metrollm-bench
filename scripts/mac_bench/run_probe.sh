#!/usr/bin/env bash
# Mac M-series PEFT PROBE. Short bench (~15 cases, stratified across all 11
# MetroLLM-Bench categories) for cross-Mac comparison of TTFT + tok/s + RAM
# without paying the 156-case wallclock.
#
# Captures the same telemetry shape as run_bench.sh, just with N small enough
# that running on M2 Air / M4 Pro / M2 Max each takes 15-30 min.
#
# Run:
#   bash scripts/mac_bench/run_probe.sh 2b           # 15 stratified MARTA cases
#   bash scripts/mac_bench/run_probe.sh 4b --ctx 16384
#
# Scoring defaults to Tier-1 only (deterministic route/fare/tool correctness,
# NO API key needed). Pass --judge to also run the Tier-2 LLM judge for the full
# composite score (needs ANTHROPIC_API_KEY).
#
# Output: results/mac_bench/<chip>-<ram>gb-<size>-probe/

set -u
cd "$(dirname "$0")/../.." || exit 1

# 15 stratified case IDs covering all 11 MetroLLM-Bench categories on MARTA.
# Picked to give 1-2 cases per category, biased toward C/K (most diagnostic).
PROBE_CASE_IDS="MARTA-A-001,MARTA-A-005,MARTA-B-001,MARTA-C-001,MARTA-C-005,MARTA-D-001,MARTA-E-001,MARTA-F-001,MARTA-G-001,MARTA-H-001,MARTA-I-001,MARTA-J-001,MARTA-K-001,MARTA-K-002,MARTA-K-003"

# ---- arg parse ----
SIZE=""
CTX=""
JUDGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ctx)        CTX="$2"; shift 2 ;;
    --ctx=*)      CTX="${1#--ctx=}"; shift ;;
    --judge)      JUDGE=1; shift ;;
    -h|--help)    grep -E '^# ' "$0" | sed 's/^# *//'; exit 0 ;;
    *)            SIZE="$1"; shift ;;
  esac
done
[[ -z "$SIZE" ]] && { echo "Usage: $0 {2b|4b|9b|27b} [--ctx N]" >&2; exit 2; }
case "$SIZE" in 2b|4b|9b|27b) ;; *) echo "Bad size" >&2; exit 2 ;; esac
SIZE_UP=$(echo "$SIZE" | tr '[:lower:]' '[:upper:]')

if [[ -z "$CTX" ]]; then
  case "$SIZE" in
    2b)  CTX=32768 ;;
    4b)  CTX=16384 ;;
    9b)  CTX=16384 ;;
    27b) CTX=16384 ;;
  esac
fi

REPO_ID="continker/Qwen3.5-${SIZE_UP}-metro-v24"
GGUF_NAME="Qwen3.5-${SIZE_UP}-metro-v24-Q4_K_M.gguf"
LOCAL_GGUF_DIR="data/mac_models"
LOCAL_GGUF="$LOCAL_GGUF_DIR/$GGUF_NAME"

CHIP=$(sysctl -n machdep.cpu.brand_string | sed 's/Apple //; s/ /-/g')
RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1024/1024/1024}')
RUN_TAG="${CHIP}-${RAM_GB}gb-${SIZE}-probe"
OUT_DIR="results/mac_bench/${RUN_TAG}"
mkdir -p "$OUT_DIR"

LLAMA_PORT=8081
MOCK_PORT=8102
LLAMA_LOG="$OUT_DIR/llama_server.log"
RSS_LOG="$OUT_DIR/llama_rss.log"
MOCK_LOG="$OUT_DIR/mock_server.log"
RAW_RESULTS="$OUT_DIR/marta_raw.json"
SCORED_RESULTS="$OUT_DIR/marta_scored.json"
TELEMETRY_JSON="$OUT_DIR/telemetry.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

command -v llama-server >/dev/null 2>&1 || { log "ERROR: brew install llama.cpp"; exit 1; }
command -v uv >/dev/null 2>&1 || { log "ERROR: brew install uv"; exit 1; }

# ---- download GGUF ----
mkdir -p "$LOCAL_GGUF_DIR"
if [[ -f "$LOCAL_GGUF" ]]; then
  log "GGUF cached: $LOCAL_GGUF ($(du -h "$LOCAL_GGUF" | awk '{print $1}'))"
else
  log "Downloading $REPO_ID/$GGUF_NAME"
  uv run --with huggingface_hub python - <<PY
from huggingface_hub import hf_hub_download
import os; os.makedirs("$LOCAL_GGUF_DIR", exist_ok=True)
hf_hub_download(repo_id="$REPO_ID", filename="$GGUF_NAME", local_dir="$LOCAL_GGUF_DIR")
PY
fi
[[ -f "$LOCAL_GGUF" ]] || { log "ERROR: download failed"; exit 1; }

# ---- kill stale processes ----
kill_port() {
  local pids; pids=$(lsof -t -i :"$1" -P -n 2>/dev/null || true)
  [[ -n "$pids" ]] && { kill $pids 2>/dev/null || true; sleep 1; kill -9 $pids 2>/dev/null || true; sleep 1; }
}
kill_port $LLAMA_PORT
kill_port $MOCK_PORT

# ---- start llama-server ----
log "Starting llama-server :$LLAMA_PORT (Metal, parallel=1, ctx=$CTX)"
llama-server --model "$LOCAL_GGUF" --port $LLAMA_PORT --n-gpu-layers 999 \
  --ctx-size "$CTX" --parallel 1 --flash-attn on --no-mmap \
  --alias "${SIZE}-metro-v24" > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
until curl -sf "http://localhost:${LLAMA_PORT}/health" >/dev/null 2>&1; do
  kill -0 "$LLAMA_PID" 2>/dev/null || { log "ERROR: llama-server died"; tail -30 "$LLAMA_LOG"; exit 1; }
  sleep 2
done
log "llama-server ready (PID=$LLAMA_PID)"

# ---- RSS sampler ----
( while kill -0 "$LLAMA_PID" 2>/dev/null; do
    rss=$(ps -o rss= -p "$LLAMA_PID" 2>/dev/null | tr -d ' ')
    [[ -n "$rss" ]] && echo "$(date +%s) $rss"
    sleep 1
  done ) > "$RSS_LOG" 2>&1 &
RSS_PID=$!

# ---- start mock_server ----
uv run python -m harness.mock_server --system marta --port $MOCK_PORT > "$MOCK_LOG" 2>&1 &
MOCK_PID=$!
until curl -sf "http://localhost:${MOCK_PORT}/health" >/dev/null 2>&1; do
  kill -0 "$MOCK_PID" 2>/dev/null || { log "ERROR: mock died"; tail -30 "$MOCK_LOG"; kill $LLAMA_PID $RSS_PID 2>/dev/null; exit 1; }
  sleep 1
done

# ---- run probe ----
log "Running probe (15 stratified cases): $PROBE_CASE_IDS"
RUN_START=$(date +%s)
uv run python -m harness.runner \
  --cases cases/marta_cases.json --system marta \
  --llm-url "http://localhost:${LLAMA_PORT}/v1" --llm-key sk-mac-bench \
  --llm-model "${SIZE}-metro-v24" \
  --case-ids "$PROBE_CASE_IDS" \
  --thinking --parallel 1 \
  --mock-url "http://localhost:${MOCK_PORT}" \
  --output "$RAW_RESULTS" 2>&1 | tee "$OUT_DIR/runner.log" | tail -5
RUN_END=$(date +%s)
log "Runner wallclock: $((RUN_END - RUN_START))s"

# ---- shutdown ----
kill "$MOCK_PID" 2>/dev/null || true
kill "$LLAMA_PID" 2>/dev/null || true
sleep 2
kill -9 "$LLAMA_PID" 2>/dev/null || true
kill "$RSS_PID" 2>/dev/null || true
wait 2>/dev/null || true

# ---- score (Tier-1 by default; --judge adds Tier-2 and needs ANTHROPIC_API_KEY) ----
SCORE_FLAGS=""
[[ "$JUDGE" == "1" ]] || SCORE_FLAGS="--no-judge"
[[ "$JUDGE" == "1" ]] && log "Scoring with Tier-2 judge (ANTHROPIC_API_KEY required)" || log "Scoring Tier-1 only (no API key; pass --judge for the full composite)"
[[ -f "$RAW_RESULTS" ]] && uv run python -m harness.scorer \
  --system marta --results "$RAW_RESULTS" --output "$SCORED_RESULTS" $SCORE_FLAGS 2>&1 | tail -3

# ---- telemetry ----
uv run python scripts/mac_bench/parse_telemetry.py \
  --llama-log "$LLAMA_LOG" --rss-log "$RSS_LOG" \
  --raw-results "$RAW_RESULTS" --scored-results "$SCORED_RESULTS" \
  --runner-wallclock $((RUN_END - RUN_START)) \
  --chip "$CHIP" --ram-gb "$RAM_GB" --size "$SIZE" --ctx-size "$CTX" \
  --output "$TELEMETRY_JSON"

log "Done. Output: $OUT_DIR"
uv run python -c "
import json
t = json.loads(open('$TELEMETRY_JSON').read())
print(f\"  chip:           {t['hardware']['chip']} ({t['hardware']['ram_gb']} GB)\")
print(f\"  model:          {t['model']['size']} ctx={t['model']['ctx_size']}\")
print(f\"  tier1:          {t['eval'].get('tier1_composite', 'n/a')}  (n={t['eval'].get('n_cases', 'n/a')})\")
print(f\"  decode tok/s:   {t['perf']['decode_tok_s_median']:.1f} median, {t['perf']['decode_tok_s_p10']:.1f} p10\")
print(f\"  ttft ms:        {t['perf']['ttft_ms_median']:.0f} median\")
print(f\"  peak rss:       {t['perf']['peak_rss_gb']:.2f} GB\")
print(f\"  wallclock:      {t['perf']['runner_wallclock_s']}s\")
"
