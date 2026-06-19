#!/usr/bin/env bash
# Mac M-series PEFT THERMAL/SUSTAINED-LOAD bench.
#
# Replays MARTA cases on a loop for N minutes against a local llama-server,
# while a parallel sampler records tok/s + RSS every 30 s. Captures the
# cold-start → sustained → throttle curve under a realistic kiosk-dialogue
# workload (multi-round tool-using cases, not synthetic 1024-token streams).
#
# **Run only on fanless / passively-cooled silicon.** On fan-cooled Macs
# (M2 Pro, M2 Max, M3/M4 Pro/Max) the curve is flat — use run_probe.sh
# instead for cross-Mac comparison.
#
# Run:
#   bash scripts/mac_bench/run_thermal.sh 2b                     # default 45 min
#   bash scripts/mac_bench/run_thermal.sh 2b --duration 30m
#   bash scripts/mac_bench/run_thermal.sh 4b --duration 60m --ctx 16384
#
# Output: results/mac_bench/<chip>-<ram>gb-<size>-thermal/
#   - thermal_curve.csv   (one row per 30 s window)
#   - thermal_curve.json  (full samples + cold/sustained/throttle summary)
#   - llama_server.log
#   - llama_rss.log
#   - mock_server.log

set -u
cd "$(dirname "$0")/../.." || exit 1

# ---- arg parse ----
SIZE=""
CTX=""
DURATION_RAW="45m"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ctx)         CTX="$2"; shift 2 ;;
    --ctx=*)       CTX="${1#--ctx=}"; shift ;;
    --duration)    DURATION_RAW="$2"; shift 2 ;;
    --duration=*)  DURATION_RAW="${1#--duration=}"; shift ;;
    -h|--help)     grep -E '^# ' "$0" | sed 's/^# *//'; exit 0 ;;
    *)             SIZE="$1"; shift ;;
  esac
done
[[ -z "$SIZE" ]] && { echo "Usage: $0 {2b|4b|9b|27b} [--ctx N] [--duration 45m]" >&2; exit 2; }
case "$SIZE" in 2b|4b|9b|27b) ;; *) echo "Bad size" >&2; exit 2 ;; esac
SIZE_UP=$(echo "$SIZE" | tr '[:lower:]' '[:upper:]')

# Parse duration: accept "45m", "30m", "1h", "1800s", or bare seconds.
case "$DURATION_RAW" in
  *m) DURATION_SEC=$(( ${DURATION_RAW%m} * 60 )) ;;
  *h) DURATION_SEC=$(( ${DURATION_RAW%h} * 3600 )) ;;
  *s) DURATION_SEC=${DURATION_RAW%s} ;;
  *)  DURATION_SEC=$DURATION_RAW ;;
esac

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
RUN_TAG="${CHIP}-${RAM_GB}gb-${SIZE}-thermal"
OUT_DIR="results/mac_bench/${RUN_TAG}"
mkdir -p "$OUT_DIR"

LLAMA_PORT=8081
MOCK_PORT=8102
LLAMA_LOG="$OUT_DIR/llama_server.log"
RSS_LOG="$OUT_DIR/llama_rss.log"
MOCK_LOG="$OUT_DIR/mock_server.log"
RAW_RESULTS="$OUT_DIR/marta_thermal_raw.json"
CURVE_CSV="$OUT_DIR/thermal_curve.csv"
CURVE_JSON="$OUT_DIR/thermal_curve.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

command -v llama-server >/dev/null 2>&1 || { log "ERROR: brew install llama.cpp"; exit 1; }
command -v uv >/dev/null 2>&1 || { log "ERROR: brew install uv"; exit 1; }

# ---- download GGUF ----
mkdir -p "$LOCAL_GGUF_DIR"
if [[ ! -f "$LOCAL_GGUF" ]]; then
  log "Downloading $REPO_ID/$GGUF_NAME"
  uv run --with huggingface_hub python - <<PY
from huggingface_hub import hf_hub_download
import os; os.makedirs("$LOCAL_GGUF_DIR", exist_ok=True)
hf_hub_download(repo_id="$REPO_ID", filename="$GGUF_NAME", local_dir="$LOCAL_GGUF_DIR")
PY
fi
[[ -f "$LOCAL_GGUF" ]] || { log "ERROR: download failed"; exit 1; }

# ---- kill stale ports ----
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

# ---- RSS sampler (1 s cadence, full duration) ----
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
  kill -0 "$MOCK_PID" 2>/dev/null || { log "ERROR: mock died"; kill $LLAMA_PID $RSS_PID 2>/dev/null; exit 1; }
  sleep 1
done

# ---- start thermal sampler in parallel (real-time poll of llama log + RSS) ----
log "Starting thermal sampler (interval=30s, duration=${DURATION_SEC}s)"
uv run python scripts/mac_bench/thermal_sampler.py \
  --llama-log "$LLAMA_LOG" --rss-log "$RSS_LOG" \
  --out-csv "$CURVE_CSV" --out-json "$CURVE_JSON" \
  --interval 30 --duration "$DURATION_SEC" > "$OUT_DIR/sampler.log" 2>&1 &
SAMPLER_PID=$!

# ---- start runner against full 156-case MARTA set; will be killed at duration ----
log "Starting runner (full MARTA, parallel=1, thinking on) — will run for ${DURATION_RAW}"
RUN_START=$(date +%s)
uv run python -m harness.runner \
  --cases cases/marta_cases.json --system marta \
  --llm-url "http://localhost:${LLAMA_PORT}/v1" --llm-key sk-mac-bench \
  --llm-model "${SIZE}-metro-v24" \
  --thinking --parallel 1 \
  --mock-url "http://localhost:${MOCK_PORT}" \
  --output "$RAW_RESULTS" > "$OUT_DIR/runner.log" 2>&1 &
RUNNER_PID=$!

# ---- wait until duration elapses or runner finishes (whichever first) ----
DEADLINE=$(( $(date +%s) + DURATION_SEC + 30 ))  # +30s grace for sampler write
while (( $(date +%s) < DEADLINE )); do
  # If sampler finished, we have all the data we need — break
  if ! kill -0 "$SAMPLER_PID" 2>/dev/null; then
    break
  fi
  # If runner finished early (very fast hardware), keep sampler going until duration
  if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
    log "Runner finished early at $(( $(date +%s) - RUN_START ))s; sampler continuing on warm llama-server"
    # Re-launch a tight idle-decode loop so the sampler still sees activity?
    # No — just let the sampler finish; flat tail is meaningful (hardware idle behavior).
    break
  fi
  sleep 5
done

# ---- shutdown ----
log "Stopping runner (PID=$RUNNER_PID)..."
kill "$RUNNER_PID" 2>/dev/null || true
sleep 2
kill -9 "$RUNNER_PID" 2>/dev/null || true

log "Waiting for sampler to finish (max 60s)..."
SAMPLER_DEADLINE=$(( $(date +%s) + 60 ))
while kill -0 "$SAMPLER_PID" 2>/dev/null && (( $(date +%s) < SAMPLER_DEADLINE )); do
  sleep 2
done
kill "$SAMPLER_PID" 2>/dev/null || true

kill "$MOCK_PID" 2>/dev/null || true
kill "$LLAMA_PID" 2>/dev/null || true
sleep 2
kill -9 "$LLAMA_PID" 2>/dev/null || true
kill "$RSS_PID" 2>/dev/null || true
wait 2>/dev/null || true

RUN_END=$(date +%s)
log "Total wallclock: $((RUN_END - RUN_START))s"

# ---- print summary ----
log "Done. Output: $OUT_DIR"
log ""
log "Thermal summary:"
if [[ -f "$CURVE_JSON" ]]; then
  uv run python -c "
import json
s = json.loads(open('$CURVE_JSON').read())
print(f\"  duration:     {s['duration_sec']}s, samples: {s['n_samples']}\")
print(f\"  cold:         {s['tok_s_cold']:.1f} tok/s\")
print(f\"  sustained:    {s['tok_s_sustained_last5']:.1f} tok/s  (last 5 samples)\")
print(f\"  median:       {s['tok_s_median_overall']:.1f} tok/s  (overall)\")
print(f\"  throttle:     {s['throttle_pct_cold_to_sustained']:+.1f}%  (cold → sustained)\")
print(f\"  peak rss:     {s['peak_rss_gb']:.2f} GB\")
"
else
  log "(no thermal_curve.json — sampler may have failed; see $OUT_DIR/sampler.log)"
fi
