#!/usr/bin/env bash
# Purpose: Run Level-3 LLM reliably in Codespaces/WSL by:
# - starting a persistent Ollama server
# - detecting available RAM and pre-pulling CPU-friendly models
# - selecting an appropriate model automatically (tiny-first) to avoid OOM/timeouts
# - warmup, run, and validate outputs
#
# Usage (defaults are safe for Codespaces):
#   ./scripts/run_level3_with_ollama_server.sh
#   MODEL=llama3.2:1b SAMPLE=50 BATCH=1 TIMEOUT=300 ./scripts/run_level3_with_ollama_server.sh
#
# Env overrides:
#   MODEL                Preferred model name (e.g., llama3.2:1b, phi3:mini, qwen2.5:1.5b, llama2:7b)
#   IN                   Path to results_stream.parquet
#   EVIDENCE             Path to topk_evidence.parquet
#   OUT_DIR              Output dir for Level-3 artifacts
#   SAMPLE               Sample size for Level-3 run
#   BATCH                Batch size per LLM call
#   TIMEOUT              Per-call timeout (seconds)
#   MAX_DELTA            Max absolute adjustment (±δ)
#   OLLAMA_KEEP_ALIVE    Suggested: 10m to keep model resident
#   OLLAMA_NUM_PARALLEL  Suggested: 1 to reduce RAM spikes
#
# Notes:
# - This script prefers tiny CPU-friendly models to leverage limited Codespace memory:
#     llama3.2:1b, phi3:mini, qwen2.5:1.5b
#   and only attempts llama2:7b if host RAM is sufficient (>= 12 GB).
# - It will pre-pull multiple candidate models and pick the first successfully installed one,
#   unless MODEL is explicitly provided.

set -euo pipefail

# Defaults tuned for Codespaces/WSL
MODEL="${MODEL:-}"  # if empty, we'll auto-select based on memory
IN="${IN:-artifacts/results_stream.parquet}"
EVIDENCE="${EVIDENCE:-artifacts/topk_evidence.parquet}"
OUT_DIR="${OUT_DIR:-artifacts/level3_llm_run}"
SAMPLE="${SAMPLE:-100}"
BATCH="${BATCH:-4}"
TIMEOUT="${TIMEOUT:-120}"
MAX_DELTA="${MAX_DELTA:-0.05}"

echo "[info] IN=$IN EVIDENCE=$EVIDENCE OUT_DIR=$OUT_DIR SAMPLE=$SAMPLE BATCH=$BATCH TIMEOUT=$TIMEOUT MAX_DELTA=$MAX_DELTA"
echo "[info] OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-unset} OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-unset}"

# Recommend keeping models alive and limiting parallel decodes to conserve memory
if [[ -z "${OLLAMA_KEEP_ALIVE:-}" ]]; then
  export OLLAMA_KEEP_ALIVE=10m
fi
if [[ -z "${OLLAMA_NUM_PARALLEL:-}" ]]; then
  export OLLAMA_NUM_PARALLEL=1
fi

# Ensure Ollama is available
if ! command -v ollama >/dev/null 2>&1; then
  echo "[error] ollama not found in PATH. Install Ollama or run in an environment where it's available."
  exit 1
fi

ollama --version || true

# Start persistent server (keeps model resident between calls)
if pgrep -f "ollama serve" >/dev/null; then
  echo "[info] ollama serve is already running"
else
  echo "[info] starting ollama serve in background"
  nohup ollama serve >/tmp/ollama_serve.log 2>&1 & disown || true
  sleep 2
fi

echo "[info] ollama ps:"
ollama ps || true

# Detect available memory (Linux/WSL)
mem_gb=0
if [[ -r /proc/meminfo ]]; then
  mem_kb=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
  if [[ -n "$mem_kb" ]]; then
    mem_gb=$((mem_kb / 1024 / 1024))
  fi
fi
echo "[info] detected RAM: ~${mem_gb} GB"

# Build candidate model list based on memory; prefer tiny CPU-friendly models
# Tiny-first to leverage Codespace limits
candidates=()
if [[ -n "$MODEL" ]]; then
  candidates+=("$MODEL")
fi

# Always try tiny models first
##candidates+=("llama3.2:1b" "phi3:mini" "qwen2.5:1.5b")
candidates+=("llama3.2:1b")

# Try 7B only if RAM is likely sufficient (>= 12 GB)
if (( mem_gb >= 12 )); then
  candidates+=("llama2" "llama2:7b")
fi

echo "[info] candidate models (in order): ${candidates[*]}"

# Helper: pull/install a model if missing
ensure_model() {
  local m="$1"
  if ollama list | awk '{print $1}' | grep -qx "$m"; then
    echo "[info] model present: $m"
    return 0
  fi
  echo "[info] pulling model: $m"
  if ollama pull "$m"; then
    echo "[info] pulled: $m"
    return 0
  else
    echo "[warn] failed to pull: $m"
    return 1
  fi
}

# Pre-pull all candidates (best-effort); select the first that installs
ACTIVE_MODEL=""
for m in "${candidates[@]}"; do
  if ensure_model "$m"; then
    if [[ -z "$ACTIVE_MODEL" ]]; then
      ACTIVE_MODEL="$m"
    fi
  fi
done

if [[ -z "$ACTIVE_MODEL" ]]; then
  echo "[error] failed to pull any candidate model."
  echo "       tried: ${candidates[*]}"
  echo "Hints:"
  echo "  - Check network/proxy."
  echo "  - Run 'ollama list' to verify availability."
  echo "  - Try explicitly: MODEL=llama3.2:1b ./scripts/run_level3_with_ollama_server.sh"
  exit 2
fi

echo "[info] selected ACTIVE_MODEL=$ACTIVE_MODEL"

# Warmup (loads model into memory). Use long timeout for cold load.
echo "[info] warmup (sample_limit=1, timeout=300) MODEL=$ACTIVE_MODEL"
PYTHONPATH=. python scripts/level3_llm.py \
  --in "$IN" \
  --evidence-file "$EVIDENCE" \
  --out_dir artifacts/level3_llm_warm \
  --model "$ACTIVE_MODEL" \
  --max_delta "$MAX_DELTA" \
  --sample_limit 1 \
  --use-ollama \
  --batch-size 1 \
  --timeout 300 \
  --warmup || true

# Optional tiny smoke before main (adjust SAMPLE_SMOKE if needed)
SAMPLE_SMOKE="${SAMPLE_SMOKE:-5}"
echo "[info] smoke run (SAMPLE=$SAMPLE_SMOKE, BATCH=1, TIMEOUT=300) MODEL=$ACTIVE_MODEL"
PYTHONPATH=. python scripts/level3_llm.py \
  --in "$IN" \
  --evidence-file "$EVIDENCE" \
  --out_dir artifacts/level3_llm_smoke \
  --model "$ACTIVE_MODEL" \
  --max_delta "$MAX_DELTA" \
  --sample_limit "$SAMPLE_SMOKE" \
  --use-ollama \
  --batch-size 1 \
  --timeout 300 \
  --no-warmup || true

# Main Level-3 run
echo "[info] main Level-3 run (SAMPLE=$SAMPLE, BATCH=$BATCH, TIMEOUT=$TIMEOUT) MODEL=$ACTIVE_MODEL"
PYTHONPATH=. python scripts/level3_llm.py \
  --in "$IN" \
  --evidence-file "$EVIDENCE" \
  --out_dir "$OUT_DIR" \
  --model "$ACTIVE_MODEL" \
  --max_delta "$MAX_DELTA" \
  --sample_limit "$SAMPLE" \
  --use-ollama \
  --batch-size "$BATCH" \
  --timeout "$TIMEOUT" \
  --no-warmup

# Validate Level-3 outputs
echo "[info] validation"
PYTHONPATH=. python scripts/validate_level3.py \
  --llm-parsed "$OUT_DIR/llm_parsed.parquet" \
  --results "$IN" \
  --evidence "$EVIDENCE" \
  --llm-raw "$OUT_DIR/llm_raw.jsonl" \
  --manifest "$OUT_DIR/manifest.json" \
  --max-delta "$MAX_DELTA" \
  --max-needs-review 0.03 \
  --max-latency-median 5.0 \
  --output-report "$OUT_DIR/validation_report.json" || true

# Inspect representative successes/errors (if inspector script present)
if [[ -f "scripts/llm_raw_inspect.py" ]]; then
  echo "[info] inspector (first successes/errors)"
  PYTHONPATH=. python scripts/llm_raw_inspect.py || true
else
  echo "[info] (optional) add scripts/llm_raw_inspect.py to inspect raw outputs quickly."
fi

echo "[info] done. See:"
echo "  - $OUT_DIR/validation_report.json"
echo "  - $OUT_DIR/llm_parsed.parquet"
echo "  - $OUT_DIR/llm_raw.jsonl"