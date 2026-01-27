#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env)
MODEL="${MODEL:-llama3.2:1b}"          # tiny CPU-friendly; fallbacks below
IN="${IN:-artifacts/results_stream.parquet}"
EVIDENCE="${EVIDENCE:-artifacts/topk_evidence.parquet}"
OUT_DIR="${OUT_DIR:-artifacts/level3_llm_run}"
SAMPLE="${SAMPLE:-100}"                 # number of transactions to process
BATCH="${BATCH:-1}"                     # use 1 for Codespaces stability
TIMEOUT="${TIMEOUT:-300}"               # per-call timeout seconds
MAX_DELTA="${MAX_DELTA:-0.05}"          # bounded adjustment ±δ
MERGE="${MERGE:-1}"                     # 1 = merge L3 outputs into results_stream_with_llm.parquet

echo "[info] Level-3 execute: MODEL=$MODEL SAMPLE=$SAMPLE BATCH=$BATCH TIMEOUT=$TIMEOUT MAX_DELTA=$MAX_DELTA"
echo "[info] IN=$IN EVIDENCE=$EVIDENCE OUT_DIR=$OUT_DIR MERGE=$MERGE"

# Recommend keeping model hot and limiting parallel decodes for Codespaces
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-10m}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
echo "[info] OLLAMA_KEEP_ALIVE=$OLLAMA_KEEP_ALIVE OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"

# 0) Ensure Ollama present
if ! command -v ollama >/dev/null 2>&1; then
  echo "[error] ollama not found. Install it (Codespaces):"
  echo "  curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi
ollama --version || true

# 1) Start persistent server
if pgrep -f "ollama serve" >/dev/null; then
  echo "[info] ollama serve already running"
else
  echo "[info] starting ollama serve in background"
  nohup ollama serve >/tmp/ollama_serve.log 2>&1 & disown || true
  sleep 2
fi
echo "[info] ollama ps:"; ollama ps || true

# 2) Install tiny model with fallbacks
candidates=("$MODEL" "llama3.2:1b" "phi3:mini" "qwen2.5:1.5b")
ACTIVE_MODEL=""
for m in "${candidates[@]}"; do
  echo "[info] checking model: $m"
  if ollama list | awk '{print $1}' | grep -qx "$m"; then
    ACTIVE_MODEL="$m"; echo "[info] model present: $m"; break
  fi
  if ollama pull "$m"; then
    ACTIVE_MODEL="$m"; echo "[info] pulled: $m"; break
  else
    echo "[warn] failed to pull: $m"
  fi
done
if [[ -z "$ACTIVE_MODEL" ]]; then
  echo "[error] failed to pull any candidate model. Tried: ${candidates[*]}"; exit 2
fi
echo "[info] ACTIVE_MODEL=$ACTIVE_MODEL"

# 3) Warmup (loads model into memory)
echo "[info] warmup (sample_limit=1, timeout=300)"
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

# 4) Main Level-3 run (no --no-warmup)
echo "[info] main Level-3 run"
PYTHONPATH=. python scripts/level3_llm.py \
  --in "$IN" \
  --evidence-file "$EVIDENCE" \
  --out_dir "$OUT_DIR" \
  --model "$ACTIVE_MODEL" \
  --max_delta "$MAX_DELTA" \
  --sample_limit "$SAMPLE" \
  --use-ollama \
  --batch-size "$BATCH" \
  --timeout "$TIMEOUT"

# 5) Validate outputs
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

# 6) Inspect first successes/errors (if inspector present)
if [[ -f "scripts/llm_raw_inspect.py" ]]; then
  echo "[info] inspector (first successes/errors)"
  PYTHONPATH=. python scripts/llm_raw_inspect.py || true
fi

# 7) Merge validated L3 adjustments back into results_stream_with_llm.parquet
if [[ "$MERGE" == "1" ]]; then
  echo "[info] merging L3 outputs into artifacts/results_stream_with_llm.parquet"
  python - "$OUT_DIR" <<'PY'
import sys, pandas as pd
out_dir = sys.argv[1]
res_path='artifacts/results_stream.parquet'
llm_path=f'{out_dir}/llm_parsed.parquet'
out_path='artifacts/results_stream_with_llm.parquet'
res = pd.read_parquet(res_path)
llm = pd.read_parquet(llm_path)
res['transaction_id']=res['transaction_id'].astype(str)
llm['transaction_id']=llm['transaction_id'].astype(str)
keep = ['transaction_id','llm_adjustment_clamped','llm_adjustment_raw','llm_adjustment_valid','llm_evidence_ids','needs_review','explanation','confidence']
llm_sel = llm[[c for c in keep if c in llm.columns]].copy()
merged = res.merge(llm_sel, on='transaction_id', how='left')
# Defaults for missing
for col, val in [('llm_adjustment_clamped',0.0), ('llm_adjustment_valid',False)]:
    if col in merged.columns:
        merged[col] = merged[col].fillna(val)
merged.to_parquet(out_path, index=False)
print("[info] wrote", out_path, "rows:", len(merged))
PY
fi

echo "[info] done. See:"
echo "  - $OUT_DIR/validation_report.json"
echo "  - $OUT_DIR/llm_parsed.parquet"
echo "  - $OUT_DIR/llm_raw.jsonl"