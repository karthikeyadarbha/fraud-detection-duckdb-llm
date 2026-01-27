#!/usr/bin/env bash
# Train & calibrate Level-1 baseline; write baseline_score to results_stream and print a summary.
# Usage:
#   ./scripts/run_level1_baseline.sh
# Env overrides:
#   IN=artifacts/features.parquet
#   ID_COL=transaction_id
#   LABEL_COL=is_fraud
#   OUT_MODEL=artifacts/level1/model.joblib
#   OUT_RESULTS=artifacts/results_stream.parquet
#   MODEL_VERSION=lgbm_v1

set -euo pipefail

IN="${IN:-artifacts/features.parquet}"
ID_COL="${ID_COL:-transaction_id}"
LABEL_COL="${LABEL_COL:-is_fraud}"
OUT_MODEL="${OUT_MODEL:-artifacts/level1/model.joblib}"
OUT_RESULTS="${OUT_RESULTS:-artifacts/results_stream.parquet}"
MODEL_VERSION="${MODEL_VERSION:-lgbm_v1}"

echo "[info] IN=$IN ID_COL=$ID_COL LABEL_COL=$LABEL_COL"
echo "[info] OUT_MODEL=$OUT_MODEL OUT_RESULTS=$OUT_RESULTS MODEL_VERSION=$MODEL_VERSION"

# deps
python - <<'PY' || true
import sys
pkgs = ["lightgbm","scikit-learn","joblib","pandas","numpy","pyarrow"]
missing=[]
for p in pkgs:
    try:
        __import__(p.replace('-', '_'))
    except Exception:
        missing.append(p)
if missing:
    print("Missing packages:", missing)
    sys.exit(1)
PY
if [[ $? -ne 0 ]]; then
  echo "[info] installing required packages..."
  pip install -q lightgbm scikit-learn joblib pandas numpy pyarrow
fi

# run training
PYTHONPATH=. python scripts/level1_baseline_train.py \
  --in "$IN" \
  --id-col "$ID_COL" \
  --label-col "$LABEL_COL" \
  --out-model "$OUT_MODEL" \
  --out-results "$OUT_RESULTS" \
  --model-version "$MODEL_VERSION"

# summary
python - <<PY
import pandas as pd, json
p='$OUT_RESULTS'
df=pd.read_parquet(p)
print("[info] rows:", len(df))
if 'baseline_score' in df.columns:
    bs=df['baseline_score'].astype(float)
    print("[info] baseline_score nulls:", int(bs.isna().sum()))
    print("[info] baseline_score min/max:", float(bs.min()), float(bs.max()))
    print(bs.describe().to_string())
else:
    print("[warn] baseline_score not found in results_stream.")
PY
echo "[info] done."