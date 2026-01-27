#!/usr/bin/env python3
"""
Prepare Level-1 features parquet from a source table that includes:
- transaction_id (string)
- is_fraud (0/1)
- numeric/bool columns to use as features

It will:
- Select numeric/bool columns, excluding derived columns
- Standardize them (z-score) to produce *_z versions
- Save to artifacts/features.parquet by default

Usage:
  PYTHONPATH=. python scripts/level1_prepare_features.py \
    --in-source artifacts/results_stream.parquet \
    --id-col transaction_id \
    --label-col is_fraud \
    --out-features artifacts/features.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

EXCLUDE_DEFAULT = {
    "baseline_score", "anomaly_score",
    "llm_adjustment_raw", "llm_adjustment_clamped", "llm_adjustment_valid",
    "llm_evidence_ids", "needs_review",
}

def parse_args():
    p = argparse.ArgumentParser(description="Prepare Level-1 features parquet.")
    p.add_argument("--in-source", required=True, help="Input parquet/csv with transaction_id, is_fraud, and candidate features")
    p.add_argument("--id-col", default="transaction_id", help="ID column name")
    p.add_argument("--label-col", default="is_fraud", help="Binary label column name (0/1)")
    p.add_argument("--exclude-cols", nargs="*", default=list(EXCLUDE_DEFAULT), help="Columns to exclude from feature candidates")
    p.add_argument("--out-features", default="artifacts/features.parquet", help="Output features parquet path")
    return p.parse_args()

def load_df(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def pick_numeric_cols(df: pd.DataFrame, id_col: str, label_col: str, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in (id_col, label_col) or c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols

def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd == 0.0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mu) / sd

def main():
    args = parse_args()
    src_path = Path(args.in_source)
    out_path = Path(args.out_features)

    df = load_df(src_path)
    if args.id_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"Missing required columns: {args.id_col}, {args.label_col} in {src_path}")

    df[args.id_col] = df[args.id_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(int)

    feat_base = pick_numeric_cols(df, args.id_col, args.label_col, args.exclude_cols)

    if not feat_base:
        raise ValueError("No numeric/bool columns found to use as features. Check your source input.")

    # Build standardized features as *_z
    feat_z = {}
    for c in feat_base:
        feat_z[f"{c}_z"] = zscore(df[c])

    out = pd.DataFrame({
        args.id_col: df[args.id_col],
        args.label_col: df[args.label_col],
        **feat_z,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[info] wrote features to {out_path}")
    print(f"[info] base feature columns ({len(feat_base)}): {feat_base}")
    print(f"[info] standardized feature columns ({len(feat_z)}): {list(feat_z.keys())}")

if __name__ == "__main__":
    raise SystemExit(main())