#!/usr/bin/env python3
from __future__ import annotations
"""
Train a calibrated baseline classifier (Level 1) on engineered tabular features
and write baseline_score (0..1) into results_stream.parquet.

- Model: LightGBM (default) or XGBoost via --model-type
- Calibration: Platt/sigmoid via CalibratedClassifierCV with CV folds (default cv=5)
- Stores: calibrated model artifact, feature_cols, model_version, feature_set_hash, optional prompt_hash

Example:
  PYTHONPATH=. python scripts/level1_baseline_train.py \
    --in artifacts/features.parquet \
    --id-col transaction_id \
    --label-col is_fraud \
    --out-model artifacts/level1/model.joblib \
    --out-results artifacts/results_stream.parquet \
    --model-version lgbm_v1

Dependencies:
  pip install lightgbm scikit-learn joblib pandas numpy pyarrow
  (optional for --model-type xgboost: pip install xgboost)
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Optional imports guarded at runtime
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None


def parse_args():
    p = argparse.ArgumentParser(description="Train & calibrate Level-1 baseline; write baseline_score to results_stream.")
    p.add_argument("--in", dest="input", required=True, help="Features parquet/csv path")
    p.add_argument("--id-col", default="transaction_id", help="ID column name (string)")
    p.add_argument("--label-col", default="is_fraud", help="Binary label column name (0/1)")
    p.add_argument("--exclude-cols", nargs="*", default=["anomaly_score", "baseline_score"], help="Columns to exclude from features")
    p.add_argument("--out-model", required=True, help="Path to save calibrated model (joblib)")
    p.add_argument("--out-results", required=True, help="Path to write/update results_stream parquet")
    p.add_argument("--model-version", default="l1_v1", help="Model version tag to store")
    p.add_argument("--prompt-hash", default=None, help="Optional prompt_hash to store alongside the baseline (string)")
    p.add_argument("--test-size", type=float, default=0.2, help="Holdout split size for validation metrics")
    p.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm", help="Baseline model type")
    p.add_argument("--calib-cv", type=int, default=5, help="CV folds for calibration (>=2)")
    return p.parse_args()


def load_df(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def pick_feature_cols(df: pd.DataFrame, id_col: str, label_col: str, exclude: List[str]) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in (id_col, label_col):
            continue
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No usable feature columns found. Ensure engineered numeric/bool features exist.")
    return cols


def feature_set_hash(df: pd.DataFrame, cols: List[str]) -> str:
    meta = {c: str(df[c].dtype) for c in cols}
    s = json.dumps({"cols": cols, "dtypes": meta}, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def train_model(model_type: str):
    if model_type == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed. pip install lightgbm")
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. pip install xgboost")
        return XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective="binary:logistic",
            scale_pos_weight=1.0,  # adjust if class imbalance is severe
            tree_method="hist",
        )


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_model = Path(args.out_model)
    out_results = Path(args.out_results)

    df = load_df(in_path)
    if args.id_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"Missing required columns: {args.id_col}, {args.label_col}")

    feat_cols = pick_feature_cols(df, args.id_col, args.label_col, args.exclude_cols)

    # Prepare data
    X_all = df[feat_cols].astype(float)
    y_all = df[args.label_col].astype(int)
    txids = df[args.id_col].astype(str)

    # Holdout split for validation metrics
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=42, stratify=y_all
    )

    # Fit calibrator with CV folds (CalibratedClassifierCV trains the base model inside)
    base = train_model(args.model_type)
    calib = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=max(2, args.calib_cv))
    calib.fit(X_train, y_train)

    # Validation metric on holdout
    try:
        val_auc = roc_auc_score(y_val, calib.predict_proba(X_val)[:, 1])
    except Exception:
        val_auc = None

    # Score entire dataset
    probs = calib.predict_proba(X_all)[:, 1]
    baseline_score = np.clip(probs, 0.0, 1.0)

    # Save model artifact
    out_model.parent.mkdir(parents=True, exist_ok=True)
    artifact: Dict[str, Any] = {
        "calibrated_model": calib,
        "feature_cols": feat_cols,
        "model_version": args.model_version,
        "feature_set_hash": feature_set_hash(df, feat_cols),
        "prompt_hash": args.prompt_hash,
        "model_type": args.model_type,
        "validation_auc": val_auc,
        "calibration_cv": max(2, args.calib_cv),
        "base_params": base.get_params() if hasattr(base, "get_params") else None,
    }
    joblib.dump(artifact, out_model)

    # Write/update results_stream with baseline_score
    res_new = pd.DataFrame({args.id_col: txids, "baseline_score": baseline_score})
    res_new["model_version"] = args.model_version

    if out_results.exists():
        existing = load_df(out_results)
        if args.id_col not in existing.columns:
            raise ValueError(f"{out_results} missing {args.id_col} column")
        existing[args.id_col] = existing[args.id_col].astype(str)

        merged = existing.merge(res_new, on=args.id_col, how="left", suffixes=("", "_new"))
        # Prefer newly computed baseline_score/model_version if present
        if "baseline_score_new" in merged.columns:
            merged["baseline_score"] = merged["baseline_score_new"].fillna(merged.get("baseline_score"))
            merged = merged.drop(columns=["baseline_score_new"])
        if "model_version_new" in merged.columns:
            merged["model_version"] = merged["model_version_new"].fillna(merged.get("model_version"))
            merged = merged.drop(columns=["model_version_new"])

        merged.to_parquet(out_results, index=False)
    else:
        out_results.parent.mkdir(parents=True, exist_ok=True)
        res_new.to_parquet(out_results, index=False)

    print(f"[info] wrote model to {out_model}")
    print(f"[info] wrote/updated results_stream to {out_results}")
    print(f"[info] features used ({len(feat_cols)}): {feat_cols}")
    if val_auc is not None:
        print(f"[info] validation AUC: {val_auc:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())