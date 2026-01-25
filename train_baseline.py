"""
Train a calibrated LightGBM baseline model (robust to sklearn API differences).

This version:
- Trains an LGBMClassifier on the training split.
- Performs ISOTONIC calibration manually using sklearn.isotonic.IsotonicRegression
  on a held-out chronological calibration split (preserves time split semantics).
- Wraps the fitted LGBM and isotonic calibrator in a small wrapper object that
  exposes predict_proba (so downstream code can call calibrated.predict_proba).
- Persists artifacts to artifacts/.

Usage:
  python train_baseline.py /path/to/transactions.csv
"""
import os
import json
import joblib
import hashlib
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier

from data_prep import load_csv, FitEncoder, transform_features

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def model_version_hash(obj: dict) -> str:
    s = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:12]


class CalibratedWrapper:
    """
    Wraps a fitted classifier and a post-hoc calibrator (e.g., IsotonicRegression).
    Exposes predict_proba(X) -> Nx2 array as sklearn classifiers do.
    """
    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        try:
            calibrated = self.calibrator.predict(raw)
        except Exception:
            # fallback: if calibrator fails, return raw scores
            calibrated = raw
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.vstack([1.0 - calibrated, calibrated]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train(csv_path: str, out_model: str = None):
    df = load_csv(csv_path)

    # normalize label
    df['is_fraud'] = df['is_fraud'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)

    # time-based split: train on older 80%, calibrate on last 20%
    df = df.sort_values('timestamp')
    split = int(len(df) * 0.8)
    df_train = df.iloc[:split].copy()
    df_calib = df.iloc[split:].copy()

    # fit encoder on training
    encoder = FitEncoder()
    encoder.fit(df_train, label_col='is_fraud')

    X_train = transform_features(df_train, encoder)
    X_calib = transform_features(df_calib, encoder)

    feature_cols = [c for c in X_train.columns if c not in ('transaction_id', 'timestamp', 'sender_account', 'receiver_account')]

    X_tr = X_train[feature_cols].values
    y_tr = df_train['is_fraud'].values
    X_cb = X_calib[feature_cols].values
    y_cb = df_calib['is_fraud'].values

    # LightGBM classifier (scikit-learn API)
    clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=64, random_state=42)
    clf.fit(X_tr, y_tr)

    # Produce raw probabilities on calibration split
    val_probs = clf.predict_proba(X_cb)[:, 1]

    # Fit isotonic calibrator on calibration split (post-hoc calibration)
    iso = IsotonicRegression(out_of_bounds='clip')
    try:
        iso.fit(val_probs, y_cb)
    except Exception:
        # If isotonic fails (rare), fall back to identity calibrator
        class IdentityCalibrator:
            def predict(self, x): return np.asarray(x)
        iso = IdentityCalibrator()

    # Wrap classifier + calibrator
    calibrated = CalibratedWrapper(clf, iso)

    # compute metrics on calibration set using calibrated probabilities
    probs = calibrated.predict_proba(X_cb)[:, 1]
    auc = roc_auc_score(y_cb, probs) if len(np.unique(y_cb)) > 1 else None
    brier = brier_score_loss(y_cb, probs) if len(y_cb) > 0 else None

    model_manifest = {
        "model_type": "lgbm_with_isotonic_calibration",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "training_rows": int(len(df_train)),
        "calibration_rows": int(len(df_calib)),
        "auc_on_calibration": float(auc) if auc is not None else None,
        "brier_score_on_calibration": float(brier) if brier is not None else None,
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }

    mver = model_version_hash(model_manifest)
    model_manifest['model_version'] = mver

    out_model = out_model or os.path.join(ARTIFACT_DIR, f"lgb_baseline_{mver}.pkl")
    out_manifest = os.path.join(ARTIFACT_DIR, f"model_manifest_baseline_{mver}.json")

    # Persist:
    # - calibrated wrapper (has model + calibrator inside)
    # - feature list
    # - encoder state (so feature transforms are reproducible)
    joblib.dump({'model': calibrated, 'features': feature_cols, 'encoder': encoder.to_dict(), 'model_version': mver}, out_model)
    with open(out_manifest, 'w') as f:
        json.dump(model_manifest, f, indent=2)

    print("Saved calibrated baseline to:", out_model)
    print("Saved manifest to:", out_manifest)
    print("Calibration AUC:", auc, "Brier:", brier)
    return out_model, out_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="path to transactions csv")
    parser.add_argument("--out", help="output model path (joblib)")
    args = parser.parse_args()
    train(args.csv, args.out)