"""
End-to-end pipeline script that supports:
- Baseline calibrated LightGBM artifact (joblib) created by train_baseline.py
- Anomaly artifact: either IsolationForest artifact (joblib) OR TensorFlow autoencoder artifact (joblib + .keras/.h5 SavedModel)
- Optional LLM step using local Ollama via llm_adapter.run_llm_for_transaction

This file includes a local CalibratedWrapper class so joblib can unpickle baseline artifacts
that were saved while running train_baseline.py as a script.
"""
import os
import joblib
import json
import pandas as pd
import numpy as np
from data_prep import load_csv, FitEncoder, transform_features
from combine_scores import combine_scores, decide_action

ARTIFACT_DIR = "artifacts"
RESULT_PATH = os.path.join(ARTIFACT_DIR, "results.parquet")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# Provide a CalibratedWrapper here so joblib can unpickle model artifacts that reference it.
# This mirrors the wrapper defined in train_baseline.py: it expects attributes 'model' and 'calibrator'.
class CalibratedWrapper:
    """
    Minimal wrapper to expose the sklearn-like predict_proba interface for a fitted model
    and a post-hoc calibrator (with a .predict method).
    This class is intentionally small and compatible with the object saved by train_baseline.py.
    """
    def __init__(self, model=None, calibrator=None):
        self.model = model
        self.calibrator = calibrator

    def predict_proba(self, X):
        # The saved object may have been pickled with model expecting DataFrame column names;
        # attempt to preserve behavior by passing a DataFrame if feature names exist on the model.
        try:
            raw = self.model.predict_proba(X)[:, 1]
        except Exception:
            # try forcing DataFrame if X is numpy and model has feature_names_in_
            try:
                cols = getattr(self.model, "feature_name_", None) or getattr(self.model, "feature_names_in_", None)
                if cols is not None and isinstance(X, np.ndarray):
                    import pandas as pd
                    raw = self.model.predict_proba(pd.DataFrame(X, columns=cols))[:, 1]
                else:
                    raw = self.model.predict_proba(X)[:, 1]
            except Exception:
                # fallback: try predict_proba on X as-is and allow exceptions to propagate
                raw = self.model.predict_proba(X)[:, 1]

        try:
            calibrated = self.calibrator.predict(raw)
        except Exception:
            calibrated = raw
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.vstack([1.0 - calibrated, calibrated]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load_baseline(path: str):
    """
    Load baseline artifact saved by train_baseline.py.
    Artifact is expected to be a joblib file containing a dict with keys:
      - 'model' : CalibratedWrapper (or similar)
      - 'features'
      - 'encoder'
    """
    obj = joblib.load(path)
    # If the loaded object is the wrapper itself (older formats), wrap into dict
    if isinstance(obj, CalibratedWrapper):
        return {'model': obj, 'features': None, 'encoder': None}
    return obj


def load_anomaly(path: str):
    """
    Load anomaly artifact. Two supported formats:
      - IsolationForest artifact: {'iso','scaler','features','manifest','encoder'}
      - Autoencoder artifact: {'scaler','features','manifest','encoder','model_file'} (model_file is .keras/.h5)
    """
    return joblib.load(path)


def normalize_anomaly(raw_scores: np.ndarray, p1: float, p99: float) -> np.ndarray:
    clipped = np.clip(raw_scores, p1, p99)
    normalized = (clipped - p1) / (p99 - p1) if (p99 - p1) > 0 else np.zeros_like(clipped)
    return normalized


def demo(baseline_model_path: str, anomaly_model_path: str, csv_path: str, run_llm: bool = False):
    baseline = load_baseline(baseline_model_path)
    anomaly = load_anomaly(anomaly_model_path)

    df = load_csv(csv_path)

    # prepare using baseline encoder if present
    encoder = FitEncoder.from_dict(baseline['encoder']) if (baseline and 'encoder' in baseline and baseline['encoder']) else None
    feats = transform_features(df, encoder).reset_index(drop=True)

    # feature columns
    feature_cols = baseline.get('features')
    if feature_cols is None:
        # If features not stored in artifact, infer by excluding meta columns
        feature_cols = [c for c in feats.columns if c not in ('transaction_id','timestamp','sender_account','receiver_account')]

    X = feats[feature_cols].values

    # baseline scoring
    clf = baseline['model']
    # ensure wrapper-like interface: if it's a dict with 'model', handle accordingly
    if isinstance(clf, dict) and 'model' in clf:
        clf = clf['model']

    baseline_probs = clf.predict_proba(X)[:, 1]

    # anomaly scoring
    anom_scores = np.zeros(len(feats), dtype=float)
    manifest = None

    if isinstance(anomaly, dict) and 'iso' in anomaly:
        print("Using IsolationForest anomaly artifact")
        iso = anomaly['iso']
        scaler = anomaly['scaler']
        features_anom = anomaly['features']
        X_anom = feats[features_anom].values
        Xs = scaler.transform(X_anom)
        raw_anom = -iso.score_samples(Xs)  # higher -> more anomalous
        manifest = anomaly.get('manifest', {})
        p1 = manifest.get('p1_raw_anomaly', float(np.percentile(raw_anom, 1)))
        p99 = manifest.get('p99_raw_anomaly', float(np.percentile(raw_anom, 99)))
        anom_scores = normalize_anomaly(raw_anom, p1, p99)

    elif isinstance(anomaly, dict) and ('model_file' in anomaly or ('manifest' in anomaly and anomaly['manifest'].get('model_type') == 'tf_autoencoder')):
        print("Using TensorFlow/Keras autoencoder artifact")
        # lazy import tensorflow
        try:
            import tensorflow as tf
        except Exception as e:
            raise RuntimeError("TensorFlow is required to score autoencoder artifact: " + str(e))

        scaler = anomaly['scaler']
        features_anom = anomaly['features']
        X_anom = feats[features_anom].values
        Xs = scaler.transform(X_anom)

        model_file = anomaly.get('model_file')
        if model_file and os.path.exists(model_file):
            ae = tf.keras.models.load_model(model_file, compile=False)
        else:
            # fallback to model_dir key (SavedModel) if present
            model_dir = anomaly.get('model_dir')
            if model_dir and os.path.exists(model_dir):
                ae = tf.keras.models.load_model(model_dir, compile=False)
            else:
                raise RuntimeError("Autoencoder artifact does not contain a valid 'model_file' or 'model_dir' path.")

        recon = ae.predict(Xs, batch_size=256)
        mse = np.mean(np.square(Xs - recon), axis=1)
        manifest = anomaly.get('manifest', {})
        p1 = manifest.get('p1_mse', float(np.percentile(mse, 1)))
        p99 = manifest.get('p99_mse', float(np.percentile(mse, 99)))
        anom_scores = normalize_anomaly(mse, p1, p99)

    else:
        raise RuntimeError("Unrecognized anomaly artifact format. Expect IsolationForest artifact or autoencoder artifact.")

    # build a small evidence pool (stub): non-fraud historical transactions from CSV
    evidence_pool = []
    if 'is_fraud' in df.columns:
        df_pool = df[df['is_fraud'].astype(str).str.upper() == 'FALSE']
    else:
        df_pool = df
    for i, r in df_pool.head(500).iterrows():
        evidence_pool.append({'id': str(r.get('transaction_id', f'idx_{i}')), 'text': f"amount={r.get('amount')}, loc={r.get('location')}"})
    topk = 5
    default_topk = evidence_pool[:topk] if len(evidence_pool) else []

    run_llm_local = run_llm
    if run_llm_local:
        from llm_adapter import run_llm_for_transaction

    records = []
    for idx in range(len(feats)):
        row = feats.iloc[idx]
        baseline_score = float(baseline_probs[idx])
        anomaly_score = float(anom_scores[idx])

        if run_llm_local:
            txn = {"transaction_id": row.get('transaction_id')}
            features_dict = row.to_dict()
            try:
                llm_out = run_llm_for_transaction(txn, features_dict, baseline_score, default_topk)
                llm_adj = float(llm_out.get('llm_adjustment', 0.0)) if not llm_out.get('needs_review', False) else 0.0
                prompt_hash = llm_out.get('_prompt_hash')
                llm_explanation = llm_out.get('explanation', '')
                llm_evidence = llm_out.get('evidence_ids', [])
                llm_confidence = llm_out.get('confidence', 0.0)
                llm_needs_review = llm_out.get('needs_review', False)
            except Exception as e:
                llm_adj = 0.0
                prompt_hash = None
                llm_explanation = ""
                llm_evidence = []
                llm_confidence = 0.0
                llm_needs_review = True
        else:
            llm_adj = 0.0
            prompt_hash = None
            llm_explanation = ""
            llm_evidence = []
            llm_confidence = 0.0
            llm_needs_review = False

        final = combine_scores(baseline_score, anomaly_score, llm_adj, policy='max')
        action = decide_action(final)

        records.append({
            "transaction_id": row.get('transaction_id'),
            "baseline_score": baseline_score,
            "anomaly_score": anomaly_score,
            "llm_adjustment": llm_adj,
            "llm_confidence": llm_confidence,
            "llm_needs_review": llm_needs_review,
            "llm_explanation": llm_explanation,
            "llm_evidence_ids": json.dumps(llm_evidence),
            "final_score": final,
            "decision": action,
            "llm_prompt_hash": prompt_hash
        })

    out_df = pd.DataFrame(records)
    out_df.to_parquet(RESULT_PATH, index=False)
    print("Saved results to", RESULT_PATH)
    return out_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_model", help="path to baseline joblib")
    parser.add_argument("anomaly_model", help="path to anomaly joblib (iso or ae)")
    parser.add_argument("csv", help="transactions csv")
    parser.add_argument("--run-llm", action="store_true", help="call LLM step (requires Ollama and llm_adapter)")
    args = parser.parse_args()
    demo(args.baseline_model, args.anomaly_model, args.csv, run_llm=args.run_llm)