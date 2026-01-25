#!/usr/bin/env python3
"""
recombine_and_evaluate.py

Load a results table (baseline + optional extras), merge Level‑2 anomaly artifacts if present,
validate/apply Level‑3 (LLM) adjustments, then compute final_score using a guarded combiner.

This script implements the recombine/ensemble policy described in the repo:
- baseline (Level-1) is authoritative
- anomaly (Level-2) is complementary and applied via gating/weighting
- LLM (Level-3) adjustments are validated and clamped before application
- an ANOM_MEDIAN_GUARD can force baseline-only and write artifacts/guard_triggered.txt

Usage:
  python recombine_and_evaluate.py \
    --in artifacts/results_stream.parquet \
    --out artifacts/results_stream_recombined.parquet \
    --combine-mode weighted \
    --anom-weight 0.05 \
    --anomaly-path artifacts/level2_isof/anomaly_scores.parquet

Notes:
- Input can be parquet or csv (extension determines writer/reader).
- The script is defensive: if anomaly artifact or LLM columns are missing, it proceeds with baseline-only.
- The combiner utility is expected at utils.combiner (see utils/combiner.py).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

import pandas as pd

# local utilities (ensure repo root on PYTHONPATH or utils is a package)
from utils.combiner import combine_scores, validate_and_apply_llm_adjustment

DEFAULT_ANOMALY_PATH = "artifacts/level2_isof/anomaly_scores.parquet"


def read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file {path} not found")
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif suf in (".csv", ".txt"):
        return pd.read_csv(path)
    else:
        # attempt parquet first, then csv
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)


def write_output(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suf = out_path.suffix.lower()
    if suf in (".parquet", ".pq"):
        df.to_parquet(out_path, index=False)
    elif suf in (".csv", ".txt"):
        df.to_csv(out_path, index=False)
    else:
        # default to parquet
        df.to_parquet(out_path, index=False)


def merge_anomaly(df: pd.DataFrame, artifact_path: Path) -> pd.DataFrame:
    """Merge anomaly_scores artifact into df on transaction_id (defensive: normalize types and dtypes)."""
    if not artifact_path.exists():
        print(f"[info] anomaly artifact {artifact_path} not found; continuing without anomaly_score")
        return df

    try:
        df_anom = pd.read_parquet(artifact_path)
    except Exception:
        try:
            df_anom = pd.read_csv(artifact_path)
        except Exception as exc:
            print(f"[warn] failed to read anomaly artifact {artifact_path}: {exc}")
            return df

    # Ensure transaction_id is string on both sides
    df = df.copy()
    if 'transaction_id' not in df.columns:
        print(f"[warn] input missing transaction_id; cannot merge anomaly")
        return df
    df['transaction_id'] = df['transaction_id'].astype(str)

    if 'transaction_id' not in df_anom.columns:
        print(f"[warn] anomaly artifact {artifact_path} missing transaction_id; skipping merge")
        return df
    df_anom = df_anom.copy()
    df_anom['transaction_id'] = df_anom['transaction_id'].astype(str)

    # Normalize anomaly score column name if necessary
    if 'anomaly_score' not in df_anom.columns:
        other_cols = [c for c in df_anom.columns if c != 'transaction_id']
        if other_cols:
            df_anom = df_anom.rename(columns={other_cols[0]: 'anomaly_score'})
        else:
            print(f"[warn] anomaly artifact {artifact_path} contains no value columns; skipping")
            return df

    # Coerce anomaly_score to numeric and report counts
    df_anom['anomaly_score'] = pd.to_numeric(df_anom['anomaly_score'], errors='coerce')
    n_num = int(df_anom['anomaly_score'].notna().sum())
    print(f"[info] anomaly artifact read: {n_num}/{len(df_anom)} numeric anomaly_score values (others -> NaN)")

    # Merge and coerce merged anomaly_score to numeric; fill missing with 0.0
    df_merged = df.merge(df_anom[['transaction_id', 'anomaly_score']], on='transaction_id', how='left')
    df_merged['anomaly_score'] = pd.to_numeric(df_merged['anomaly_score'], errors='coerce').fillna(0.0)

    print(f"[info] after merge: anomaly_score > 0 for {int((df_merged['anomaly_score']>0).sum())} rows out of {len(df_merged)}")
    return df_merged

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recombine baseline + anomaly + LLM adjustments into final_score")
    p.add_argument("--in", dest="in_path", required=True, help="Input results (parquet or csv) with baseline_score")
    p.add_argument("--out", dest="out_path", required=True, help="Output path (parquet or csv) to write recombined results")
    p.add_argument("--combine-mode", default="baseline", choices=["baseline", "gated", "weighted"],
                   help="Combine mode for anomaly (baseline|gated|weighted)")
    p.add_argument("--anom-weight", type=float, default=0.05, help="Anomaly weight when using weighted/gated modes")
    p.add_argument("--anomaly-path", default=None, help="Path to anomaly_scores artifact (parquet). If omitted, default used.")
    p.add_argument("--anom-gate-threshold", default=None, help="Gate threshold override (float). If unset, uses ANOM_GATE_THRESHOLD env or combiner default")
    p.add_argument("--guard-threshold", default=None, help="ANOM_MEDIAN_GUARD override (float). If unset, uses env ANOM_MEDIAN_GUARD")
    p.add_argument("--quiet", action="store_true", help="Minimize console output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    try:
        df_all = read_input(in_path)
    except Exception as exc:
        print(f"[error] failed to read input {in_path}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    if "baseline_score" not in df_all.columns:
        print("[warn] baseline_score column not found in input; recombine requires baseline_score", file=sys.stderr)

    # Merge anomaly artifact if present
    anom_path = Path(args.anomaly_path) if args.anomaly_path else Path(os.getenv("LEVEL2_ANOMALY_PATH", DEFAULT_ANOMALY_PATH))
    try:
        df_all = merge_anomaly(df_all, anom_path)
    except Exception as exc:
        print(f"[warn] exception while merging anomaly: {exc}", file=sys.stderr)

    # Validate and apply LLM adjustments (if any)
    try:
        df_all = validate_and_apply_llm_adjustment(
            df_all,
            adj_col='llm_adjustment',
            evidence_col='llm_evidence_ids',
            topk_ids_col='topk_evidence_ids',
            max_delta=0.05
        )
    except Exception as exc:
        print(f"[warn] validate_and_apply_llm_adjustment failed: {exc}", file=sys.stderr)

    # Determine combine params
    chosen_combine_mode = args.combine_mode
    anom_weight = args.anom_weight
    try:
        if args.anom_gate_threshold is not None:
            anom_gate_threshold = float(args.anom_gate_threshold)
        else:
            anom_gate_threshold = float(os.getenv("ANOM_GATE_THRESHOLD", 0.2))
    except Exception:
        anom_gate_threshold = 0.2

    try:
        if args.guard_threshold is not None:
            guard_threshold = float(args.guard_threshold)
        else:
            guard_threshold = float(os.getenv("ANOM_MEDIAN_GUARD", 0.6))
    except Exception:
        guard_threshold = None

    # Compute combine (combiner will enforce guard and policies)
    try:
        df_final = combine_scores(
            df_all,
            combine_mode=chosen_combine_mode,
            anom_weight=anom_weight,
            anom_gate_threshold=anom_gate_threshold,
            guard_threshold=guard_threshold
        )
    except Exception as exc:
        print(f"[error] combine_scores failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3

    # Write output
    try:
        write_output(df_final, out_path)
    except Exception as exc:
        print(f"[error] failed to write output {out_path}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 4

    # Print diagnostics
    if not args.quiet:
        print(f"[info] wrote recombined results to {out_path}")
        try:
            if "anomaly_score" in df_final.columns:
                anom_median = float(df_final["anomaly_score"].median())
                print(f"[info] anomaly_score median={anom_median:.4f}")
            if "final_score" in df_final.columns and "baseline_score" in df_final.columns:
                eq_all = (df_final["final_score"] == df_final["baseline_score"]).all()
                print(f"[info] final_score equals baseline_score for all rows?: {eq_all}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())