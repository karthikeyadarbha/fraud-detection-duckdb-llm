"""
utils.combiner

Combiner utilities for reconciling Level-1 / Level-2 / Level-3 signals.

Provides:
- validate_and_apply_llm_adjustment(df, ...)
- combine_scores(df, combine_mode, anom_weight, anom_gate_threshold, guard_threshold)

Designed to be safe and defensive: baseline is authoritative; anomalies and LLM adjustments
are applied under gating/weights; guard can force baseline-only and emits artifacts/guard_triggered.txt.
"""
import os
from pathlib import Path
import pandas as pd

def validate_and_apply_llm_adjustment(df, adj_col='llm_adjustment', evidence_col='llm_evidence_ids',
                                      topk_ids_col='topk_evidence_ids', max_delta=0.05):
    """
    Validate LLM adjustments:
      - ensure adjustments exist and are numeric
      - ensure evidence_ids are subset of topk_evidence_ids (best-effort)
      - clamp adjustment to [-max_delta, +max_delta]
      - produce 'llm_adjustment_valid' boolean and 'llm_adjustment_clamped' numeric column
    Returns modified df (a copy).
    """
    df = df.copy()

    if adj_col not in df.columns:
        df[adj_col] = 0.0
    if evidence_col not in df.columns:
        df[evidence_col] = [[] for _ in range(len(df))]
    if topk_ids_col not in df.columns:
        df[topk_ids_col] = [[] for _ in range(len(df))]

    def validate_row(adj, evidence_ids, topk_ids):
        valid = True
        try:
            adj_val = float(adj)
        except Exception:
            valid = False
            adj_val = 0.0
        # evidence check (best-effort; coerce to sets when possible)
        try:
            e_ids = set(evidence_ids) if isinstance(evidence_ids, (list, set, tuple)) else set()
            t_ids = set(topk_ids) if isinstance(topk_ids, (list, set, tuple)) else set()
            if not e_ids.issubset(t_ids):
                valid = False
        except Exception:
            valid = False
        # clamp adj
        clamped = max(-max_delta, min(max_delta, adj_val))
        return valid, clamped

    out_valid = []
    out_clamped = []
    for adj, eid, tid in zip(df[adj_col], df[evidence_col], df[topk_ids_col]):
        v, c = validate_row(adj, eid, tid)
        out_valid.append(v)
        out_clamped.append(c)

    df['llm_adjustment_valid'] = out_valid
    df['llm_adjustment_clamped'] = out_clamped
    return df

def combine_scores(df, combine_mode='baseline', anom_weight=0.05, anom_gate_threshold=0.2, guard_threshold=None):
    """
    Combine baseline_score, anomaly_score and optional llm_adjustment_clamped into final_score.

    combine_mode:
      - 'baseline' : final_score = baseline_score
      - 'gated'    : anomaly applied only when baseline_score < anom_gate_threshold
      - 'weighted' : final = baseline*(1-w) + anomaly*w

    Guard:
      - If guard_threshold provided (float 0..1) and median(anomaly_score) > guard_threshold,
        force baseline-only and write artifacts/guard_triggered.txt
    """
    df = df.copy()

    if 'baseline_score' not in df.columns:
        raise ValueError("baseline_score column not present in DataFrame")

    # If anomaly column missing -> baseline-only
    if 'anomaly_score' not in df.columns:
        df['final_score'] = df['baseline_score']
        return df

    # runtime guard: check median
    try:
        anom_median = float(df['anomaly_score'].median())
    except Exception:
        anom_median = None

    # guard threshold may come from env
    if guard_threshold is None:
        _env = os.getenv('ANOM_MEDIAN_GUARD', None)
        try:
            guard_threshold = float(_env) if _env is not None else None
        except Exception:
            guard_threshold = None

    if anom_median is not None and guard_threshold is not None and anom_median > float(guard_threshold):
        print(f"[guard] anomaly median={anom_median:.3f} > {guard_threshold} — forcing baseline-only")
        try:
            Path("artifacts").mkdir(parents=True, exist_ok=True)
            with open("artifacts/guard_triggered.txt", "w") as fh:
                fh.write(f"anom_median={anom_median:.6f}, threshold={guard_threshold}\n")
        except Exception:
            pass
        df['final_score'] = df['baseline_score']
        return df

    # If LLM adjustments present, apply them first (additive and clamped)
    base_col = 'baseline_score'
    if 'llm_adjustment_clamped' in df.columns:
        df['final_score'] = (df['baseline_score'] + df['llm_adjustment_clamped']).clip(0.0, 1.0)
        base_col = 'final_score'

    # Combine anomaly according to mode
    if combine_mode == 'baseline' or 'anomaly_score' not in df.columns:
        df['final_score'] = df[base_col]
        return df

    if combine_mode == 'gated':
        df['final_score'] = df[base_col]
        mask = df['baseline_score'] < anom_gate_threshold
        df.loc[mask, 'final_score'] = (
            (df.loc[mask, base_col] * (1.0 - anom_weight)) +
            (df.loc[mask, 'anomaly_score'] * anom_weight)
        ).clip(0.0, 1.0)
        return df

    if combine_mode == 'weighted':
        df['final_score'] = (
            (df[base_col] * (1.0 - anom_weight)) +
            (df['anomaly_score'] * anom_weight)
        ).clip(0.0, 1.0)
        return df

    # fallback
    df['final_score'] = df[base_col]
    return df