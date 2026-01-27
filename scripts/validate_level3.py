#!/usr/bin/env python3
"""
scripts/validate_level3.py

Run validation checks on Level-3 LLM artifacts and produce a pass/fail summary.

This variant is defensive and fixes ambiguous boolean-evaluation bugs
and handles array-like top_k_evidence cells robustly.

Usage:
  PYTHONPATH=. python scripts/validate_level3.py \
    --llm-parsed artifacts/level3_llm_run/llm_parsed.parquet \
    --results artifacts/results_stream.parquet \
    --evidence artifacts/topk_evidence.parquet \
    --llm-raw artifacts/level3_llm_run/llm_raw.jsonl \
    --manifest artifacts/level3_llm_run/manifest.json \
    --max-delta 0.05 \
    --max-needs-review 0.03 \
    --max-latency-median 5.0 \
    --output-report artifacts/level3_llm_run/validation_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


DEFAULTS = {
    "max_delta": 0.05,
    "max_needs_review_rate": 0.03,
    "max_parse_error_rate": 0.01,
    "max_latency_median": 5.0,  # seconds
    "max_ollama_error_rate": 0.05,
    "max_explanation_len": 250,
    "max_clamped_at_limit_rate": 0.25,  # fraction of rows exactly at +/- max_delta
}


def load_manifest(manifest_path: Optional[Path]) -> Dict[str, Any]:
    if not manifest_path:
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _extract_ids_from_item(item) -> Set[str]:
    ids = set()
    if item is None:
        return ids
    # dict with id
    if isinstance(item, dict):
        if "id" in item:
            ids.add(str(item["id"]))
        return ids
    # string (maybe JSON)
    if isinstance(item, str):
        try:
            parsed = json.loads(item)
        except Exception:
            return ids
        return _extract_ids_from_item(parsed)
    # list/tuple/array
    if isinstance(item, (list, tuple, np.ndarray, pd.Series)):
        for e in list(item):
            ids.update(_extract_ids_from_item(e))
        return ids
    return ids


def load_topk_map(evidence_path: Optional[Path], llm_raw_path: Optional[Path]) -> Dict[str, Set[str]]:
    """
    Returns mapping transaction_id -> set(evidence_ids)
    Tries evidence_path first (parquet/csv/jsonl). If absent, attempts to parse prompts in llm_raw.jsonl.

    This implementation is defensive against array-like dataframe cells (numpy arrays / lists / pd.Series).
    """
    mapping: Dict[str, Set[str]] = {}
    if evidence_path and evidence_path.exists():
        suf = evidence_path.suffix.lower()
        try:
            if suf in (".parquet", ".pq"):
                ev_df = pd.read_parquet(evidence_path)
            elif suf in (".csv", ".txt"):
                ev_df = pd.read_csv(evidence_path)
            else:
                ev_df = None
        except Exception:
            ev_df = None

        if ev_df is not None:
            for _, r in ev_df.iterrows():
                tx = str(r.get("transaction_id"))
                val = r.get("top_k_evidence", None)
                ids = set()

                # None / NaN handling
                if val is None:
                    mapping[tx] = ids
                    continue
                # If val is array-like (list, tuple, ndarray, Series)
                if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                    ids = _extract_ids_from_item(val)
                    mapping[tx] = ids
                    continue
                # If val is a string, try parse JSON, else ignore
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        ids = _extract_ids_from_item(parsed)
                        mapping[tx] = ids
                        continue
                    except Exception:
                        # not JSON: can't extract ids reliably; leave as empty set
                        mapping[tx] = ids
                        continue
                # If it's a dict
                if isinstance(val, dict):
                    ids = _extract_ids_from_item(val)
                    mapping[tx] = ids
                    continue
                # Fallback: attempt pd.isna safely (scalar)
                try:
                    if pd.isna(val):
                        mapping[tx] = set()
                        continue
                except Exception:
                    pass
                # unknown type: leave empty set
                mapping[tx] = ids

    # fallback: try to extract top_k from llm_raw.jsonl prompts if mapping is empty
    if (not mapping) and llm_raw_path and llm_raw_path.exists():
        try:
            with open(llm_raw_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    tx = str(obj.get("transaction_id", ""))
                    prompt = obj.get("prompt")
                    if not prompt:
                        continue
                    # heuristic: find top_k_evidence snippet in prompt and parse next JSON array/object
                    key = '"top_k_evidence"'
                    idx = prompt.find(key)
                    if idx == -1:
                        key = "'top_k_evidence'"
                        idx = prompt.find(key)
                    if idx == -1:
                        continue
                    snippet = prompt[idx:]
                    # find first '[' or '{' after key
                    b_idx = None
                    for ch in ('[', '{'):
                        p = snippet.find(ch)
                        if p != -1:
                            b_idx = p
                            break
                    if b_idx is None:
                        continue
                    # find matching closing bracket using simple parse (works for well-formed JSON)
                    sub = snippet[b_idx:]
                    # attempt to extract balanced JSON substring
                    depth = 0
                    end = None
                    for i, ch in enumerate(sub):
                        if ch in '[{':
                            depth += 1
                        elif ch in ']}':
                            depth -= 1
                            if depth == 0:
                                end = i
                                break
                    if end is None:
                        continue
                    candidate = sub[: end + 1]
                    try:
                        parsed = json.loads(candidate)
                        ids = _extract_ids_from_item(parsed)
                        mapping[tx] = ids
                    except Exception:
                        # ignore parse errors
                        continue
        except Exception:
            pass

    return mapping


def clamp_value(v: float, max_delta: float) -> float:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 0.0
    return max(-max_delta, min(max_delta, float(v)))


def collect_llm_raw_stats(llm_raw_path: Optional[Path]) -> Tuple[int, int, List[float]]:
    """
    Returns (total_lines, ollama_error_count, list_of_elapsed_seconds)
    """
    total = 0
    ollama_err = 0
    elapsed = []
    if not llm_raw_path or not llm_raw_path.exists():
        return total, ollama_err, elapsed
    with open(llm_raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                o = json.loads(line)
            except Exception:
                continue
            ro = o.get("raw_output", "")
            if isinstance(ro, str) and ro.startswith("__OLLAMA_ERROR__"):
                ollama_err += 1
            es = o.get("elapsed_seconds", None)
            if es is not None:
                try:
                    elapsed.append(float(es))
                except Exception:
                    pass
    return total, ollama_err, elapsed


def run_checks(
    llm_parsed_path: Path,
    results_path: Optional[Path],
    evidence_path: Optional[Path],
    llm_raw_path: Optional[Path],
    manifest_path: Optional[Path],
    max_delta_arg: Optional[float],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    # Load parsed parquet
    if not llm_parsed_path.exists():
        raise FileNotFoundError(f"{llm_parsed_path} not found")
    df = pd.read_parquet(llm_parsed_path)
    total = len(df)
    report["rows"] = total

    # Load manifest to get max_delta if provided
    manifest = load_manifest(manifest_path) if manifest_path else {}
    max_delta = max_delta_arg if max_delta_arg is not None else manifest.get("max_delta", DEFAULTS["max_delta"])
    try:
        max_delta = float(max_delta)
    except Exception:
        max_delta = DEFAULTS["max_delta"]
    report["max_delta_used"] = max_delta

    # Basic presence checks
    required_cols = ["transaction_id", "llm_adjustment_raw", "llm_adjustment_clamped", "llm_adjustment_valid", "llm_evidence_ids", "needs_review", "llm_raw_output"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    report["missing_columns"] = missing_cols

    # rates (use explicit column checks to avoid ambiguous truth evaluations)
    if "needs_review" in df.columns:
        needs_review_count = int(df["needs_review"].astype(bool).sum())
    else:
        needs_review_count = total
    if "llm_adjustment_valid" in df.columns:
        llm_valid_count = int(df["llm_adjustment_valid"].astype(bool).sum())
    else:
        llm_valid_count = 0

    if "parse_error" in df.columns:
        parsed_with_error = df["parse_error"].fillna("").astype(str)
        parse_error_counts = parsed_with_error[parsed_with_error != ""].value_counts().to_dict()
    else:
        parse_error_counts = {}

    report.update({
        "needs_review_count": needs_review_count,
        "needs_review_rate": needs_review_count / total if total else 0.0,
        "llm_valid_count": llm_valid_count,
        "llm_valid_rate": llm_valid_count / total if total else 0.0,
        "parse_error_counts": parse_error_counts,
    })

    # evidence subset check
    topk_map = load_topk_map(evidence_path, llm_raw_path)
    evidence_mismatch = []
    missing_topk = 0
    # iterate safely (use itertuples for speed and stability)
    for row in df.itertuples(index=False):
        # access by attribute if available, else fallback to dict access
        try:
            tx = str(getattr(row, "transaction_id"))
        except Exception:
            tx = str(row[0]) if len(row) > 0 else ""
        eids = getattr(row, "llm_evidence_ids", None)
        if eids is None:
            eids = []
        # ensure eids is a list
        if not isinstance(eids, (list, tuple, set)):
            # sometimes stored as numpy array
            try:
                eids = list(eids)
            except Exception:
                eids = []
        topk_ids = topk_map.get(tx)
        if topk_ids is None:
            missing_topk += 1
            continue
        bad = [eid for eid in eids if eid not in topk_ids]
        if bad:
            evidence_mismatch.append({"transaction_id": tx, "bad_ids": bad, "topk_ids": list(topk_ids)})
    report["evidence_mismatch_count"] = len(evidence_mismatch)
    report["evidence_mismatch_examples"] = evidence_mismatch[:10]
    report["missing_topk_count"] = missing_topk

    # clamping correctness and percent at clamp limit
    clamped_mismatch = []
    at_limit_count = 0
    for row in df.itertuples(index=False):
        try:
            raw_adj = getattr(row, "llm_adjustment_raw", None)
            clamped = getattr(row, "llm_adjustment_clamped", None)
            expected = clamp_value(raw_adj, max_delta)
            if clamped is None:
                clamped_val = 0.0
            else:
                clamped_val = float(clamped)
            if abs(clamped_val - expected) > 1e-8:
                clamped_mismatch.append({"transaction_id": str(getattr(row, "transaction_id", "")), "raw": raw_adj, "clamped": clamped_val, "expected": expected})
            if abs(expected - max_delta) < 1e-8 or abs(expected + max_delta) < 1e-8:
                at_limit_count += 1
        except Exception:
            clamped_mismatch.append({"transaction_id": str(getattr(row, "transaction_id", "")), "raw": getattr(row, "llm_adjustment_raw", None), "clamped": getattr(row, "llm_adjustment_clamped", None), "expected": "<error>"})
    report["clamped_mismatch_count"] = len(clamped_mismatch)
    report["clamped_mismatch_examples"] = clamped_mismatch[:10]
    report["at_limit_count"] = at_limit_count
    report["at_limit_rate"] = at_limit_count / total if total else 0.0

    # confidence and explanation checks
    conf_bad = 0
    long_expl = []
    max_expl_len = thresholds.get("max_explanation_len", DEFAULTS["max_explanation_len"])
    for row in df.itertuples(index=False):
        conf = getattr(row, "confidence", None)
        if conf is not None:
            try:
                if not (0.0 <= float(conf) <= 1.0):
                    conf_bad += 1
            except Exception:
                conf_bad += 1
        expl = getattr(row, "explanation", "") or ""
        if not isinstance(expl, str):
            expl = str(expl)
        if len(expl) > max_expl_len:
            long_expl.append({"transaction_id": str(getattr(row, "transaction_id", "")), "len": len(expl), "explanation": expl[:200]})
    report["confidence_bad_count"] = conf_bad
    report["long_explanation_count"] = len(long_expl)
    report["long_explanation_examples"] = long_expl[:10]

    # transaction_id uniqueness
    if "transaction_id" in df.columns:
        dup_tx = int(df["transaction_id"].duplicated().sum())
    else:
        dup_tx = total
    report["duplicate_tx_count"] = int(dup_tx)

    # elapsed_seconds / raw outputs
    total_raw_lines, ollama_err_count, elapsed_list = collect_llm_raw_stats(llm_raw_path)
    report["llm_raw_total_lines"] = total_raw_lines
    report["ollama_error_count"] = ollama_err_count
    report["ollama_error_rate"] = (ollama_err_count / total_raw_lines) if total_raw_lines else 0.0
    if elapsed_list:
        report["elapsed_median"] = float(np.median(elapsed_list))
        report["elapsed_mean"] = float(np.mean(elapsed_list))
        report["elapsed_p90"] = float(np.percentile(elapsed_list, 90))
    else:
        report["elapsed_median"] = None
        report["elapsed_mean"] = None
        report["elapsed_p90"] = None

    # compare counts vs results_stream (optional)
    if results_path and results_path.exists():
        try:
            res_df = pd.read_parquet(results_path)
            report["results_rows"] = len(res_df)
            res_txs = set(res_df["transaction_id"].astype(str).tolist())
            parsed_txs = set(df["transaction_id"].astype(str).tolist())
            missing_in_results = sorted(list(parsed_txs - res_txs))[:10]
            report["parsed_not_in_results_count"] = len(parsed_txs - res_txs)
            report["parsed_not_in_results_examples"] = missing_in_results
        except Exception:
            report["results_rows"] = None
            report["parsed_not_in_results_count"] = None
    else:
        report["results_rows"] = None

    # Compose pass/fail verdicts based on thresholds
    verdicts = {}
    verdicts["needs_review_rate_ok"] = report["needs_review_rate"] <= thresholds.get("max_needs_review_rate", DEFAULTS["max_needs_review_rate"])
    parse_err_total = sum(report["parse_error_counts"].values()) if report.get("parse_error_counts") else 0
    verdicts["parse_error_ok"] = (parse_err_total / total if total else 0.0) <= thresholds.get("max_parse_error_rate", DEFAULTS["max_parse_error_rate"])
    verdicts["ollama_error_ok"] = report["ollama_error_rate"] <= thresholds.get("max_ollama_error_rate", DEFAULTS["max_ollama_error_rate"])
    verdicts["clamped_ok"] = report["clamped_mismatch_count"] == 0
    verdicts["at_limit_rate_ok"] = report["at_limit_rate"] <= thresholds.get("max_clamped_at_limit_rate", DEFAULTS["max_clamped_at_limit_rate"])
    verdicts["confidence_ok"] = report["confidence_bad_count"] == 0
    verdicts["explanation_len_ok"] = report["long_explanation_count"] == 0
    verdicts["latency_ok"] = True
    if report["elapsed_median"] is not None:
        verdicts["latency_ok"] = report["elapsed_median"] <= thresholds.get("max_latency_median", DEFAULTS["max_latency_median"])

    report["verdicts"] = verdicts
    report["passed_all"] = all(verdicts.values())

    return report


def pretty_print_report(report: Dict[str, Any]) -> None:
    def kv(k):
        v = report.get(k)
        print(f"{k:30}: {v}")

    print("=" * 80)
    print("Level-3 LLM Validation Report")
    print("generated_at:", datetime.utcnow().isoformat() + "Z")
    print("=" * 80)
    kv("rows")
    kv("results_rows")
    kv("max_delta_used")
    print("-" * 80)
    kv("needs_review_count")
    kv("needs_review_rate")
    kv("llm_valid_count")
    kv("llm_valid_rate")
    print("-" * 80)
    print("parse_error_counts:")
    for k, v in (report.get("parse_error_counts") or {}).items():
        print(f"  {k}: {v}")
    print("-" * 80)
    kv("evidence_mismatch_count")
    kv("missing_topk_count")
    if report.get("evidence_mismatch_examples"):
        print("evidence_mismatch_examples (up to 5):")
        for ex in report["evidence_mismatch_examples"]:
            print(f"  {ex}")
    print("-" * 80)
    kv("clamped_mismatch_count")
    kv("at_limit_count")
    kv("at_limit_rate")
    if report.get("clamped_mismatch_examples"):
        print("clamped_mismatch_examples (up to 5):")
        for ex in report["clamped_mismatch_examples"]:
            print(f"  {ex}")
    print("-" * 80)
    kv("confidence_bad_count")
    kv("long_explanation_count")
    if report.get("long_explanation_examples"):
        print("long_explanation_examples (up to 5):")
        for ex in report["long_explanation_examples"]:
            print(f"  {ex}")
    print("-" * 80)
    kv("duplicate_tx_count")
    print("-" * 80)
    kv("llm_raw_total_lines")
    kv("ollama_error_count")
    kv("ollama_error_rate")
    kv("elapsed_median")
    kv("elapsed_mean")
    kv("elapsed_p90")
    print("-" * 80)
    print("Verdicts (True = OK):")
    for k, v in (report.get("verdicts") or {}).items():
        print(f"  {k:30}: {v}")
    print("=" * 80)
    print("Overall PASS:", report.get("passed_all"))
    print("=" * 80)


def parse_args():
    p = argparse.ArgumentParser(description="Validate Level-3 LLM artifacts and output a pass/fail summary.")
    p.add_argument("--llm-parsed", required=True, help="Path to llm_parsed.parquet")
    p.add_argument("--results", required=False, help="Path to original results_stream.parquet (optional)")
    p.add_argument("--evidence", required=False, help="Path to topk_evidence.parquet (optional)")
    p.add_argument("--llm-raw", required=False, help="Path to llm_raw.jsonl (optional)")
    p.add_argument("--manifest", required=False, help="Path to manifest.json (optional)")
    p.add_argument("--max-delta", type=float, default=None, help="Max delta (overrides manifest if provided)")
    p.add_argument("--max-needs-review", type=float, default=DEFAULTS["max_needs_review_rate"], help="Threshold for needs_review rate")
    p.add_argument("--max-parse-error", type=float, default=DEFAULTS["max_parse_error_rate"], help="Threshold for parse error rate")
    p.add_argument("--max-ollama-error", type=float, default=DEFAULTS["max_ollama_error_rate"], help="Threshold for raw Ollama error rate")
    p.add_argument("--max-latency-median", type=float, default=DEFAULTS["max_latency_median"], help="Threshold median elapsed_seconds (s)")
    p.add_argument("--max-explanation-len", type=int, default=DEFAULTS["max_explanation_len"], help="Max explanation length (chars)")
    p.add_argument("--max-clamped-at-limit-rate", type=float, default=DEFAULTS["max_clamped_at_limit_rate"], help="Max fraction of rows exactly at +/- max_delta")
    p.add_argument("--output-report", required=False, help="Optional path to save JSON report")
    return p.parse_args()


def main():
    args = parse_args()
    llm_parsed = Path(args.llm_parsed)
    results = Path(args.results) if args.results else None
    evidence = Path(args.evidence) if args.evidence else None
    llm_raw = Path(args.llm_raw) if args.llm_raw else None
    manifest = Path(args.manifest) if args.manifest else None

    thresholds = {
        "max_needs_review_rate": float(args.max_needs_review),
        "max_parse_error_rate": float(args.max_parse_error),
        "max_ollama_error_rate": float(args.max_ollama_error),
        "max_latency_median": float(args.max_latency_median),
        "max_explanation_len": int(args.max_explanation_len),
        "max_clamped_at_limit_rate": float(args.max_clamped_at_limit_rate),
    }

    try:
        report = run_checks(
            llm_parsed_path=llm_parsed,
            results_path=results,
            evidence_path=evidence,
            llm_raw_path=llm_raw,
            manifest_path=manifest,
            max_delta_arg=args.max_delta,
            thresholds=thresholds,
        )
    except Exception as e:
        print("Validation failed with exception:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

    pretty_print_report(report)
    if args.output_report:
        try:
            with open(args.output_report, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, default=lambda o: "<non-serializable>")
            print(f"Wrote report to {args.output_report}")
        except Exception as e:
            print("Failed to write report:", e, file=sys.stderr)

    # Exit non-zero if any critical verdict is False
    critical = [
        "needs_review_rate_ok",
        "parse_error_ok",
        "ollama_error_ok",
        "clamped_ok",
        "confidence_ok",
        "explanation_len_ok",
        "latency_ok",
    ]
    failed = [k for k in critical if not report.get("verdicts", {}).get(k, False)]
    if failed:
        print("VALIDATION FAILED on:", failed, file=sys.stderr)
        sys.exit(3)
    print("VALIDATION PASSED", file=sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()