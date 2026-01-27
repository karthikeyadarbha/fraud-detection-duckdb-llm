#!/usr/bin/env python3
from __future__ import annotations
"""
Level 3 — LLM (explain & constrained adjust)
"""
import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import subprocess


@dataclass
class Args:
    input: Path
    evidence_file: Path
    out_dir: Path
    model: str
    max_delta: float
    sample_limit: Optional[int]
    use_ollama: bool
    batch_size: int
    timeout: int
    warmup: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Run Level-3 (LLM explain & constrained adjust)")
    p.add_argument("--in", dest="input", required=True, help="results_stream parquet/csv with baseline_score and transaction_id")
    p.add_argument("--evidence-file", required=True, help="top-K evidence parquet/csv with transaction_id, evidence_id, text (optional rank)")
    p.add_argument("--out_dir", required=True, help="output directory for llm run artifacts")
    p.add_argument("--model", default="llama3.2:1b", help="Ollama model name/tag (e.g., llama3.2:1b, phi3:mini)")
    p.add_argument("--max_delta", type=float, default=0.05, help="Maximum absolute adjustment allowed")
    p.add_argument("--sample_limit", type=int, default=None, help="Limit number of rows to process")
    p.add_argument("--use-ollama", action="store_true", help="Use Ollama CLI (required for local models)")
    p.add_argument("--batch-size", type=int, default=1, help="Number of rows per batch (sequential for Ollama CLI)")
    p.add_argument("--timeout", type=int, default=120, help="Per-call timeout in seconds")
    p.add_argument("--warmup", action="store_true", help="Warm up run (process a synthetic row only)")
    a = p.parse_args()
    return Args(
        input=Path(a.input),
        evidence_file=Path(a.evidence_file),
        out_dir=Path(a.out_dir),
        model=a.model,
        max_delta=float(a.max_delta),
        sample_limit=a.sample_limit,
        use_ollama=bool(a.use_ollama),
        batch_size=int(a.batch_size),
        timeout=int(a.timeout),
        warmup=bool(a.warmup),
    )


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _group_evidence(evd: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    if "transaction_id" not in evd.columns:
        raise ValueError("evidence file missing 'transaction_id'")
    cols = set(evd.columns)
    id_col = "evidence_id" if "evidence_id" in cols else ("id" if "id" in cols else None)
    text_col = "text" if "text" in cols else ("evidence_text" if "evidence_text" in cols else None)
    rank_col = "rank" if "rank" in cols else None
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for _, r in evd.iterrows():
        tx = str(r["transaction_id"])
        eid = str(r[id_col]) if id_col and not pd.isna(r.get(id_col, None)) else None
        txt = str(r[text_col]) if text_col and not pd.isna(r.get(text_col, None)) else None
        rk = int(r[rank_col]) if rank_col and not pd.isna(r.get(rank_col, None)) else None
        item = {}
        if eid is not None:
            item["id"] = eid
        if txt is not None:
            item["text"] = txt[:400]
        if rk is not None:
            item["rank"] = rk
        groups.setdefault(tx, []).append(item)
    for tx, lst in groups.items():
        if any("rank" in it for it in lst):
            groups[tx] = sorted(lst, key=lambda d: d.get("rank", 1_000_000))
    return groups


def _select_feature_snippet(row: pd.Series) -> str:
    candidates = [
        "amount_z",
        "time_since_last_transaction_z",
        "spending_deviation_score_z",
        "velocity_score_z",
        "geo_anomaly_score_z",
    ]
    parts = []
    for c in candidates:
        if c in row and pd.notna(row[c]):
            try:
                v = float(row[c])
                parts.append(f"{c}={v:+.2f}")
            except Exception:
                continue
    return ", ".join(parts) if parts else "n/a"


def _build_prompt(txid: str, baseline: float, feats: str, evidence: List[Dict[str, Any]], max_delta: float) -> str:
    eid_list = [e.get("id") for e in evidence if e.get("id") is not None]
    e_lines = []
    for i, e in enumerate(evidence, 1):
        eid = e.get("id", f"E{i}")
        txt = e.get("text", "")
        e_lines.append(f"- id: {eid}\n  text: {txt}")

    example = {
        "transaction_id": "T123",
        "llm_adjustment": 0.01,
        "evidence_ids": ["E1", "E2"],
        "explanation": "Velocity and device mismatch suggest mild risk; small upward adjustment.",
        "confidence": 0.6
    }
    fallback = {
        "transaction_id": "<TX>",
        "llm_adjustment": 0.0,
        "evidence_ids": [],
        "explanation": "unable to produce strict JSON; no change",
        "confidence": None
    }

    prompt = f"""You are a strict fraud-triage assistant. RETURN EXACTLY one valid JSON object and NOTHING ELSE.
Do NOT echo the input. Do NOT include baseline_score, features, or top_k_evidence in your output.

Schema (must include these keys and types):
{{
  "transaction_id": "<string>",
  "llm_adjustment": <number>,       // additive delta in [-{max_delta:.4f}, +{max_delta:.4f}]
  "evidence_ids": ["E1","E2"],      // list of evidence ids (subset of provided ids only)
  "explanation": "<string>",        // concise factual explanation (<= 250 chars)
  "confidence": <number|null>       // optional (0.0 .. 1.0) or null
}}

Example:
{json.dumps(example, ensure_ascii=False)}

If you cannot produce a valid JSON, return this exact fallback (replace <TX> with the id):
{json.dumps(fallback, ensure_ascii=False)}

Rules:
- evidence_ids MUST be a subset of: {eid_list if eid_list else []}
- llm_adjustment MUST be within [-{max_delta:.4f}, +{max_delta:.4f}] and small in magnitude.
- Be concise and factual based ONLY on provided evidence and features.

Transaction:
- transaction_id: {txid}
- baseline_score: {baseline:.4f}
- features: {feats}

Top-K Evidence:
{chr(10).join(e_lines) if e_lines else "- (none)"}
"""
    return prompt


def _extract_first_json_blob(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_code_fences(s: str) -> str:
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _load_json_relaxed(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    blob = _extract_first_json_blob(raw)
    candidate = blob or raw
    candidate_stripped = _strip_code_fences(candidate)
    try:
        obj = json.loads(candidate_stripped)
        if isinstance(obj, str):
            inner = _strip_code_fences(obj)
            inner_blob = _extract_first_json_blob(inner) or inner
            obj2 = json.loads(inner_blob)
            if isinstance(obj2, dict):
                return obj2, inner_blob
            return None, None
        if isinstance(obj, dict):
            return obj, candidate_stripped
        inner_blob = _extract_first_json_blob(candidate_stripped)
        if inner_blob:
            obj2 = json.loads(inner_blob)
            if isinstance(obj2, dict):
                return obj2, inner_blob
        return None, None
    except Exception:
        raw2 = _strip_code_fences(raw)
        inner_blob = _extract_first_json_blob(raw2)
        if inner_blob:
            try:
                obj3 = json.loads(inner_blob)
                if isinstance(obj3, dict):
                    return obj3, inner_blob
            except Exception:
                pass
        return None, None


def _run_ollama(model: str, prompt: str, timeout: int) -> Tuple[bool, str, float]:
    t0 = time.time()
    cmd = ["ollama", "run", model]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            msg = proc.stderr.decode("utf-8", errors="replace").strip()
            return False, f"__OLLAMA_ERROR__ rc={proc.returncode} err={msg}", elapsed
        out = proc.stdout.decode("utf-8", errors="replace")
        return True, out, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return False, "__OLLAMA_ERROR__ timeout", elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return False, f"__OLLAMA_ERROR__ {repr(e)}", elapsed


def run_level3(args: Args) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "llm_raw.jsonl"

    rows_for_llm: List[Dict[str, Any]] = []
    evidence_map: Dict[str, List[Dict[str, Any]]] = {}

    if args.warmup:
        rows_for_llm = [{
            "transaction_id": "WARMUP",
            "baseline_score": 0.0,
            "feature_snippet": "warmup",
        }]
        evidence_map = {"WARMUP": [{"id": "E1", "text": "baseline=0.0000"}]}
    else:
        df = _read_table(args.input)
        if "transaction_id" not in df.columns or "baseline_score" not in df.columns:
            raise ValueError("input must contain 'transaction_id' and 'baseline_score' columns")
        df["transaction_id"] = df["transaction_id"].astype(str)
        if args.sample_limit is not None:
            df = df.head(int(args.sample_limit))
        rows_for_llm = []
        for _, r in df.iterrows():
            rows_for_llm.append({
                "transaction_id": str(r["transaction_id"]),
                "baseline_score": float(r["baseline_score"]),
                "feature_snippet": _select_feature_snippet(r),
            })
        evd = _read_table(args.evidence_file)
        evd["transaction_id"] = evd["transaction_id"].astype(str)
        evidence_map = _group_evidence(evd)

    print(f"[info] running Level-3 LLM on {len(rows_for_llm)} rows; model={args.model}; max_delta={args.max_delta}; batch_size={args.batch_size}")
    total = len(rows_for_llm)
    parsed_rows: List[Dict[str, Any]] = []

    if raw_path.exists():
        raw_path.unlink()

    for i in range(0, total, max(1, args.batch_size)):
        batch = rows_for_llm[i : i + max(1, args.batch_size)]
        for row in batch:
            txid = row["transaction_id"]
            baseline = float(row["baseline_score"])
            feats = row["feature_snippet"]
            evlist = evidence_map.get(txid, []) or []
            prompt = _build_prompt(txid, baseline, feats, evlist, args.max_delta)

            if not args.use_ollama:
                ok, raw_out, elapsed = (True, json.dumps({
                    "transaction_id": txid,
                    "llm_adjustment": 0.0,
                    "evidence_ids": [e.get("id") for e in evlist if e.get("id")],
                    "explanation": "LLM disabled; no adjustment.",
                    "confidence": None
                }), 0.0)
            else:
                ok, raw_out, elapsed = _run_ollama(args.model, prompt, args.timeout)

            raw_line = {
                "transaction_id": txid,
                "timestamp_utc": _now_utc_iso(),
                "prompt": None,  # keep raw lean; warmups logged in warm dir
                "raw_output": raw_out,
                "model": args.model,
                "elapsed_seconds": elapsed,
            }
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(raw_line, ensure_ascii=False) + "\n")

            parsed = _parse_and_validate(
                txid=txid,
                raw_output=raw_out,
                provided_evidence_ids=[e.get("id") for e in evlist if e.get("id")],
                max_delta=args.max_delta,
            )
            parsed["elapsed_seconds"] = elapsed
            parsed_rows.append(parsed)

        print(f"[info] processed {min(i+len(batch), total)}/{total} rows (last batch size={len(batch)})")

    # Save parsed outputs with dtype normalization
    parsed_out_path = out_dir / "llm_parsed.parquet"
    import json as _json

    def _to_str_json(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        if isinstance(x, (dict, list)):
            return _json.dumps(x, ensure_ascii=False)
        return str(x)

    def _to_list_str(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, str):
            try:
                obj = _json.loads(v)
                if isinstance(obj, list):
                    return [str(i) for i in obj]
                return [str(obj)]
            except Exception:
                return [v]
        if isinstance(v, (list, tuple, set)):
            return [str(i) for i in v]
        return [str(v)]

    df_out = pd.DataFrame(parsed_rows)

    if "llm_parsed_json" in df_out.columns:
        df_out["llm_parsed_json"] = df_out["llm_parsed_json"].apply(_to_str_json).astype("string")

    if "llm_evidence_ids" in df_out.columns:
        df_out["llm_evidence_ids"] = df_out["llm_evidence_ids"].apply(_to_list_str)

    for c in ["llm_adjustment_raw", "llm_adjustment_clamped", "confidence", "elapsed_seconds"]:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    for c in ["llm_adjustment_valid", "needs_review", "at_limit"]:
        if c in df_out.columns:
            df_out[c] = df_out[c].astype("boolean")

    df_out = df_out.convert_dtypes()
    df_out.to_parquet(parsed_out_path, index=False)
    print(f"[info] wrote parsed outputs to {parsed_out_path}")

    manifest = {
        "generated_at": _now_utc_iso(),
        "model": args.model,
        "max_delta": args.max_delta,
        "rows": total,
        "sample_limit": args.sample_limit,
        "batch_size": args.batch_size,
        "timeout": args.timeout,
        "warmup": args.warmup,
        "use_ollama": args.use_ollama,
        "out_dir": str(args.out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[info] wrote manifest to {out_dir / 'manifest.json'}")
    return 0


def _parse_and_validate(
    txid: str,
    raw_output: str,
    provided_evidence_ids: List[str],
    max_delta: float,
) -> Dict[str, Any]:
    parsed_json_text: Optional[str] = None
    parsed_obj: Optional[Dict[str, Any]] = None
    needs_review = False
    at_limit = False
    valid = False
    conf = None
    adj_raw = 0.0
    evidence_ids: List[str] = []
    explanation = ""

    if isinstance(raw_output, str) and raw_output.startswith("__OLLAMA_ERROR__"):
        needs_review = True
    else:
        obj, json_text_used = _load_json_relaxed(raw_output)
        parsed_obj = obj
        parsed_json_text = json_text_used
        if parsed_obj is None:
            needs_review = True

    if parsed_obj:
        present_adj = "llm_adjustment" in parsed_obj
        present_evid = "evidence_ids" in parsed_obj
        present_expl = "explanation" in parsed_obj

        try:
            raw_adj = parsed_obj.get("llm_adjustment", 0.0)  # type: ignore
            adj_raw = float(raw_adj)
        except Exception:
            needs_review = True
            adj_raw = 0.0

        explanation = str(parsed_obj.get("explanation", "") or "")  # type: ignore
        if len(explanation) == 0:
            needs_review = True
        elif len(explanation) > 600:
            needs_review = True
            explanation = explanation[:600]

        ev_ids = parsed_obj.get("evidence_ids", [])  # type: ignore
        if isinstance(ev_ids, (list, tuple)):
            evidence_ids = [str(e) for e in ev_ids if e is not None]
        elif ev_ids is None:
            evidence_ids = []
        else:
            evidence_ids = [str(ev_ids)]

        if any(e not in set(provided_evidence_ids) for e in evidence_ids):
            needs_review = True

        if "confidence" in parsed_obj and parsed_obj["confidence"] is not None:  # type: ignore
            try:
                c = float(parsed_obj["confidence"])  # type: ignore
                if math.isnan(c) or c < 0 or c > 1:
                    needs_review = True
                else:
                    conf = c
            except Exception:
                needs_review = True

        if not (present_adj and present_evid and present_expl):
            needs_review = True

        if abs(adj_raw) > max_delta + 1e-9:
            at_limit = True
        valid = not needs_review

    adj_clamped = float(np.clip(adj_raw, -max_delta, max_delta))

    return {
        "transaction_id": txid,
        "llm_parsed_json": parsed_json_text if parsed_json_text is not None else (raw_output if isinstance(raw_output, str) else None),
        "llm_adjustment_raw": adj_raw,
        "llm_adjustment_clamped": adj_clamped,
        "llm_adjustment_valid": bool(valid),
        "needs_review": bool(needs_review or not valid or (evidence_ids and not set(evidence_ids).issubset(set(provided_evidence_ids)))),
        "at_limit": bool(at_limit or abs(adj_clamped) >= max_delta - 1e-12),
        "llm_evidence_ids": evidence_ids,
        "explanation": explanation,
        "confidence": conf,
    }


def main() -> int:
    args = parse_args()
    if not args.use_ollama:
        print("[warn] --use-ollama not set; will return zero adjustments with stubbed outputs.")
    rc = run_level3(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())