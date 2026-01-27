#!/usr/bin/env python3
"""
scripts/level3_llm.py

Level-3 LLM runner for fraud triage (LLaMA via Ollama CLI by default).

Features:
- Tight SYSTEM_INSTRUCTIONS requiring strict JSON output (single or batched).
- Robust Ollama caller (prefers `ollama run`, falls back to other commands).
- Batch support via --batch-size to amortize startup latency.
- Excludes anomaly_score from features passed to the LLM.
- Validation, clamping to +/- max_delta, needs_review flag on failures.
- Persists llm_raw.jsonl, llm_parsed.parquet, and manifest.json with timezone-aware timestamps.
- Warm-up behavior: optional single warm-up call to Ollama to load model into memory before main loop.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# SYSTEM INSTRUCTIONS (tightened; batched-capable)
# ---------------------------

SYSTEM_INSTRUCTIONS = """
You are a strict fraud-triage assistant. RETURN EXACTLY valid JSON and NOTHING ELSE.
Do NOT output any prose, commentary, markup, or extra tokens outside the JSON. The JSON must match
the schema and types exactly.

When Input contains a single transaction, return exactly a single JSON object matching the schema.
When Input contains a transactions array (batched mode), return exactly a JSON array with one JSON
object per input transaction in the same order. Do NOT include extra text or commentary.

Schema (must match exactly for each object):
{
  "transaction_id": "<string>",
  "llm_adjustment": <number>,       // additive delta to baseline_score (numeric)
  "evidence_ids": ["E1","E2"],     // list of evidence IDs (must be subset of provided top_k_evidence ids)
  "explanation": "<string>",        // concise factual explanation based ONLY on the provided top_k_evidence and features (<=250 chars)
  "confidence": <number>            // optional (0.0 .. 1.0)
}

Strict rules:
- Return EXACTLY the required JSON (single object or array). No surrounding text, no code fences.
- evidence_ids MUST reference the ids exactly as provided in the top_k_evidence list for that transaction.
- llm_adjustment must be numeric. The caller will clamp to +/- {max_delta:.4f}.
- Keep explanation factual and based only on the provided top_k_evidence and features.
- Temperature must be 0 (deterministic). Be concise.
- If you cannot produce a valid JSON, return a JSON object (or an array of objects for batched input) following the schema shape and include a short explanation string describing the reason (still return valid JSON).
"""

PROMPT_TEMPLATE = """
{system}

Input:
{input_json}

Respond with the JSON (a single JSON object for single-input or an array of objects for batched input) only.
"""

# ---------------------------
# Helpers: prompt, call, parse
# ---------------------------


def build_prompt_single(tx_row: Dict[str, Any], top_k_evidence: List[Dict[str, Any]], max_delta: float) -> str:
    in_obj = {
        "transaction_id": tx_row.get("transaction_id"),
        "baseline_score": float(tx_row.get("baseline_score")) if tx_row.get("baseline_score") is not None else None,
        "features": tx_row.get("features", {}),
        "top_k_evidence": top_k_evidence,
        "max_delta": max_delta,
    }
    system = SYSTEM_INSTRUCTIONS.replace("{max_delta}", f"{max_delta:.4f}")
    payload = json.dumps(in_obj, ensure_ascii=False, indent=2)
    return PROMPT_TEMPLATE.format(system=system.strip(), input_json=payload)


def build_prompt_batch(tx_rows: List[Dict[str, Any]], max_delta: float) -> str:
    in_obj = {"transactions": tx_rows, "max_delta": max_delta}
    system = SYSTEM_INSTRUCTIONS.replace("{max_delta}", f"{max_delta:.4f}")
    payload = json.dumps(in_obj, ensure_ascii=False, indent=2)
    return PROMPT_TEMPLATE.format(system=system.strip(), input_json=payload)


def call_ollama_generate(model: str, prompt: str, timeout: int = 30) -> str:
    """
    Robust Ollama caller: prefer `ollama run <model>` (available in many CLI installs),
    fall back to generate/chat if present. Pipes the prompt to stdin and returns stdout.
    Raises RuntimeError with diagnostic details on failure.
    """
    def _run_cmd(cmd):
        proc = subprocess.run(cmd, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        return proc.returncode, stdout, stderr

    # Probe help text to detect available subcommands
    try:
        help_proc = subprocess.run(["ollama", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        help_text = (help_proc.stdout.decode("utf-8", errors="replace") + help_proc.stderr.decode("utf-8", errors="replace")).lower()
    except FileNotFoundError:
        raise RuntimeError("ollama CLI not found in PATH. Install ollama or run with --no-ollama.")
    except Exception:
        help_text = ""

    candidates = []
    if "run" in help_text:
        candidates.append(("run", ["ollama", "run", model]))
    if "generate" in help_text:
        candidates.append(("generate", ["ollama", "generate", model, "--temperature", "0"]))
    if "chat" in help_text:
        candidates.append(("chat", ["ollama", "chat", model, "--no-stream"]))

    # Add fallback attempts (in reasonable order)
    for name, cmd in [("run", ["ollama", "run", model]),
                      ("generate", ["ollama", "generate", model, "--temperature", "0"]),
                      ("chat", ["ollama", "chat", model, "--no-stream"])]:
        if not any(name == c[0] for c in candidates):
            candidates.append((name, cmd))

    last_err = []
    for name, cmd in candidates:
        try:
            rc, out, err = _run_cmd(cmd)
            if rc == 0 and out:
                return out
            last_err.append((cmd, rc, (err.strip() or out.strip())))
        except subprocess.TimeoutExpired:
            last_err.append((cmd, "timeout", "command timed out"))
        except Exception as e:
            last_err.append((cmd, "exc", str(e)))

    err_lines = "\n".join([f"cmd={c} rc={r} err={e}" for (c, r, e) in last_err])
    raise RuntimeError(f"ollama run/generate/chat attempts failed. Details:\n{err_lines}")


def extract_first_json(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    start = s.find("{")
    if start == -1:
        start = s.find("[")
        if start == -1:
            return None
    depth = 0
    end = None
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{" or ch == "[":
            depth += 1
        elif ch == "}" or ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    return s[start:end + 1]


def parse_json_loose(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(s, str):
        return None, None
    try:
        obj = json.loads(s)
        return obj, s
    except Exception:
        js = extract_first_json(s)
        if js is None:
            return None, None
        try:
            obj = json.loads(js)
            return obj, js
        except Exception:
            return None, js


# ---------------------------
# Validation & clamping
# ---------------------------


def parse_and_validate_llm_response(
    raw_text: str,
    input_tx_id: str,
    topk_ids: List[str],
    max_delta: float,
) -> Dict[str, Any]:
    out = {
        "transaction_id": input_tx_id,
        "llm_adjustment_raw": None,
        "llm_adjustment_clamped": 0.0,
        "llm_adjustment_valid": False,
        "llm_evidence_ids": [],
        "explanation": None,
        "confidence": None,
        "parsed_json": None,
        "parse_error": None,
        "needs_review": True,
    }

    parsed, parsed_str = parse_json_loose(raw_text)
    if parsed is None:
        out["parse_error"] = "invalid_json_or_no_json_found"
        out["parsed_json"] = None
        return out

    out["parsed_json"] = parsed

    txid = parsed.get("transaction_id")
    if txid != input_tx_id:
        out["parse_error"] = "txid_mismatch"
        return out

    eids = parsed.get("evidence_ids", [])
    if not isinstance(eids, list):
        out["parse_error"] = "evidence_ids_not_list"
        return out
    if not set(eids).issubset(set(topk_ids)):
        out["parse_error"] = "evidence_ids_not_subset_of_topk"
        return out
    out["llm_evidence_ids"] = eids

    adj = parsed.get("llm_adjustment")
    try:
        adj_val = float(adj)
    except Exception:
        out["parse_error"] = "llm_adjustment_not_numeric"
        return out
    out["llm_adjustment_raw"] = adj_val

    expl = parsed.get("explanation", "")
    if not isinstance(expl, str):
        out["parse_error"] = "explanation_not_string"
        return out
    out["explanation"] = expl[:1000]

    conf = parsed.get("confidence", None)
    try:
        if conf is not None:
            conf_val = float(conf)
            out["confidence"] = max(0.0, min(1.0, conf_val))
    except Exception:
        out["confidence"] = None

    clamped = max(-max_delta, min(max_delta, out["llm_adjustment_raw"]))
    out["llm_adjustment_clamped"] = float(clamped)
    out["llm_adjustment_valid"] = True
    out["needs_review"] = False
    return out


# ---------------------------
# I/O & orchestrator
# ---------------------------


@dataclass
class LLMResultRow:
    transaction_id: str
    baseline_score: Optional[float]
    llm_adjustment_raw: Optional[float]
    llm_adjustment_clamped: float
    llm_adjustment_valid: bool
    llm_evidence_ids: List[str]
    explanation: Optional[str]
    confidence: Optional[float]
    needs_review: bool
    llm_raw_output: str
    llm_parsed_json: Optional[Dict[str, Any]]
    prompt_ref: str
    timestamp_utc: str


def write_raw_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk


def run_level3(
    input_path: Path,
    evidence_path: Optional[Path],
    out_dir: Path,
    model: str,
    max_delta: float,
    sample_limit: Optional[int],
    use_ollama: bool,
    timeout: int,
    start: Optional[int],
    end: Optional[int],
    topk_col: str,
    batch_size: int = 1,
    warmup: bool = True,
    warmup_timeout: int = 300,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_path = out_dir / "llm_raw.jsonl"
    parsed_out_path = out_dir / "llm_parsed.parquet"
    manifest_path = out_dir / "manifest.json"

    if not input_path.exists():
        print(f"[error] input file {input_path} not found", file=sys.stderr)
        return 2
    suf = input_path.suffix.lower()
    if suf in (".parquet", ".pq"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if sample_limit:
        df = df.head(sample_limit)

    if start is not None or end is not None:
        df = df.iloc[start:end]

    def _is_missing_scalar(x):
        if x is None:
            return True
        if isinstance(x, (list, tuple, pd.Series, np.ndarray)):
            return False
        try:
            return bool(pd.isna(x))
        except Exception:
            return False

    evidence_map: Dict[str, List[Dict[str, Any]]] = {}
    if evidence_path:
        if not evidence_path.exists():
            print(f"[error] evidence file {evidence_path} not found", file=sys.stderr)
            return 3
        suf2 = evidence_path.suffix.lower()
        if suf2 in (".parquet", ".pq"):
            ev_df = pd.read_parquet(evidence_path)
        elif suf2 in (".csv", ".txt"):
            ev_df = pd.read_csv(evidence_path)
        else:
            try:
                with open(evidence_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        o = json.loads(line)
                        tx = str(o.get("transaction_id"))
                        evidence_map[tx] = o.get("top_k_evidence", [])
                ev_df = None
            except Exception:
                ev_df = None
        if 'ev_df' in locals() and ev_df is not None:
            for _, r in ev_df.iterrows():
                tx = str(r.get("transaction_id"))
                val = r.get("top_k_evidence", None)

                if _is_missing_scalar(val):
                    evidence_map[tx] = []
                    continue

                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (list, tuple)):
                            evidence_map[tx] = list(parsed)
                        elif isinstance(parsed, dict):
                            evidence_map[tx] = [parsed]
                        else:
                            evidence_map[tx] = [{"id": "E1", "text": str(parsed)}]
                    except Exception:
                        evidence_map[tx] = [{"id": "E1", "text": val}]
                    continue

                if isinstance(val, (list, tuple)):
                    evidence_map[tx] = list(val)
                elif isinstance(val, (pd.Series, np.ndarray)):
                    try:
                        evidence_map[tx] = list(val.tolist())
                    except Exception:
                        evidence_map[tx] = [{"id": "E1", "text": str(val)}]
                elif isinstance(val, dict):
                    evidence_map[tx] = [val]
                else:
                    evidence_map[tx] = [{"id": "E1", "text": str(val)}]

    results: List[LLMResultRow] = []
    total = len(df)
    print(f"[info] running Level-3 LLM on {total} rows; model={model}; max_delta={max_delta}; batch_size={batch_size}")

    # Optional warm-up: run a single request to load model into memory (does not add to parsed results)
    if use_ollama and warmup and total > 0:
        try:
            # Build a tiny warm-up prompt using the first row's minimal context if available
            first_row = df.reset_index(drop=True).iloc[0]
            txid = str(first_row.get("transaction_id"))
            # extract a minimal top_k for warmup
            if evidence_map:
                top_k = evidence_map.get(txid, [])[:1]
            else:
                top_k = []
                if topk_col in df.columns:
                    raw_e = first_row.get(topk_col)
                    if not _is_missing_scalar(raw_e):
                        try:
                            parsed = json.loads(raw_e) if isinstance(raw_e, str) else raw_e
                            if isinstance(parsed, (list, tuple)):
                                top_k = parsed[:1]
                            elif isinstance(parsed, dict):
                                top_k = [parsed]
                        except Exception:
                            top_k = [{"id": "E1", "text": str(raw_e)}]
            # warmup prompt (single)
            warm_prompt = build_prompt_single({"transaction_id": "WARMUP", "baseline_score": 0.0, "features": {}}, top_k, max_delta)
            start_w = time.time()
            try:
                warm_raw = call_ollama_generate(model, warm_prompt, timeout=warmup_timeout)
            except Exception as e:
                warm_raw = f"__OLLAMA_WARMUP_ERROR__ {repr(e)}"
            warm_elapsed = time.time() - start_w
            warm_line = {
                "transaction_id": "WARMUP",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "prompt": "<warmup prompt>",
                "raw_output": warm_raw,
                "model": model,
                "elapsed_seconds": warm_elapsed,
            }
            try:
                write_raw_jsonl(raw_out_path, warm_line)
            except Exception:
                pass
            print(f"[info] warmup completed, elapsed={warm_elapsed:.2f}s")
        except Exception as e:
            print(f"[warn] warmup failed: {e}")

    # Build per-row data in memory first for easier batching
    rows_for_batching = []
    for idx, row in df.reset_index(drop=True).iterrows():
        txid = str(row.get("transaction_id"))
        baseline_score = row.get("baseline_score", None)

        # gather features for prompt (robust to array-like); DO NOT include anomaly_score
        raw_features_cell = row.get("features", None) if "features" in row else None
        if not _is_missing_scalar(raw_features_cell):
            val = raw_features_cell
            if isinstance(val, str):
                try:
                    features = json.loads(val)
                except Exception:
                    features = {"raw_features": val}
            elif isinstance(val, (dict, list)):
                features = val
            else:
                features = {"features": val}
        else:
            features = {}
            # exclude anomaly_score from features passed to LLM
            for c in df.columns:
                if c in ("transaction_id", "baseline_score", topk_col, "anomaly_score"):
                    continue
                v = row.get(c)
                # include only simple scalar fields
                if isinstance(v, (int, float, str, bool)) and not isinstance(v, (list, tuple, pd.Series, np.ndarray)):
                    features[c] = v

        # If the features object contains anomaly_score, remove it explicitly
        if isinstance(features, dict) and "anomaly_score" in features:
            features.pop("anomaly_score", None)

        if evidence_map:
            top_k = evidence_map.get(txid, [])
        else:
            if topk_col and topk_col in df.columns:
                raw_e = row.get(topk_col)
                is_missing = _is_missing_scalar(raw_e)

                if is_missing:
                    top_k = []
                else:
                    if isinstance(raw_e, str):
                        try:
                            parsed = json.loads(raw_e)
                            if isinstance(parsed, (list, tuple)):
                                top_k = list(parsed)
                            elif isinstance(parsed, dict):
                                top_k = [parsed]
                            else:
                                top_k = [{"id": "E1", "text": str(parsed)}]
                        except Exception:
                            top_k = [{"id": "E1", "text": raw_e}]
                    elif isinstance(raw_e, (list, tuple)):
                        top_k = list(raw_e)
                    elif isinstance(raw_e, (pd.Series, np.ndarray)):
                        try:
                            top_k = list(raw_e.tolist())
                        except Exception:
                            top_k = [{"id": "E1", "text": str(raw_e)}]
                    elif isinstance(raw_e, dict):
                        top_k = [raw_e]
                    else:
                        top_k = [{"id": "E1", "text": str(raw_e)}]
            else:
                top_k = []

        canonical_topk: List[Dict[str, str]] = []
        for i, e in enumerate(top_k):
            if isinstance(e, dict) and "id" in e and "text" in e:
                canonical_topk.append({"id": str(e["id"]), "text": str(e["text"])})
            else:
                canonical_topk.append({"id": f"E{i+1}", "text": str(e)})

        rows_for_batching.append({
            "transaction_id": txid,
            "baseline_score": baseline_score,
            "features": features,
            "top_k_evidence": canonical_topk,
            "row_index": idx
        })

    # Now process in batches
    processed = 0
    for batch in chunked_iterable(rows_for_batching, batch_size):
        # Build prompt: single or batched
        if len(batch) == 1:
            prompt = build_prompt_single(
                {
                    "transaction_id": batch[0]["transaction_id"],
                    "baseline_score": batch[0]["baseline_score"],
                    "features": batch[0]["features"],
                },
                batch[0]["top_k_evidence"],
                max_delta,
            )
        else:
            tx_rows_for_prompt = []
            for b in batch:
                tx_rows_for_prompt.append({
                    "transaction_id": b["transaction_id"],
                    "baseline_score": (float(b["baseline_score"]) if b["baseline_score"] is not None else None),
                    "features": b["features"],
                    "top_k_evidence": b["top_k_evidence"]
                })
            prompt = build_prompt_batch(tx_rows_for_prompt, max_delta)

        start_t = time.time()
        if use_ollama:
            try:
                raw_out = call_ollama_generate(model, prompt, timeout=timeout)
            except Exception as e:
                raw_out = f'__OLLAMA_ERROR__ {repr(e)}'
        else:
            raw_out = "__NO_RUNNER__"
        elapsed = time.time() - start_t

        # Parse the raw_out. Expected: single JSON obj or list of objs
        parsed_top, parsed_str = parse_json_loose(raw_out)
        # If parsed_top is a list, map one-to-one by order; if dict and batch=1, map; if dict and batch>1, try to map by txid keys
        mapped_results = {}
        if isinstance(parsed_top, list):
            # map by order - if lengths match map by order, otherwise try mapping by transaction_id
            if len(parsed_top) == len(batch):
                for i, obj in enumerate(parsed_top):
                    mapped_results[batch[i]["transaction_id"]] = (obj, json.dumps(obj, ensure_ascii=False))
            else:
                for obj in parsed_top:
                    if isinstance(obj, dict) and "transaction_id" in obj:
                        mapped_results[str(obj["transaction_id"])] = (obj, json.dumps(obj, ensure_ascii=False))
        elif isinstance(parsed_top, dict):
            # Single dict returned
            if len(batch) == 1:
                mapped_results[batch[0]["transaction_id"]] = (parsed_top, json.dumps(parsed_top, ensure_ascii=False))
            else:
                # Maybe returned an object keyed by txid
                for k, v in parsed_top.items():
                    if isinstance(v, dict):
                        mapped_results[str(k)] = (v, json.dumps(v, ensure_ascii=False))

        # For any tx in batch not mapped, we'll set raw parsed outcome to the entire raw_out (parser will try to extract)
        for b in batch:
            txid = b["transaction_id"]
            topk_ids = [e["id"] for e in b["top_k_evidence"]]
            if txid in mapped_results:
                obj, obj_str = mapped_results[txid]
                raw_text_for_tx = obj_str
            else:
                raw_text_for_tx = raw_out

            parsed_valid = parse_and_validate_llm_response(raw_text_for_tx, txid, topk_ids, max_delta)

            rr = LLMResultRow(
                transaction_id=txid,
                baseline_score=(float(b["baseline_score"]) if b["baseline_score"] is not None else None),
                llm_adjustment_raw=(parsed_valid.get("llm_adjustment_raw") if parsed_valid.get("llm_adjustment_raw") is not None else None),
                llm_adjustment_clamped=float(parsed_valid.get("llm_adjustment_clamped", 0.0)),
                llm_adjustment_valid=bool(parsed_valid.get("llm_adjustment_valid", False)),
                llm_evidence_ids=list(parsed_valid.get("llm_evidence_ids", [])),
                explanation=parsed_valid.get("explanation"),
                confidence=parsed_valid.get("confidence"),
                needs_review=bool(parsed_valid.get("needs_review", True)),
                llm_raw_output=raw_out,
                llm_parsed_json=parsed_valid.get("parsed_json"),
                prompt_ref=f"prompt_batch_{processed // batch_size}",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )

            raw_line = {
                "transaction_id": txid,
                "timestamp_utc": rr.timestamp_utc,
                "prompt": prompt if len(batch) == 1 else f"<batched prompt; batch_size={len(batch)}>",
                "raw_output": raw_out,
                "model": model,
                "elapsed_seconds": elapsed,
            }
            try:
                write_raw_jsonl(raw_out_path, raw_line)
            except Exception as e:
                print(f"[warn] failed to append raw jsonl: {e}")

            results.append(rr)

        processed += len(batch)
        if processed % 100 == 0 or processed == total:
            print(f"[info] processed {processed}/{total} rows (last batch size={len(batch)})")

    parsed_rows = []
    for r in results:
        parsed_rows.append({
            "transaction_id": r.transaction_id,
            "baseline_score": r.baseline_score,
            "llm_adjustment_raw": r.llm_adjustment_raw,
            "llm_adjustment_clamped": r.llm_adjustment_clamped,
            "llm_adjustment_valid": r.llm_adjustment_valid,
            "llm_evidence_ids": r.llm_evidence_ids,
            "explanation": r.explanation,
            "confidence": r.confidence,
            "needs_review": r.needs_review,
            "llm_raw_output": r.llm_raw_output,
            "llm_parsed_json": r.llm_parsed_json,
            "prompt_ref": r.prompt_ref,
            "timestamp_utc": r.timestamp_utc,
        })
    if parsed_rows:
        pd.DataFrame(parsed_rows).to_parquet(parsed_out_path, index=False)
        print(f"[info] wrote parsed outputs to {parsed_out_path}")
    else:
        print("[warn] no parsed rows to write")

    manifest = {
        "model": model,
        "runner": "ollama" if use_ollama else "noop",
        "max_delta": max_delta,
        "sample_count": len(results),
        "batch_size": batch_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[info] wrote manifest to {manifest_path}")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Level-3 LLM triage and produce validated adjustments (artifacts)")
    p.add_argument("--in", dest="input", required=True, help="Input parquet/csv with transaction_id & baseline_score")
    p.add_argument("--evidence-file", dest="evidence", default=None, help="Optional evidence file (parquet/csv/jsonl) mapping transaction_id->top_k_evidence")
    p.add_argument("--out_dir", default="artifacts/level3_llm", help="Output directory for LLM artifacts")
    p.add_argument("--model", default="llama2", help="LLM model name for ollama run")
    p.add_argument("--max_delta", type=float, default=0.05, help="Max absolute additive delta LLM may propose (will be clamped)")
    p.add_argument("--sample_limit", type=int, default=None, help="If set, only run on first N rows")
    p.add_argument("--use-ollama", dest="use_ollama", action="store_true", help="Enable Ollama CLI runner")
    p.add_argument("--no-ollama", dest="use_ollama", action="store_false", help="Disable Ollama CLI runner")
    p.set_defaults(use_ollama=False)
    p.add_argument("--timeout", type=int, default=120, help="Timeout seconds for each LLM call (increased default to tolerate cold starts)")
    p.add_argument("--warmup-timeout", type=int, default=300, help="Timeout seconds for the warmup call (longer)")
    p.add_argument("--no-warmup", dest="warmup", action="store_false", help="Disable warmup call")
    p.add_argument("--warmup", dest="warmup", action="store_true", help="Enable warmup call (default)")
    p.set_defaults(warmup=True)
    p.add_argument("--start", type=int, default=None, help="Start row index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="End row index (exclusive)")
    p.add_argument("--topk-col", dest="topk_col", default="top_k_evidence", help="Column name in input that contains top_k_evidence if no evidence-file provided")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=1, help="Number of transactions to batch per LLM call (>=1). Batching amortizes latency.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    evidence_path = Path(args.evidence) if args.evidence else None
    out_dir = Path(args.out_dir)
    model = args.model
    max_delta = float(args.max_delta)
    sample_limit = int(args.sample_limit) if args.sample_limit else None
    use_ollama = bool(args.use_ollama)
    timeout = int(args.timeout)
    warmup_timeout = int(args.warmup_timeout)
    start = args.start
    end = args.end
    topk_col = args.topk_col
    batch_size = max(1, int(args.batch_size))
    warmup = bool(args.warmup)

    rc = run_level3(
        input_path=in_path,
        evidence_path=evidence_path,
        out_dir=out_dir,
        model=model,
        max_delta=max_delta,
        sample_limit=sample_limit,
        use_ollama=use_ollama,
        timeout=timeout,
        start=start,
        end=end,
        topk_col=topk_col,
        batch_size=batch_size,
        warmup=warmup,
        warmup_timeout=warmup_timeout,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())