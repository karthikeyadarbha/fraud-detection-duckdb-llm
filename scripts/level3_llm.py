#!/usr/bin/env python3
"""
scripts/level3_llm.py

Level-3 LLM runner for fraud triage (LLaMA via Ollama CLI by default).

This is the updated version with robust handling of array-like cells in evidence/features columns
(to avoid ambiguous truth-value errors when pandas returns array-like results for isna/notna).

Usage (example):
  PYTHONPATH=. python scripts/level3_llm.py \
    --in artifacts/results_stream.parquet \
    --evidence-file artifacts/topk_evidence.parquet \
    --out_dir artifacts/level3_llm \
    --model llama2 \
    --max_delta 0.05 \
    --sample_limit 1000 \
    --use-ollama \
    --timeout 30
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# Helpers: prompt, call, parse
# ---------------------------

SYSTEM_INSTRUCTIONS = """
You are a strict fraud-triage assistant. Answer ONLY with a single valid JSON object that matches the schema exactly.

Schema:
{
  "transaction_id": "<string>",
  "llm_adjustment": <number>,       # additive delta to baseline score (range -0.10 .. +0.10). Will be clamped by caller to +/-{max_delta}
  "evidence_ids": ["E1","E2"],     # IDs referencing provided evidence items (must be subset of provided top_k_evidence ids)
  "explanation": "<string>",        # concise factual explanation based ONLY on provided evidence (<=250 chars)
  "confidence": <number>           # optional 0..1 numeric confidence
}

Rules:
- DO NOT output any text outside the JSON object (no markdown, no comments).
- Only reference evidence by id. evidence_ids MUST be a subset of the provided top_k_evidence ids.
- llm_adjustment must be a numeric additive delta; we will clamp it to +/- the configured max_delta.
- explanation must be concise and based solely on the evidence.
- Temperature must be 0 (deterministic).
- If you cannot produce a valid JSON that follows these rules, output the most faithful JSON indicating the reason, but keep the JSON schema shape.
"""

PROMPT_TEMPLATE = """
{system}

Input:
{input_json}

Respond with the JSON object only.
"""


def build_prompt(tx_row: Dict[str, Any], top_k_evidence: List[Dict[str, Any]], max_delta: float) -> str:
    # tx_row should include transaction_id, baseline_score, features (dict)
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


def call_ollama_generate(model: str, prompt: str, timeout: int = 30) -> str:
    """
    Robust Ollama caller: detect available Ollama subcommand and invoke it,
    piping the prompt via stdin. Returns stdout string or raises RuntimeError.
    Prefers 'run' (present on your system), falls back to 'generate' or 'chat'.
    """
    def _run_cmd(cmd):
        proc = subprocess.run(cmd, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        return proc.returncode, stdout, stderr

    # Probe ollama --help to find available subcommands (best-effort)
    try:
        help_proc = subprocess.run(["ollama", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        help_text = (help_proc.stdout.decode("utf-8", errors="replace") + help_proc.stderr.decode("utf-8", errors="replace")).lower()
    except FileNotFoundError:
        raise RuntimeError("ollama CLI not found in PATH. Install ollama or run with --no-ollama.")
    except Exception:
        help_text = ""

    # Preferred order for your environment (run is available)
    candidates = []
    if "run" in help_text:
        candidates.append(("run", ["ollama", "run", model]))
    if "generate" in help_text:
        candidates.append(("generate", ["ollama", "generate", model, "--temperature", "0"]))
    if "chat" in help_text:
        candidates.append(("chat", ["ollama", "chat", model, "--no-stream"]))

    # Always include fallback attempts in reasonable order if not detected
    for name, cmd in [("run", ["ollama", "run", model]),
                      ("generate", ["ollama", "generate", model, "--temperature", "0"]),
                      ("chat", ["ollama", "chat", model, "--no-stream"])]:
        if not any(name == c[0] for c in candidates):
            candidates.append((name, cmd))

    last_err = []
    for name, cmd in candidates:
        try:
            rc, out, err = _run_cmd(cmd)
            # Prefer returning stdout if available
            if rc == 0 and out:
                return out
            # Sometimes the command returns rc==0 but useful output is on stderr (rare). Capture errors to summarize.
            last_err.append((cmd, rc, err.strip() or out.strip()))
        except subprocess.TimeoutExpired:
            last_err.append((cmd, "timeout", "command timed out"))
        except Exception as e:
            last_err.append((cmd, "exc", str(e)))

    # No candidate succeeded
    err_lines = "\n".join([f"cmd={c} rc={r} err={e}" for (c, r, e) in last_err])
    raise RuntimeError(f"ollama run/generate/chat attempts failed. Details:\n{err_lines}")


def extract_first_json(s: str) -> Optional[str]:
    """
    Try to extract the first JSON object substring from s. Returns JSON string or None.
    Heuristic: find first '{' then find matching '}'.
    """
    if not isinstance(s, str):
        return None
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    end = None
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    return s[start : end + 1]


def parse_json_loose(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Try to load JSON from s directly, or extract a JSON substring if s contains extraneous text.
    Returns (parsed_obj or None, raw_json_str_or_None)
    """
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
    """
    Parse raw LLM text and apply validation & clamping, returning a dict with standardized fields.
    """
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

    # Validate fields
    txid = parsed.get("transaction_id")
    if txid != input_tx_id:
        out["parse_error"] = "txid_mismatch"
        return out

    # evidence ids
    eids = parsed.get("evidence_ids", [])
    if not isinstance(eids, list):
        out["parse_error"] = "evidence_ids_not_list"
        return out
    # ensure subset
    if not set(eids).issubset(set(topk_ids)):
        out["parse_error"] = "evidence_ids_not_subset_of_topk"
        return out
    out["llm_evidence_ids"] = eids

    # llm_adjustment
    adj = parsed.get("llm_adjustment")
    try:
        adj_val = float(adj)
    except Exception:
        out["parse_error"] = "llm_adjustment_not_numeric"
        return out
    out["llm_adjustment_raw"] = adj_val

    # explanation
    expl = parsed.get("explanation", "")
    if not isinstance(expl, str):
        out["parse_error"] = "explanation_not_string"
        return out
    out["explanation"] = expl[:1000]  # truncate to reasonable length

    # confidence (optional)
    conf = parsed.get("confidence", None)
    try:
        if conf is not None:
            conf_val = float(conf)
            out["confidence"] = max(0.0, min(1.0, conf_val))
    except Exception:
        out["confidence"] = None

    # All validations passed; clamp adjustment
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
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_path = out_dir / "llm_raw.jsonl"
    parsed_out_path = out_dir / "llm_parsed.parquet"
    manifest_path = out_dir / "manifest.json"

    # Read input dataframe
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

    # optional start/end slicing
    if start is not None or end is not None:
        df = df.iloc[start:end]

    # helper to detect missing scalar values safely (don't call pd.isna on array-likes)
    def _is_missing_scalar(x):
        if x is None:
            return True
        if isinstance(x, (list, tuple, pd.Series, np.ndarray)):
            return False
        try:
            return bool(pd.isna(x))
        except Exception:
            return False

    # Load evidence mapping if provided
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
            # try jsonl
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

                # If it's already a JSON string, try to parse it to list/dict
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
                        # Not JSON; treat as a single textual evidence item
                        evidence_map[tx] = [{"id": "E1", "text": val}]
                    continue

                # If it's list/tuple/ndarray/Series, coerce to list
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
                    # Fallback: coerce scalar to single evidence item
                    evidence_map[tx] = [{"id": "E1", "text": str(val)}]

    # Prepare loop
    results: List[LLMResultRow] = []
    total = len(df)
    print(f"[info] running Level-3 LLM on {total} rows; model={model}; max_delta={max_delta}")

    for idx, row in df.reset_index(drop=True).iterrows():
        try:
            txid = str(row.get("transaction_id"))
            baseline_score = row.get("baseline_score", None)

            # gather features for prompt (robust to array-like)
            features = None
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
                for c in df.columns:
                    if c in ("transaction_id", "baseline_score", topk_col):
                        continue
                    v = row.get(c)
                    # include simple scalars only
                    if isinstance(v, (int, float, str, bool)) and not isinstance(v, (list, tuple, pd.Series, np.ndarray)):
                        features[c] = v

            # --- Robust collect top_k evidence for this tx ---
            if evidence_map:
                top_k = evidence_map.get(txid, [])
            else:
                if topk_col and topk_col in df.columns:
                    raw_e = row.get(topk_col)
                    # handle missing / NaN values robustly
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
            # --- end robust collect ---

            # ensure evidence items have id+text
            canonical_topk: List[Dict[str, str]] = []
            for i, e in enumerate(top_k):
                if isinstance(e, dict) and "id" in e and "text" in e:
                    canonical_topk.append({"id": str(e["id"]), "text": str(e["text"])})
                else:
                    canonical_topk.append({"id": f"E{i+1}", "text": str(e)})

            prompt = build_prompt({"transaction_id": txid, "baseline_score": baseline_score, "features": features}, canonical_topk, max_delta)

            # call LLM
            start_t = time.time()
            if use_ollama:
                try:
                    raw_out = call_ollama_generate(model, prompt, timeout=timeout)
                except Exception as e:
                    raw_out = f'__OLLAMA_ERROR__ {repr(e)}'
            else:
                raw_out = "__NO_RUNNER__"

            elapsed = time.time() - start_t

            # parse and validate
            topk_ids = [e["id"] for e in canonical_topk]
            parsed_valid = parse_and_validate_llm_response(raw_out, txid, topk_ids, max_delta)

            # build result row
            rr = LLMResultRow(
                transaction_id=txid,
                baseline_score=(float(baseline_score) if baseline_score is not None else None),
                llm_adjustment_raw=(parsed_valid.get("llm_adjustment_raw") if parsed_valid.get("llm_adjustment_raw") is not None else None),
                llm_adjustment_clamped=float(parsed_valid.get("llm_adjustment_clamped", 0.0)),
                llm_adjustment_valid=bool(parsed_valid.get("llm_adjustment_valid", False)),
                llm_evidence_ids=list(parsed_valid.get("llm_evidence_ids", [])),
                explanation=parsed_valid.get("explanation"),
                confidence=parsed_valid.get("confidence"),
                needs_review=bool(parsed_valid.get("needs_review", True)),
                llm_raw_output=raw_out,
                llm_parsed_json=parsed_valid.get("parsed_json"),
                prompt_ref=f"prompt_tx_{txid}",
                timestamp_utc=datetime.utcnow().isoformat() + "Z",
            )

            # persist raw line
            raw_line = {
                "transaction_id": txid,
                "timestamp_utc": rr.timestamp_utc,
                "prompt": prompt,
                "raw_output": raw_out,
                "model": model,
                "elapsed_seconds": elapsed,
            }
            try:
                write_raw_jsonl(raw_out_path, raw_line)
            except Exception as e:
                print(f"[warn] failed to append raw jsonl: {e}")

            results.append(rr)

            # progress logging
            if (idx + 1) % 100 == 0:
                print(f"[info] processed {idx+1}/{total} rows (last tx={txid})")

        except KeyboardInterrupt:
            print("[info] interrupted by user")
            break
        except Exception as e:
            print(f"[error] row {idx} txid {row.get('transaction_id')} failed: {e}", file=sys.stderr)

    # build parsed DataFrame
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

    # write manifest
    manifest = {
        "model": model,
        "runner": "ollama" if use_ollama else "noop",
        "max_delta": max_delta,
        "sample_count": len(results),
        "created_at": datetime.utcnow().isoformat() + "Z",
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
    p.add_argument("--model", default="llama2", help="LLM model name for ollama generate")
    p.add_argument("--max_delta", type=float, default=0.05, help="Max absolute additive delta LLM may propose (will be clamped)")
    p.add_argument("--sample_limit", type=int, default=None, help="If set, only run on first N rows")
    # allow both flags to explicitly enable or disable Ollama
    p.add_argument("--use-ollama", dest="use_ollama", action="store_true", help="Enable Ollama CLI runner")
    p.add_argument("--no-ollama", dest="use_ollama", action="store_false", help="Disable Ollama CLI runner")
    p.set_defaults(use_ollama=False)  # default: disabled; explicit --use-ollama turns it on
    p.add_argument("--timeout", type=int, default=30, help="Timeout seconds for each LLM call")
    p.add_argument("--start", type=int, default=None, help="Start row index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="End row index (exclusive)")
    p.add_argument("--topk-col", dest="topk_col", default="top_k_evidence", help="Column name in input that contains top_k_evidence if no evidence-file provided")
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
    start = args.start
    end = args.end
    topk_col = args.topk_col

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
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())