"""
Lightweight LLM adapter for Level 3: builds prompt, calls local Ollama (or a stub),
validates JSON output, enforces guardrails and returns a dict:

{
  "explanation": str,
  "evidence_ids": [ ... ],
  "llm_adjustment": float,
  "confidence": float,
  "needs_review": bool
}

Guardrails:
- temperature=0
- evidence_ids must be subset of provided topk ids
- clamp llm_adjustment to [-delta, +delta]
- persist prompt_hash
"""
import json
import hashlib
import subprocess
from typing import List, Dict, Any

DEFAULT_DELTA = 0.2  # max adjustment allowed


def build_prompt(transaction: Dict[str, Any], features: Dict[str, Any], baseline_score: float, topk_evidence: List[Dict[str, Any]]) -> str:
    evidence_lines = []
    for e in topk_evidence:
        # each evidence should have id and brief text
        eid = e.get('id')
        txt = e.get('text', '')
        evidence_lines.append(f"{eid} | {txt}")
    evidence_text = "\n".join(evidence_lines)

    prompt = f"""
You are a fraud triage assistant. Given the transaction, a baseline fraud probability, and the top-K evidence items,
produce a concise human-readable explanation and a bounded numeric adjustment to the baseline_score.

Transaction ID: {transaction.get('transaction_id')}
Baseline score: {baseline_score:.4f}
Features: {json.dumps(features)}
Top evidence (id | summary):
{evidence_text}

INSTRUCTIONS:
Return ONLY a single JSON object with keys:
- explanation (string, short)
- evidence_ids (array of evidence ids; MUST be a subset of provided top-K ids)
- llm_adjustment (float; a signed delta to add to baseline_score; MUST be between -{DEFAULT_DELTA} and {DEFAULT_DELTA})
- confidence (float between 0 and 1)
- needs_review (boolean; true if LLM cannot confidently follow instructions)

Set temperature to 0 when calling the model.
"""
    return prompt.strip()


def prompt_hash(prompt: str, topk_ids: List[str]) -> str:
    s = prompt + json.dumps(sorted(topk_ids), sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def call_ollama(prompt: str, model: str = "llama") -> str:
    """
    Example using local ollama CLI. Replace or adapt if you call OpenAI or other provider.
    Returns raw text output.
    """
    try:
        cmd = ["ollama", "generate", model, "--temperature", "0", prompt]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return proc.stdout
    except FileNotFoundError:
        # Ollama not installed; raise a clear error
        raise RuntimeError("Ollama CLI not found. Install Ollama or replace call_ollama with your LLM call.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LLM call failed: {e.stderr or e.stdout}")


def validate_and_parse(raw_output: str, topk_ids: List[str], delta_max: float = DEFAULT_DELTA) -> Dict[str, Any]:
    # Attempt to parse JSON from raw_output. Accept raw_output that may contain surrounding text by finding first {..}
    start = raw_output.find('{')
    end = raw_output.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return {"needs_review": True, "explanation": "invalid_json", "evidence_ids": [], "llm_adjustment": 0.0, "confidence": 0.0}

    json_blob = raw_output[start:end+1]
    try:
        payload = json.loads(json_blob)
    except Exception:
        return {"needs_review": True, "explanation": "invalid_json_parse", "evidence_ids": [], "llm_adjustment": 0.0, "confidence": 0.0}

    # Validate keys
    explanation = str(payload.get('explanation','')).strip()
    evidence_ids = payload.get('evidence_ids', [])
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    # evidence subset check
    if not set(evidence_ids).issubset(set(topk_ids)):
        return {"needs_review": True, "explanation": "evidence_ids_not_subset", "evidence_ids": [], "llm_adjustment": 0.0, "confidence": 0.0}

    try:
        adj = float(payload.get('llm_adjustment', 0.0))
    except Exception:
        adj = 0.0
    # clamp
    adj = max(-delta_max, min(delta_max, adj))
    conf = float(payload.get('confidence', 0.0))
    needs_review = bool(payload.get('needs_review', False))

    return {
        "explanation": explanation,
        "evidence_ids": evidence_ids,
        "llm_adjustment": adj,
        "confidence": max(0.0, min(1.0, conf)),
        "needs_review": needs_review
    }


def run_llm_for_transaction(transaction: Dict, features: Dict, baseline_score: float, topk_evidence: List[Dict], model: str="llama"):
    prompt = build_prompt(transaction, features, baseline_score, topk_evidence)
    phash = prompt_hash(prompt, [e['id'] for e in topk_evidence])
    raw = call_ollama(prompt, model=model)
    parsed = validate_and_parse(raw, [e['id'] for e in topk_evidence])
    # attach provenance
    parsed['_prompt'] = prompt
    parsed['_raw'] = raw
    parsed['_prompt_hash'] = phash
    return parsed