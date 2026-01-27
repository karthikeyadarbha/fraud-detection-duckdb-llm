#!/usr/bin/env python3
import json
from pathlib import Path

RAW = Path("artifacts/level3_llm_run/llm_raw.jsonl")

def main():
    if not RAW.exists():
        print(f"Missing {RAW}"); return
    successes, errors = [], []
    for l in RAW.read_text().splitlines():
        if not l.strip(): continue
        try:
            j = json.loads(l)
        except Exception:
            continue
        ro = j.get("raw_output","")
        if isinstance(ro,str) and ro.startswith("__OLLAMA_ERROR__"):
            errors.append({
                "transaction_id": j.get("transaction_id"),
                "elapsed_seconds": j.get("elapsed_seconds"),
                "error": ro[:400],
            })
        else:
            successes.append(j)
        if len(successes)>=2 and len(errors)>=5: break

    print("=== Successes (up to 2) ===")
    for s in successes[:2]:
        print(json.dumps(s, indent=2, ensure_ascii=False))
    print("\n=== Errors (up to 5) ===")
    for e in errors[:5]:
        print(json.dumps(e, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()