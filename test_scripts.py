# import os, requests, json
# OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# payload = {"model":"olmo-3", "prompt":"Hello, explain the model in short", "temperature":0.0}
# r = requests.post(OLLAMA_URL, json=payload, timeout=60)
# r.raise_for_status()
# try:
#   content_type = r.headers.get("content-type", "")
#   if content_type.startswith("application/json"):
#     try:
#         print(r.json())
#     except json.JSONDecodeError:
#         print(r.text)
#   else:
#     print(r.text)
#     print(r.json() if r.headers.get("content-type","").startswith("application/json") else r.text)
# except requests.exceptions.RequestException as e:
#     raise SystemExit(f"Request to {OLLAMA_URL} failed: {e}") from e

import duckdb
con = duckdb.connect("fraud_poc.db")  # or the path you use, or ':memory:' if in-memory
# adjust table/column names: id, prompt, model_response_raw, risk_score
print(con.execute("SELECT id, model, prompt, model_response_raw, response_text, risk_score FROM llm_results ORDER BY created_at DESC LIMIT 20").fetchall())