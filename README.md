# Fraud Detection PoC — DuckDB + Local LLM (Ollama)

This repository is a proof‑of‑concept for scoring transaction fraud risk using a DuckDB-backed pipeline and a local LLM (Ollama). It includes end‑to‑end code for ingestion, robust streaming assembly of LLM outputs, deterministic feature computation, evidence retrieval/verification, batching/parallel processing, repair/reprocessing flows, and unit tests.

This README reflects the latest merged changes:
- schema migration: prompt + model metadata on `llm_results`, `llm_features` table
- deterministic features pipeline (amount_z, velocity)
- evidence verification utilities (validate LLM evidence against retrieved top‑K)
- updated notebook with batching/parallel/reprocess flows
- unit tests and runner to apply migration + compute features

---

## Table of contents

- What this project does
- What's new (latest merge)
- Architecture / data flow
- Data attributes & metadata considered by the LLM
- Metadata / signals that typically indicate HIGH risk
- Prompt & JSON format example
- Scoring approach (recommended hybrid)
- Best practices & anti-hallucination checks
- Prerequisites
- Quick start & execution steps
- Files added/updated in the latest merge
- Troubleshooting & common errors
- Security & privacy notes
- Next steps

---

## What this project does

- Scores transactions with a numeric `risk_score` (0.0–1.0) using:
  - Deterministic features (amount zscore, velocity, etc.),
  - Retrieval signals (top‑K similar historical records),
  - An LLM that produces/explains/adjusts the numeric score.
- Persists LLM outputs, prompts, parsed JSON, provenance metadata, and computed features in DuckDB.
- Supports batch and parallel ingestion modes, robust stream assembly, and a repair flow to reprocess missing/NaN results.
- Verifies evidence returned by the LLM against retrieved neighbors and flags invalid outputs (`needs_review`).

---

## What's new (latest merge)

- Added migration to extend `llm_results` with:
  - `prompt_text`, `prompt_hash`, `model_name`, `model_version`, `model_params`.
- Added `llm_features` table with deterministic features:
  - `amount_z`, `velocity_1h`, `velocity_24h`, `fraud_fraction_topk`, `mean_similarity_topk`, `computed_at`.
- Added `src/data/features.py` — compute/persist features.
- Added `src/llm/verify.py` — evidence verification utilities.
- Notebook updated to run migrations, compute features, call LLM, verify evidence and persist metadata.
- Unit tests added for parsing and evidence verification.

---

## Architecture / data flow (short)

1. Transactions in `transactions` table.
2. Deterministic features computed and persisted into `llm_features`.
3. For each transaction (or batch):
   - Retrieve top‑K similar historical records (embedding retrieval; placeholder available).
   - Build prompt with structured features + neighbor snippets.
   - Call Ollama (stream or non-stream) to get response.
   - Assemble the stream, parse JSON → `{ risk_score, confidence, explanation, evidence_tx_ids }`.
   - Verify evidence IDs are subset of retrieved top‑K; if not, set `needs_review`.
   - Persist `llm_results` row with prompt metadata, parsed_response, `risk_score`, and flags.

---

## Data attributes & metadata the LLM should be given (recommended)

Include the following raw fields and derived features in the prompt (structure them clearly):

Core transactional attributes (raw fields)
- tx_id (string)
- account_id / customer_id
- card_bin (first 6 digits) or hashed PAN (never send full PAN)
- timestamp / created_at (ISO8601)
- amount (numeric)
- currency (ISO code)
- merchant_id / merchant_name
- merchant_category_code (MCC)
- description / merchant_text (free text)
- terminal_id / pos_id
- ip_address (or hashed), country from geo IP
- billing_country / shipping_country
- card_present (bool)
- auth_result / avs_result / cvv_result
- channel (web/mobile/POS)
- device_id / device_fingerprint / user_agent
- previous_disposition / historical_label for account or card

Deterministic / engineered features (high‑signal)
- amount_z (z-score by account)
- amount_percentile
- velocity_1h, velocity_24h (counts)
- unique_merchant_count_24h
- failed_auth_count_recent
- new_device_flag (bool)
- avg_transaction_interval
- fraud_fraction_topk (fraction of top-K neighbors labeled fraud)
- mean_similarity_topk (embedding similarity mean)
- merchant_risk_score (merchant-level historical fraud rate)
- recency of top-K neighbors (age_days) and weighted fraud signal

Retrieval context
- topK_neighbors array: each neighbor = { tx_id, similarity, label, age_days, amount, short_snippet }
- This list is given so the LLM can reference evidence; evidence IDs returned by the LLM will be validated against this list.

Provenance & metadata (must persist)
- prompt_text, prompt_hash (sha256 of prompt + params), model_name, model_version, model_params
- raw_lines (stream chunks), assembled `llm_response`
- parsed_response (json object returned by LLM)
- risk_score (numeric), confidence (numeric), needs_review (bool)

---

## Metadata / signals that typically indicate HIGH risk

Common patterns that strongly raise risk_score (heuristics to start with):
- Large outlier amount
  - amount_z ≥ 3 or amount_percentile ≥ 0.99
- High velocity
  - velocity_1h ≥ 5 or velocity_24h ≥ 10 (domain dependent)
- High similarity to labeled fraud
  - fraud_fraction_topk ≥ 0.6 or weighted_fraud ≥ 0.5
- Impossible travel (geo mismatch + short time delta)
- New device combined with high amount and multiple failed auth attempts
- Merchant with historically high fraud rate
- Multiple declines then a success (card-testing pattern)
- Suspicious IP (proxy/Tor/blacklisted ASN)
- LLM returns fabricated evidence IDs (not in topK) — treat as reduced confidence and set `needs_review`

Suggested action mapping (example)
- risk_score ≥ 0.90  → auto‑block / decline (high confidence)
- 0.50 ≤ risk_score < 0.90 → manual review / escalate
- risk_score < 0.50 → allow (but monitor)

Tune thresholds on labeled data.

---

## Prompt & JSON output example (strict format)

Prompt must place structured features first and instruct the LLM to output only JSON. Example JSON payload to embed in prompt:

```json
{
  "tx": {
    "tx_id": "tx_123",
    "account_id": "acct_456",
    "timestamp": "2026-01-08T15:06:22Z",
    "amount": 1250.00,
    "currency": "USD",
    "merchant_id": "m_789",
    "merchant_category": "electronics",
    "card_bin": "123456",
    "channel": "web",
    "device_new": true,
    "ip_country": "US",
    "billing_country": "US",
    "auth_result": "approved"
  },
  "features": {
    "amount_z": 3.7,
    "amount_percentile": 0.995,
    "velocity_1h": 6,
    "velocity_24h": 12,
    "failed_auth_count_recent": 3,
    "fraud_fraction_top5": 0.6,
    "mean_similarity_top5": 0.78,
    "merchant_risk_score": 0.4
  },
  "topk_neighbors": [
    {"tx_id":"old_100","similarity":0.82,"label":"fraud","age_days":10,"amount":1400},
    {"tx_id":"old_203","similarity":0.79,"label":"fraud","age_days":45,"amount":1200},
    {"tx_id":"old_299","similarity":0.63,"label":"legit","age_days":200,"amount":800}
  ],
  "instructions": "Return ONLY this exact JSON object: {\"risk_score\": <number 0..1>, \"confidence\": <number 0..1>, \"explanation\": \"<=200 chars\", \"evidence_tx_ids\": [list of tx_id strings] }"
}
```

Important: set `temperature=0.0` and include 1–2 few-shot examples in the prompt mapping feature patterns → risk scores to reduce hallucination.

---

## Scoring approach (recommended hybrid)

- Compute deterministic baseline `score0` from engineered features (e.g., weighted combination of fraud_fraction_topk, amount_z, velocity).
- Pass `score0` and evidence to LLM and ask it to return either:
  - a final `risk_score` (constrained to ±delta from `score0`), or
  - an adjustment value (e.g., +0.05) and explanation.
- Calibrate final scores using labeled data (Platt scaling or isotonic regression) to ensure empirical probabilities align with predicted scores.

---

## Best practices & anti-hallucination checks

- Evidence verification: ensure `evidence_tx_ids` returned by LLM are subset of the retrieved top‑K. If not, set `needs_review = TRUE`.
- Numeric validation: parse & clamp `risk_score` to [0,1]. If missing/NaN, fallback to `score0` and flag for review.
- Persist prompt_text, prompt_hash, model metadata and `raw_lines` for audit and reprocessing.
- Use strict prompt phrasing: "OUTPUT ONLY JSON between markers <<<JSON>>> ... <<<ENDJSON>>>."
- Redact PII in prompts or store redacted prompts and hashed identifiers.
- Monitor `needs_review` rate and JSON parse success rate; tune prompt and batch size accordingly.

---

## Prerequisites

- Python 3.9+
- DuckDB (Python package)
- pandas, numpy, requests, pytest
- Ollama server/CLI if using Ollama locally (not a pip package)
- (Optional) gh CLI for PR automation

---

## Quick start & execution steps

1. Clone repo:
   ```bash
   git clone https://github.com/karthikeyadarbha/fraud-detection-duckdb-llm.git
   cd fraud-detection-duckdb-llm
   ```

2. Create and activate venv; install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Place your DuckDB file `fraud_poc.duckdb` in repo root or set:
   ```bash
   export FRAUD_DB_PATH=/full/path/to/fraud_poc.duckdb
   ```

4. Backup DB:
   ```bash
   cp $FRAUD_DB_PATH ${FRAUD_DB_PATH}.bak
   ```

5. Apply migration and compute features:
   ```bash
   python scripts/run_migration_and_features.py
   ```
   - This runs `scripts/migrations/001_add_metadata_and_features.sql` and computes features into `llm_features`.

6. Run the notebook:
   - Open `notebooks/fraud_duckdb_poc.ipynb` and run cells in sequence. Configure `OLLAMA_URL`, `LLM_MODEL`, `PROCESS_MODE`, `BATCH_SIZE` in the CONFIG cell.

7. Run unit tests:
   ```bash
   pytest -q tests/test_parse_and_verify.py
   ```

---

## Files added / updated in the latest merge

- `scripts/migrations/001_add_metadata_and_features.sql` — schema migration
- `src/data/features.py` — deterministic feature computation & persistence
- `src/llm/verify.py` — evidence verification helpers
- `src/llm/utils.py` — parsing and streaming helpers
- `scripts/run_migration_and_features.py` — runner to apply migration and compute features
- `notebooks/fraud_duckdb_poc.ipynb` — updated ingestion, batching, repair and verification flows
- `tests/test_parse_and_verify.py` — unit tests

---

## Troubleshooting & common errors

- DuckDB file not found
  - Ensure `FRAUD_DB_PATH` points to the DB or place `fraud_poc.duckdb` in repo root.
- LLM/OLLAMA timeouts
  - Pre-pull the model: `ollama pull <model>` and keep `ollama serve` running; increase timeouts in notebook.
- Batch JSON parse fails
  - Make prompt stricter, reduce batch size, or fallback to per-item calls.
- Many `needs_review` rows
  - Improve prompt examples, reduce temperature, validate evidence and reprocess flagged rows.

---

## Security & privacy notes

- Redact or hash PII before sending to LLM if retention of raw prompts/responses is not permitted.
- Store prompt/response logs only for limited retention window or encrypt at rest.
- Review regulatory constraints for customer data processing.

---

## Next steps & improvement roadmap

- Replace retrieval placeholder with real embedding retrieval (FAISS / DuckDB embedding table).
- Implement deterministic baseline model (sklearn/lightGBM) and use LLM as constrained adjuster.
- Add calibration and evaluation on labeled dataset (AUC, calibration curves).
- Add CI to run unit tests and daily DQ checks.
- Add monitoring dashboard for latency, parse success, and `needs_review` rate.