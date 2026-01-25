# Fraud Detection Prototype (LightGBM + IsolationForest + LLM adapter)

This repository contains a minimal prototype to implement the 3-level fraud scoring system:

- Level 1: Supervised baseline (LightGBM) → calibrated baseline_score (authoritative).
- Level 2: Unsupervised anomaly detection (IsolationForest) → anomaly_score (0..1).
- Level 3: LLM adapter (local Ollama/LLaMA) → explanation + bounded adjustment (not authoritative).

Files:
- data_prep.py — CSV loader and feature engineering; encoder to persist sender/merchant stats.
- train_baseline.py — trains calibrated LightGBM baseline and writes artifacts/.
- train_anomaly.py — trains IsolationForest anomaly model and writes artifacts/.
- llm_adapter.py — builds prompt, calls Ollama (CLI), validates JSON and enforces guardrails.
- combine_scores.py — combine function and decision thresholds.
- run_pipeline.py — example offline pipeline: load models, score CSV, combine, write results.
- requirements.txt — Python dependencies.

Quickstart (local, minimal):
1. Create a virtualenv and install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Train baseline:
   python train_baseline.py path/to/financial_fraud_detection_dataset.csv

   This saves artifacts/lgb_baseline_{version}.pkl and a manifest JSON.

3. Train anomaly model:
   python train_anomaly.py path/to/financial_fraud_detection_dataset.csv

4. Run pipeline (offline demo):
   python run_pipeline.py artifacts/lgb_baseline_{version}.pkl artifacts/iso_anomaly_{version}.pkl path/to/financial_fraud_detection_dataset.csv

Notes / Next steps:
- The LLM step uses Ollama CLI by default. If you use a remote model (OpenAI), replace call_ollama(...) in llm_adapter.py.
- Persist prompt_hash, raw prompt, and raw LLM output for audit logging.
- Use time-based train/validation/test splits; maintain manifests with model_version and training metadata.
- Add CI unit tests for feature engineering, small deterministic dataset tests, and integration tests.
- For production, host models in a model registry (MLflow or S3), instrument feature drift & calibration monitoring, and enforce PII masking when calling external LLMs.

Security:
- Do not send raw PII to external LLM providers. Prefer local LLM (Ollama) or remove/mask PII.