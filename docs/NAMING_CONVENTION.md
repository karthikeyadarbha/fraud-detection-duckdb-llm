# Naming convention (opinionated)

Principles
- Lowercase snake_case for DB objects and columns.
- Use semantic, compact names: <domain>_<entity>[_<purpose>]
- Use `v_` prefix for views, `mv_` for materialized views, `tmp_` for ephemeral tables.
- Job/pipeline IDs use dotted or dotted-like notation: team.pipeline.task
- Model versions follow semantic versioning: vMAJOR.MINOR.PATCH

Examples
- Tables: transactions_canonical, transactions_staging, embeddings_transactions, llm_results
- Views: v_transactions_enriched
- Job names: fraud.ingest.transactions, ml.fraud.embed_build
- Model registry: model_id = fraud_xgb:v1.2.0
- Prompt templates: prompt_fraud_risk_v1
- Filenames: fraud-poc-duckdb.ipynb, fraud-poc-pipeline.py

Enforcement
- Add CI lint to enforce allowed characters and patterns (max length, allowed chars).