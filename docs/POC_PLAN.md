# PoC Plan & Phases

Goal
- Demonstrate a reproducible pipeline that enriches transaction records with LLM explanations and retrieval-grounded context while keeping full provenance in DuckDB.

Phases
- Phase 0 — Discovery & Contracts
  - Define dataset owners, SLOs and data contracts (git YAML).
  - Seed the mini-catalog into DuckDB.
- Phase 1 — Offline PoC
  - Run `notebooks/fraud_duckdb_poc.ipynb` on historical synthetic data.
  - Validate retrieval relevance and LLM explanation quality.
- Phase 2 — Analyst Pilot
  - Expose enriched view in Metabase/Streamlit; analysts receive LLM explanations (no auto-block).
  - Collect analyst feedback and labels.
- Phase 3 — Controlled Autonomy
  - Automate actions only for high-confidence cases with defined rollback.
  - Implement batching, caching, and rate-limiting for LLM calls.
- Phase 4 — Scale & Harden
  - Migrate embeddings to Milvus/Weaviate if needed.
  - Harden security, Observability, model registry, and CI.

Success Criteria (examples)
- Retrieval relevance: relevance@K >= target
- Model uplift: e.g., AUC improvement or precision@k improvement vs baseline
- Analyst efficiency: time-to-triage reduction
- Governance: all LLM calls logged with prompt_hash, model_version and provenance

Deliverables
- Runnable notebook PoC
- Canonical schema and catalog table definitions (SQL)
- Data contract YAMLs (git)
- Documentation and runbook
- Optional: docker-compose for local PoC environment