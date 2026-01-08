# Fraud-detection PoC — DuckDB + Embeddings + LLMs

Purpose
- Validate an auditable, in-place data-management pattern that enriches transaction records with semantic retrieval and LLM-generated explanations to improve fraud detection, triage, and analyst productivity.
- Use mostly open-source components and store all metadata, provenance and LLM outputs in DuckDB for governance and traceability.

Repository layout (recommended)
- notebooks/
  - fraud_duckdb_poc.ipynb         ← runnable PoC notebook (provided)
- sql/
  - schema.sql                     ← canonical production schemas
  - catalog_schema.sql             ← mini-catalog schemas
- data_contracts/
  - transactions.yml               ← git-backed data contract example
- docs/
  - ARCHITECTURE.md
  - POC_PLAN.md
  - NAMING_CONVENTION.md
  - SECURITY_PRIVACY.md
  - RUNBOOK.md
- tools/
  - register_dataset.py            ← CLI to push YAML contracts into DuckDB (optional)
- README.md                        ← this file
- CONTRIBUTING.md                  ← check-in and CI rules

Quick start (PoC)
1. Install dependencies:
   - Python 3.8+ and pip
   - pip install duckdb sentence-transformers scikit-learn pandas numpy openai (if using OpenAI)
2. Open and run `notebooks/fraud_duckdb_poc.ipynb` in Jupyter. The notebook:
   - Creates DuckDB tables
   - Inserts synthetic transactions
   - Builds embeddings (sentence-transformers)
   - Builds an in-memory NearestNeighbors index and simulates retrieval
   - Calls a mock or real LLM (OpenAI if OPENAI_API_KEY is provided)
   - Persists LLM results and provenance in DuckDB
3. Register the data contract (optional):
   - python tools/register_dataset.py --db fraud_poc.duckdb data_contracts/transactions.yml
4. Review catalog & metadata:
   - Query `sql/catalog_schema.sql` tables from the notebook or via DuckDB CLI
   - Inspect `llm_results`, `audit_log`, `lineage` and catalog tables

Deliverables in this repo
- Runnable notebook: `notebooks/fraud_duckdb_poc.ipynb`
- Table schemas: `sql/schema.sql`, `sql/catalog_schema.sql`
- Data contract example: `data_contracts/transactions.yml`
- Documentation: `docs/*.md`
- Utility: `tools/register_dataset.py`

Recommended commit message for this initial check-in
- feat(poc): add DuckDB fraud-detection PoC notebook, schemas, mini-catalog and documentation

If you want, I can:
- Produce a `docker-compose.yml` that runs a Jupyter server + local FAISS and a minimal LLM serving container for local PoC,
- Create GitHub Actions that lint naming and auto-register `data_contracts/*.yml` into the DuckDB file on push,
- Or open a PR template and CI job files to enforce naming and schema checks.