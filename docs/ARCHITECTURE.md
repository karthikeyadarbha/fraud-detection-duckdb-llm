# Architecture: Fraud-detection PoC

Overview
- DuckDB is the canonical, queryable analytics store used to hold transactions, embeddings metadata, LLM outputs and the mini-catalog.
- Embeddings are computed using a sentence-transformers model (Hugging Face) and stored as JSON/BLOB for PoC or pushed to FAISS/Milvus for scale.
- Retrieval is nearest-neighbour (semantic) over embeddings (FAISS local in PoC).
- LLM calls (mock or real) are performed from Python tasks that assemble a RAG-style prompt and parse JSON output (risk_score, explanation, evidence).
- All LLM outputs, prompt metadata (hash/template id), model version and provenance are stored in DuckDB for auditability.

Components
- Ingest Layer: Python/Prefect/Airflow jobs produce canonical transactions into DuckDB (include ingestion_job_id).
- Embedding job: batch job computes embeddings and stores them; produces emb_job_id.
- Vector store / ANN: FAISS (local PoC) or Milvus/Weaviate (prod).
- LLM Service: local or hosted LLM; call via Python and persist results.
- Mini-catalog: YAML files in `data_contracts/` tracked in git + catalog tables in DuckDB.

Dataflow (concise)
1. Raw -> landing -> ingest -> `transactions` (DuckDB) with `ingestion_job_id`.
2. Embed descriptions -> write to `embeddings` table and index vector store -> `emb_job_id`.
3. Incoming tx -> compute emb -> retrieve top-K similar `tx_id`s -> assemble prompt -> call LLM.
4. Parse response -> persist to `llm_results` with provenance -> trigger alert or analyst queue.
5. Analyst labels -> `transaction_labels` -> used for retraining and evaluation.

RAG & Prompt pattern
- Retrieval returns top-K historical transactions with labels and concise structured facts.
- Prompt template includes masked transaction info, retrieved cases, and the expected JSON schema.
- Persist prompt_hash and template id; store full prompt only when allowed.

Scaling notes
- DuckDB: excellent for PoC and analytical queries; when vectors grow to millions or QPS is high, move vectors to dedicated vector DB (Milvus/Weaviate) and keep metadata in DuckDB.
- LLMs: small local models on CPU for PoC; for production use GPU-backed TGI / OpenLLM or a managed provider depending on quality/latency requirements.