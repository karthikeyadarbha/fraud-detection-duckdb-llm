-- Canonical production schemas for the PoC

CREATE TABLE IF NOT EXISTS transactions (
  tx_id VARCHAR PRIMARY KEY,
  account_id VARCHAR,
  amount DOUBLE,
  currency VARCHAR,
  merchant VARCHAR,
  merchant_mcc VARCHAR,
  timestamp TIMESTAMP,
  description VARCHAR,
  device_info VARCHAR,
  geo_country VARCHAR,
  ingestion_job_id VARCHAR,
  raw_source VARCHAR,
  pii_masked BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS transaction_labels (
  tx_id VARCHAR,
  label INTEGER,              -- 0 = legit, 1 = fraud
  labeled_by VARCHAR,         -- analyst id or automated process
  label_ts TIMESTAMP,
  label_source VARCHAR,       -- e.g., 'analyst_queue','chargeback'
  label_job_id VARCHAR,
  PRIMARY KEY (tx_id, label_ts)
);

CREATE TABLE IF NOT EXISTS embeddings (
  tx_id VARCHAR PRIMARY KEY,
  emb_blob BLOB,              -- raw vector bytes or base64; PoC uses JSON text
  emb_model VARCHAR,          -- embedding model + version
  emb_created_at TIMESTAMP DEFAULT current_timestamp,
  emb_job_id VARCHAR          -- batch job id
);

CREATE TABLE IF NOT EXISTS llm_results (
  id VARCHAR PRIMARY KEY,     -- uuid
  tx_id VARCHAR,
  llm_model VARCHAR,          -- model name + version
  llm_provider VARCHAR,       -- provider name
  llm_prompt_hash VARCHAR,    -- hash of prompt for de-duplication
  llm_prompt VARCHAR,         -- stored only if allowed by compliance
  llm_response VARCHAR,       -- raw response
  parsed_response JSON,       -- structured parse if available
  risk_score DOUBLE,
  evidence_tx_ids VARCHAR,    -- JSON list of retrieved tx ids
  retrieved_embedding_ids VARCHAR,
  call_latency_ms INTEGER,
  cost_estimate DOUBLE,
  provenance JSON,            -- {ingestion_job, embeddings_job, prompt_version}
  created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS model_registry (
  model_name VARCHAR PRIMARY KEY,
  model_type VARCHAR,         -- 'xgboost','llm','embedder'
  model_version VARCHAR,
  storage_uri VARCHAR,
  owner VARCHAR,
  created_at TIMESTAMP,
  metadata JSON
);

CREATE TABLE IF NOT EXISTS alerts (
  alert_id VARCHAR PRIMARY KEY,
  tx_id VARCHAR,
  alert_ts TIMESTAMP DEFAULT current_timestamp,
  alert_reason VARCHAR,
  alert_score DOUBLE,
  action_taken VARCHAR,
  assigned_to VARCHAR,
  status VARCHAR,             -- 'open','investigating','closed'
  provenance JSON
);

CREATE TABLE IF NOT EXISTS audit_log (
  log_id VARCHAR PRIMARY KEY,
  object_type VARCHAR,        -- e.g., 'llm_call','ingest','labeling'
  object_id VARCHAR,
  actor VARCHAR,
  action VARCHAR,
  details JSON,
  ts TIMESTAMP DEFAULT current_timestamp
);