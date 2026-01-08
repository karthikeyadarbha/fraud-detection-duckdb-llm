-- Mini-catalog schema (store metadata in DuckDB)

CREATE TABLE IF NOT EXISTS catalog_datasets (
  dataset_id VARCHAR PRIMARY KEY,
  title VARCHAR,
  owner VARCHAR,
  description VARCHAR,
  created_at TIMESTAMP DEFAULT current_timestamp,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS catalog_columns (
  dataset_id VARCHAR,
  column_name VARCHAR,
  column_type VARCHAR,
  is_nullable BOOLEAN,
  description VARCHAR,
  example_value VARCHAR,
  PRIMARY KEY (dataset_id, column_name)
);

CREATE TABLE IF NOT EXISTS catalog_contracts (
  dataset_id VARCHAR PRIMARY KEY,
  contract_yaml TEXT,
  contract_hash VARCHAR,
  contact_email VARCHAR,
  freshness_slo VARCHAR,
  retention_policy VARCHAR,
  pii_policy JSON,
  quality_checks JSON,
  created_at TIMESTAMP DEFAULT current_timestamp,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lineage (
  lineage_id VARCHAR PRIMARY KEY,
  dataset_id VARCHAR,
  parent_datasets JSON,
  transform_sql TEXT,
  job_id VARCHAR,
  commit_sha VARCHAR,
  run_ts TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS model_registry_minimal (
  model_id VARCHAR PRIMARY KEY,
  model_name VARCHAR,
  model_type VARCHAR,
  model_version VARCHAR,
  training_data_snapshot VARCHAR,
  created_at TIMESTAMP DEFAULT current_timestamp,
  metadata JSON
);

CREATE TABLE IF NOT EXISTS llm_call_catalog (
  llm_call_id VARCHAR PRIMARY KEY,
  tx_id VARCHAR,
  prompt_hash VARCHAR,
  prompt_template_id VARCHAR,
  llm_provider VARCHAR,
  llm_model VARCHAR,
  model_version VARCHAR,
  call_ts TIMESTAMP DEFAULT current_timestamp,
  parsed_response JSON,
  risk_score DOUBLE,
  retrieved_ids JSON,
  provenance JSON
);

CREATE TABLE IF NOT EXISTS audit_log_catalog (
  audit_id VARCHAR PRIMARY KEY,
  object_type VARCHAR,
  object_id VARCHAR,
  actor VARCHAR,
  action VARCHAR,
  details JSON,
  ts TIMESTAMP DEFAULT current_timestamp
);