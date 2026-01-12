-- Migration: add metadata columns to llm_results and create llm_features table
BEGIN TRANSACTION;
-- Add metadata columns
ALTER TABLE llm_results ADD COLUMN IF NOT EXISTS prompt_text TEXT;
ALTER TABLE llm_results ADD COLUMN IF NOT EXISTS prompt_hash VARCHAR;
ALTER TABLE llm_results ADD COLUMN IF NOT EXISTS model_name VARCHAR;
ALTER TABLE llm_results ADD COLUMN IF NOT EXISTS model_version VARCHAR;
ALTER TABLE llm_results ADD COLUMN IF NOT EXISTS model_params VARCHAR;

-- Add numeric features table
CREATE TABLE IF NOT EXISTS llm_features (
  tx_id VARCHAR PRIMARY KEY,
  amount_z DOUBLE,
  velocity_1h INTEGER,
  velocity_24h INTEGER,
  fraud_fraction_topk DOUBLE,
  mean_similarity_topk DOUBLE,
  computed_at TIMESTAMP DEFAULT current_timestamp
);

COMMIT;
