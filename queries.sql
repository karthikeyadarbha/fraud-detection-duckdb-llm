-- List datasets and owners
SELECT dataset_id, title, owner, created_at, updated_at FROM catalog_datasets ORDER BY dataset_id;

-- View a contract YAML quickly
SELECT contract_yaml FROM catalog_contracts WHERE dataset_id = 'bankcorp.fraud.transactions';

-- List columns for a dataset
SELECT column_name, column_type, is_nullable, description FROM catalog_columns WHERE dataset_id = 'bankcorp.fraud.transactions' ORDER BY column_name;

-- Recent LLM calls and provenance
SELECT llm_call_id, tx_id, llm_provider, llm_model, call_ts, parsed_response->>'risk_score' as risk_score
FROM llm_call_catalog ORDER BY call_ts DESC LIMIT 100;

-- Find lineage for dataset
SELECT * FROM lineage WHERE dataset_id = 'bankcorp.fraud.transactions' ORDER BY run_ts DESC LIMIT 10;