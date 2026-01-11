# Fraud Detection DuckDB LLM - Schema Migration and Features

This document describes the schema migration, feature pipeline, and evidence verification improvements.

## Overview

The following enhancements have been implemented:

1. **Schema Migration**: Added metadata columns to `llm_results` and created `llm_features` table
2. **Feature Pipeline**: Computes deterministic features (amount z-scores, velocity, retrieval placeholders)
3. **Evidence Verification**: Verifies LLM-provided evidence against retrieved topK IDs
4. **Enhanced Notebook**: Updated to use new modules and workflows

## Files Added/Modified

### New Files

- `scripts/migrations/001_add_metadata_and_features.sql` - Database migration script
- `src/data/features.py` - Feature computation module
- `src/llm/verify.py` - Evidence verification module
- `src/llm/parse.py` - LLM response parsing utilities
- `tests/test_parse_and_verify.py` - Unit tests for parsing and verification
- `.gitignore` - Git ignore patterns for Python and temporary files

### Modified Files

- `notebooks/fraud_duckdb_poc.ipynb` - Updated to integrate new modules

## Deployment Steps

### 1. Run the Migration

Apply the database migration to add new columns and tables:

```bash
# From the repository root
duckdb fraud_poc.duckdb < scripts/migrations/001_add_metadata_and_features.sql
```

This migration:
- Adds metadata columns to `llm_results`: `prompt_text`, `prompt_hash`, `model_name`, `model_version`, `model_params`
- Creates `llm_features` table for storing computed features
- Uses `IF NOT EXISTS` for idempotency (safe to run multiple times)

### 2. Install Dependencies

Ensure all Python dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages include:
- `duckdb`
- `pandas`
- `numpy`
- `pytest` (for testing)

### 3. Run Tests

Verify the implementation with unit tests:

```bash
pytest tests/test_parse_and_verify.py -v
```

Expected output: All 12 tests should pass.

### 4. Run the Notebook

Open and run the updated notebook:

```bash
jupyter notebook notebooks/fraud_duckdb_poc.ipynb
```

The notebook now:
- Runs the migration automatically (or instructs you to run it)
- Computes features before calling the LLM
- Uses modular imports from `src/` packages
- Verifies evidence and stores prompt metadata
- Sets `needs_review` flag when evidence is invalid

## Module Documentation

### Feature Pipeline (`src/data/features.py`)

Computes and persists features to the `llm_features` table:

```python
from src.data.features import compute_and_persist_features

# Compute features for all transactions
compute_and_persist_features('fraud_poc.duckdb', topk=5)
```

**Features computed:**
- `amount_z`: Z-score of transaction amount (per account)
- `velocity_1h`: Number of transactions in last 1 hour
- `velocity_24h`: Number of transactions in last 24 hours
- `fraud_fraction_topk`: Placeholder for fraud rate in topK neighbors
- `mean_similarity_topk`: Placeholder for average similarity to topK

### Evidence Verification (`src/llm/verify.py`)

Verifies that LLM-provided evidence IDs are a subset of retrieved topK IDs:

```python
from src.llm.verify import verify_evidence

topk_ids = ['tx_001', 'tx_002', 'tx_003']
evidence_ids = ['tx_001', 'tx_003']

is_valid, invalid = verify_evidence(evidence_ids, topk_ids)
# is_valid = True, invalid = []
```

### Parsing Utilities (`src/llm/parse.py`)

Robust parsing of risk scores from various formats:

```python
from src.llm.parse import parse_risk_score

# Handles multiple formats
parse_risk_score('0.82')      # 0.82
parse_risk_score('82%')       # 0.82
parse_risk_score('0,82')      # 0.82
parse_risk_score('Risk is 0.5')  # 0.5
parse_risk_score('null')      # NaN
```

## Database Schema

### New Columns in `llm_results`

- `prompt_text` (TEXT): Full prompt sent to LLM
- `prompt_hash` (VARCHAR): Hash of prompt for deduplication
- `model_name` (VARCHAR): Model name (e.g., 'gemma')
- `model_version` (VARCHAR): Model version (e.g., '2b')
- `model_params` (VARCHAR): JSON string of model parameters

### New Table: `llm_features`

```sql
CREATE TABLE llm_features (
  tx_id VARCHAR PRIMARY KEY,
  amount_z DOUBLE,
  velocity_1h INTEGER,
  velocity_24h INTEGER,
  fraud_fraction_topk DOUBLE,
  mean_similarity_topk DOUBLE,
  computed_at TIMESTAMP DEFAULT current_timestamp
);
```

## Workflow

### Updated LLM Processing Workflow

1. **Migration**: Run SQL migration to add columns/tables
2. **Feature Computation**: Compute features for all transactions
3. **Retrieval**: Retrieve topK similar transactions (placeholder)
4. **LLM Call**: Generate risk assessment
5. **Parsing**: Parse risk score from response
6. **Verification**: Verify evidence against topK
7. **Storage**: Store results with metadata and needs_review flag

### Example Notebook Cell Flow

```python
# 1. Run migration (one-time, idempotent)
con.execute(migration_sql)

# 2. Compute features
compute_and_persist_features(DB_PATH, topk=5)

# 3. Process transactions
for tx in unprocessed_txs:
    # Retrieve topK
    topk_ids = retrieve_topk(tx_id, k=5)
    
    # Call LLM
    response = call_llm(prompt)
    
    # Parse and verify
    risk_score = parse_risk_score(response)
    is_valid, invalid = verify_evidence(evidence_ids, topk_ids)
    
    # Store with metadata
    safe_insert_llm_result(con, row_id, tx_id, model, response, 
                          risk_score, not is_valid, now,
                          prompt_text=prompt, evidence_ids=evidence_ids, 
                          topk_ids=topk_ids)
```

## Testing

### Run All Tests

```bash
pytest tests/test_parse_and_verify.py -v
```

### Test Coverage

- Risk score parsing (various formats)
- Evidence verification (valid, invalid, edge cases)
- None/empty input handling
- Numeric and percentage parsing

## Notes

- The migration is idempotent (`IF NOT EXISTS`) and safe to run multiple times
- Feature computation uses placeholder retrieval features (fraud_fraction_topk, mean_similarity_topk set to 0.0)
- Replace placeholder retrieval logic with actual embedding-based similarity when available
- The notebook preserves existing streaming assembly and robust parsing
- All changes maintain backward compatibility with existing data

## Troubleshooting

### Migration fails

Ensure DuckDB file exists and is accessible:
```bash
ls -l fraud_poc.duckdb
```

### Import errors in notebook

Ensure the notebook is run from the repository root or add parent directory to path:
```python
import sys
sys.path.insert(0, '..')
```

### Features not computing

Check that transactions table has data:
```sql
SELECT COUNT(*) FROM transactions;
```

## Future Enhancements

- Replace placeholder retrieval with actual embedding-based similarity
- Add support for extracting evidence IDs from LLM responses
- Implement batch feature computation for performance
- Add feature versioning and tracking
- Create visualization dashboard for features and evidence verification metrics
