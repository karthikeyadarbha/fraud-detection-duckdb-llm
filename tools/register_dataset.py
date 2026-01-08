#!/usr/bin/env python3
"""
Register a data_contract YAML into the DuckDB mini-catalog.

Usage:
  python tools/register_dataset.py --db fraud_poc.duckdb data_contracts/transactions.yml
"""
import argparse
import duckdb
import yaml
import hashlib
import json
from datetime import datetime

def sha256_text(t: str) -> str:
    import hashlib
    return hashlib.sha256(t.encode('utf-8')).hexdigest()

def register(db_path: str, yaml_path: str):
    con = duckdb.connect(db_path)
    # Ensure catalog schema exists
    con.execute(open('sql/catalog_schema.sql').read())

    with open(yaml_path, 'r') as f:
        contract_text = f.read()
    contract = yaml.safe_load(contract_text)

    dataset = contract.get('dataset')
    title = contract.get('title', '')
    owner = contract.get('owner', '')
    description = contract.get('description', '')
    contract_hash = sha256_text(contract_text)
    now = datetime.utcnow().isoformat()

    # Upsert dataset
    con.execute("""
    INSERT INTO catalog_datasets(dataset_id, title, owner, description, created_at, updated_at)
    VALUES (?, ?, ?, ?, current_timestamp, ?)
    ON CONFLICT (dataset_id) DO UPDATE SET title=excluded.title, owner=excluded.owner, description=excluded.description, updated_at=excluded.updated_at
    """, (dataset, title, owner, description, now))

    # Upsert contract
    con.execute("""
    INSERT INTO catalog_contracts(dataset_id, contract_yaml, contract_hash, contact_email, freshness_slo, retention_policy, pii_policy, quality_checks, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp, ?)
    ON CONFLICT (dataset_id) DO UPDATE SET contract_yaml=excluded.contract_yaml, contract_hash=excluded.contract_hash, contact_email=excluded.contact_email, freshness_slo=excluded.freshness_slo, retention_policy=excluded.retention_policy, pii_policy=excluded.pii_policy, quality_checks=excluded.quality_checks, updated_at=excluded.updated_at
    """, (dataset, contract_text, contract_hash, owner, contract.get('slo', {}).get('freshness',''), json.dumps(contract.get('retention',{})), json.dumps(contract.get('pii_policy',{})), json.dumps(contract.get('quality_checks',[])), now))

    # Populate columns table from schema section (if present)
    columns = contract.get('schema', [])
    for col in columns:
        con.execute("""
        INSERT INTO catalog_columns(dataset_id, column_name, column_type, is_nullable, description, example_value)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (dataset_id, column_name) DO UPDATE SET column_type=excluded.column_type, is_nullable=excluded.is_nullable, description=excluded.description
        """, (dataset, col.get('name'), col.get('type'), not bool(col.get('required', False)), col.get('description',''), col.get('example', None)))

    print(f"Registered {dataset} in {db_path} (hash={contract_hash})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True, help='DuckDB path (e.g., fraud_poc.duckdb or :memory:)')
    parser.add_argument('yaml', help='data contract YAML file path')
    args = parser.parse_args()
    register(args.db, args.yaml)