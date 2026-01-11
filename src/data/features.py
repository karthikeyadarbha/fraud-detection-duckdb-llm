"""Feature computation utilities for fraud POC."""
import duckdb
import numpy as np
import pandas as pd

def compute_amount_zscores(con):
    """Compute per-account mean/std and amount zscore for recent transactions."""
    q = """
    SELECT tx_id, account_id, amount
    FROM transactions
    """
    df = con.execute(q).df()
    if df.empty:
        return pd.DataFrame(columns=['tx_id','amount_z'])
    grp = df.groupby('account_id')['amount']
    stats = grp.agg(['mean','std']).reset_index().rename(columns={'mean':'acct_mean','std':'acct_std'})
    merged = df.merge(stats, on='account_id', how='left')
    merged['acct_std'] = merged['acct_std'].replace(0, np.nan)
    merged['amount_z'] = (merged['amount'] - merged['acct_mean']) / merged['acct_std']
    merged['amount_z'] = merged['amount_z'].fillna(0.0)
    return merged[['tx_id','amount_z']]

def compute_velocity(con):
    """Compute velocity counts in last 1h and 24h; uses created_at in transactions."""
    q = """
    SELECT t1.tx_id, 
           SUM(CASE WHEN (t2.created_at >= t1.created_at - INTERVAL '1 hour') THEN 1 ELSE 0 END) AS velocity_1h,
           SUM(CASE WHEN (t2.created_at >= t1.created_at - INTERVAL '24 hour') THEN 1 ELSE 0 END) AS velocity_24h
    FROM transactions t1
    LEFT JOIN transactions t2 ON t1.account_id = t2.account_id
    GROUP BY t1.tx_id, t1.created_at
    """
    df = con.execute(q).df()
    return df[['tx_id','velocity_1h','velocity_24h']]

def compute_placeholder_retrieval_features(con, topk=5):
    """
    Placeholder for retrieval-based features.
    
    If you have embeddings and labels table, replace with real retrieval logic.
    Here we compute dummy zeros for schema compatibility.
    """
    df = con.execute("SELECT tx_id FROM transactions").df()
    if df.empty:
        return pd.DataFrame(columns=['tx_id','fraud_fraction_topk','mean_similarity_topk'])
    df['fraud_fraction_topk'] = 0.0
    df['mean_similarity_topk'] = 0.0
    return df[['tx_id','fraud_fraction_topk','mean_similarity_topk']]


def compute_and_persist_features(db_path, topk=5):
    """
    Compute and persist features to llm_features table.
    
    Args:
        db_path: Path to DuckDB database file
        topk: Number of top-k neighbors for retrieval features (default 5)
    """
    con = duckdb.connect(db_path)
    amount_z_df = compute_amount_zscores(con)
    velocity_df = compute_velocity(con)
    retrieval_df = compute_placeholder_retrieval_features(con, topk=topk)
    # Merge
    merged = amount_z_df.merge(velocity_df, on='tx_id', how='outer').merge(retrieval_df, on='tx_id', how='outer')
    merged = merged.fillna({'amount_z':0.0,'velocity_1h':0,'velocity_24h':0,'fraud_fraction_topk':0.0,'mean_similarity_topk':0.0})
    merged['computed_at'] = pd.Timestamp.utcnow()
    # write into llm_features table (replace or upsert)
    con.register('tmp_features', merged)
    con.execute('CREATE TABLE IF NOT EXISTS llm_features (tx_id VARCHAR PRIMARY KEY, amount_z DOUBLE, velocity_1h INTEGER, velocity_24h INTEGER, fraud_fraction_topk DOUBLE, mean_similarity_topk DOUBLE, computed_at TIMESTAMP)')
    con.execute('INSERT OR REPLACE INTO llm_features SELECT * FROM tmp_features')
    con.close()

if __name__ == '__main__':
    import os
    db = os.environ.get('FRAUD_DB_PATH','fraud_poc.duckdb')
    compute_and_persist_features(db)
