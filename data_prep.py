"""
Minimal data preparation and feature engineering utilities.

Usage:
  from data_prep import load_csv, FitEncoder, transform_features

This module implements:
- load_csv(path): reads CSV and parses timestamp
- FitEncoder: fit historical stats (sender amount mean/std, merchant fraud rates)
- transform_features(df, encoder): returns feature DataFrame ready for modeling

Notes:
- Robustly converts/validates timestamp column (coerce invalid values to NaT).
- transform_features now tolerates missing or non-datetime timestamp values.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
import pandas.api.types as ptypes


def load_csv(path: str) -> pd.DataFrame:
    # Read CSV. Allow pandas to parse timestamp, but coerce afterwards to ensure datetime dtype.
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
    except Exception:
        # Fallback if parse_dates fails for any reason
        df = pd.read_csv(path)

    # Ensure timestamp column is datetimelike (coerce invalid -> NaT)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    return df


class FitEncoder:
    """
    Fit encoder computes per-sender statistics and merchant_category fraud rates.
    Save the dict (encoder.state) alongside models for consistent transforms.
    """

    def __init__(self, min_sender_count: int = 5, smoothing: float = 10.0):
        self.min_sender_count = min_sender_count
        self.smoothing = smoothing
        self.state: Dict = {}

    def fit(self, df: pd.DataFrame, label_col: str = "is_fraud"):
        d = df.copy()
        # normalize labels to 0/1
        d[label_col] = d[label_col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)

        # sender stats
        sender_stats = d.groupby('sender_account')['amount'].agg(['mean', 'std', 'count']).rename(columns={'count':'n'})
        self.state['sender_stats'] = sender_stats.to_dict(orient='index')

        # merchant category fraud rate (smoothed)
        cat = d.groupby('merchant_category').agg(total=('is_fraud','size'), fraud=('is_fraud','sum'))
        global_rate = d['is_fraud'].mean() if len(d) else 0.0
        cat['merchant_risk'] = (cat['fraud'] + self.smoothing * global_rate) / (cat['total'] + self.smoothing)
        self.state['merchant_risk'] = cat['merchant_risk'].to_dict()
        self.state['global_rate'] = float(global_rate)

        # store known device hashes per sender (for new-device flag)
        known_devices = d.groupby('sender_account')['device_hash'].agg(lambda x: set(x.dropna().astype(str).tolist()))
        self.state['known_devices'] = {k: v for k, v in known_devices.to_dict().items()}

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # convenience wrapper
        return transform_features(df, encoder=self)

    def to_dict(self):
        # make state JSON-serializable (convert sets)
        serial = dict(self.state)
        serial['sender_stats'] = {k: v for k, v in serial.get('sender_stats', {}).items()}
        serial['known_devices'] = {k: list(v) for k, v in serial.get('known_devices', {}).items()}
        return serial

    @classmethod
    def from_dict(cls, d: Dict):
        obj = cls()
        obj.state = dict(d)
        # restore known_devices to sets
        obj.state['known_devices'] = {k: set(v) for k, v in obj.state.get('known_devices', {}).items()}
        return obj


def transform_features(df: pd.DataFrame, encoder: Optional[FitEncoder] = None) -> pd.DataFrame:
    df = df.copy()

    # Ensure timestamp is datetimelike; coerce invalid values to NaT
    if 'timestamp' in df.columns and not ptypes.is_datetime64_any_dtype(df['timestamp']):
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except Exception:
            df['timestamp'] = pd.NaT

    # basic timestamp features - handle missing timestamps safely
    if 'timestamp' in df.columns and ptypes.is_datetime64_any_dtype(df['timestamp']):
        df['hour'] = df['timestamp'].dt.hour.fillna(0).astype(int)
        df['dayofweek'] = df['timestamp'].dt.dayofweek.fillna(0).astype(int)
        df['is_night'] = df['hour'].isin(list(range(0,6)) + [23]).astype(int)
    else:
        df['hour'] = 0
        df['dayofweek'] = 0
        df['is_night'] = 0

    # numeric anomaly features ensure numeric
    for col in ['time_since_last_transaction', 'spending_deviation_score', 'velocity_score', 'geo_anomaly_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    # amount features
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)

    # amount_z using sender stats from encoder if present
    if encoder and 'sender_stats' in encoder.state:
        means = encoder.state['sender_stats']
        def _amount_z(row):
            s = means.get(row['sender_account'])
            if s and s.get('std') and s.get('std') > 0 and s.get('n',0) >= encoder.min_sender_count:
                return (row['amount'] - s['mean']) / (s['std'] if s['std']>0 else 1.0)
            else:
                # fallback to global z
                return np.nan
        df['amount_z'] = df.apply(_amount_z, axis=1)
    else:
        df['amount_z'] = np.nan

    # fallback global z
    global_mean = df['amount'].mean()
    global_std = df['amount'].std(ddof=0) if df['amount'].std(ddof=0) > 0 else 1.0
    df['amount_z'] = df['amount_z'].fillna((df['amount'] - global_mean) / global_std)

    # device_change / new_device
    if encoder and 'known_devices' in encoder.state:
        known = encoder.state['known_devices']
        def _new_device(row):
            s = known.get(row['sender_account'], set())
            return 0 if row['device_hash'] in s else 1
        df['new_device'] = df.apply(_new_device, axis=1)
    else:
        # fallback heuristic: device_change by comparing to previous in same sender sorted by timestamp
        if 'timestamp' in df.columns and ptypes.is_datetime64_any_dtype(df['timestamp']):
            df = df.sort_values(['sender_account','timestamp'])
        else:
            df = df.sort_values(['sender_account'])
        df['prev_device'] = df.groupby('sender_account')['device_hash'].shift(1)
        df['new_device'] = (df['device_hash'] != df['prev_device']).astype(int).fillna(0)

    # merchant risk
    if encoder and 'merchant_risk' in encoder.state:
        mr = encoder.state['merchant_risk']
        global_rate = encoder.state.get('global_rate', 0.0)
        df['merchant_risk'] = df['merchant_category'].map(mr).fillna(global_rate)
    else:
        df['merchant_risk'] = 0.0

    # ip coarse feature (first octet) to capture network grouping
    def ip_prefix(ip):
        try:
            return str(ip).split('.')[0]
        except Exception:
            return '0'
    if 'ip_address' in df.columns:
        df['ip_prefix'] = df['ip_address'].fillna('0').apply(ip_prefix)
    else:
        df['ip_prefix'] = '0'

    # one-hot small categorical features (transaction_type, payment_channel, device_used) - return as dummies
    cats = ['transaction_type','payment_channel','device_used']
    for c in cats:
        if c in df.columns:
            df[c] = df[c].fillna('NA').astype(str)
        else:
            df[c] = 'NA'

    dummies = pd.get_dummies(df[['transaction_type','payment_channel','device_used','ip_prefix']], prefix=['tt','pc','du','ipp'])
    out = pd.concat([df[['transaction_id','timestamp','sender_account','receiver_account','amount','amount_z','time_since_last_transaction','spending_deviation_score','velocity_score','geo_anomaly_score','hour','is_night','dayofweek','new_device','merchant_risk']], dummies], axis=1)
    # fill NaNs
    out = out.fillna(0.0)
    return out