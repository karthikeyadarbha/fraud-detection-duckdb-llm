"""Parsing utilities for LLM outputs."""
import json
import re
import math
import numpy as np


def parse_risk_score(value):
    """
    Parse risk score from various formats.
    
    Returns float in 0..1 or math.nan if not parseable.
    Handles numeric types, strings, JSON objects, percentages, etc.
    
    Args:
        value: Value to parse (can be int, float, string, JSON, etc.)
        
    Returns:
        float: Parsed risk score between 0 and 1, or math.nan if unparseable
    """
    if value is None:
        return math.nan
    # numeric types
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            v = float(value)
            return math.nan if math.isnan(v) else v
        except (ValueError, TypeError):
            return math.nan
    s = str(value).strip()
    # try JSON content
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            # common keys
            for key in ("risk_score","score","risk","riskScore"):
                if key in obj:
                    return parse_risk_score(obj[key])
        elif isinstance(obj, (int, float)):
            return float(obj)
    except Exception:
        pass
    low = s.lower()
    if low in ("","null","none","n/a","na","nan"):
        return math.nan
    # percent like 82%
    m = re.search(r'(-?\d+(?:[.,]\d+)?)\s*%', s)
    if m:
        try:
            num = float(m.group(1).replace(',','.'))
            return num/100.0
        except:
            return math.nan
    # find first numeric token
    m = re.search(r'(-?\d+(?:[.,]\d+)?)', s)
    if m:
        try:
            num = float(m.group(1).replace(',','.'))
        except:
            return math.nan
        if num < 0:
            return math.nan
        if num > 1 and num <= 100:
            return num/100.0
        return float(num)
    return math.nan
