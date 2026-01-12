"""Tests for parsing and evidence verification."""
import pytest
import math
from src.llm.verify import verify_evidence
from src.llm.parse import parse_risk_score


@pytest.mark.parametrize('val,expected', [
    ('0.82', 0.82),
    ('82%', 0.82),
    ('0,82', 0.82),
    ('null', float('nan')),
    (None, float('nan')),
    ('Risk is 0.5', 0.5)
])
def test_parse_risk(val, expected):
    """Test parsing various risk score formats."""
    out = parse_risk_score(val)
    if math.isnan(expected):
        assert math.isnan(out)
    else:
        assert abs(out - expected) < 1e-6


def test_verify_evidence():
    """Test evidence verification logic."""
    topk = ['a','b','c']
    ok, invalid = verify_evidence(['a','c'], topk)
    assert ok and invalid == []
    ok2, invalid2 = verify_evidence(['a','x'], topk)
    assert not ok2 and invalid2 == ['x']


def test_verify_evidence_none():
    """Test evidence verification with None input."""
    topk = ['a','b','c']
    ok, invalid = verify_evidence(None, topk)
    assert not ok
    assert invalid == []


def test_verify_evidence_empty():
    """Test evidence verification with empty evidence list."""
    topk = ['a','b','c']
    ok, invalid = verify_evidence([], topk)
    assert ok  # Empty list is valid (subset of anything)
    assert invalid == []


def test_parse_risk_numeric():
    """Test parsing numeric values."""
    assert parse_risk_score(0.5) == 0.5
    assert parse_risk_score(1) == 1.0
    assert parse_risk_score(0) == 0.0


def test_parse_risk_percentage():
    """Test parsing percentage strings."""
    assert abs(parse_risk_score('50%') - 0.5) < 1e-6
    assert abs(parse_risk_score('100%') - 1.0) < 1e-6
    assert abs(parse_risk_score('0%') - 0.0) < 1e-6


def test_parse_risk_invalid():
    """Test parsing invalid values."""
    result = parse_risk_score('invalid')
    assert math.isnan(result)  # Should be NaN
    
    result = parse_risk_score('')
    assert math.isnan(result)  # Should be NaN
