"""
Combiner for baseline, anomaly and LLM adjustment.

Provides:
- combine_scores(baseline, anomaly, llm_adj, policy='max'|'weighted', weights=(..))
- decide_action(final_score, thresholds)

Policy:
- default: final = max(baseline, anomaly, baseline+llm_adj)
- weighted: final = w1*baseline + w2*anomaly + w3*(baseline+llm_adj)
"""
from typing import Tuple


def combine_scores(baseline: float, anomaly: float, llm_adj: float = 0.0, policy: str = "max", weights: Tuple[float,float,float]=(0.7,0.2,0.1)) -> float:
    adjusted = max(0.0, min(1.0, baseline + llm_adj))
    if policy == "max":
        return max(baseline, anomaly, adjusted)
    elif policy == "weighted":
        w1, w2, w3 = weights
        s = w1*baseline + w2*anomaly + w3*adjusted
        # normalize weights if not summing to 1
        denom = (w1 + w2 + w3)
        if denom > 0:
            s = s / denom
        return max(0.0, min(1.0, s))
    else:
        return max(baseline, anomaly, adjusted)


def decide_action(final_score: float, block_threshold: float = 0.9, review_threshold: float = 0.6) -> str:
    if final_score >= block_threshold:
        return "block"
    elif final_score >= review_threshold:
        return "review"
    else:
        return "allow"