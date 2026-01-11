"""Evidence verification utilities for LLM outputs."""

def verify_evidence(evidence_ids, topk_ids):
    """
    Verify that evidence IDs are a subset of top-k IDs.
    
    Args:
        evidence_ids: List of evidence transaction IDs provided by LLM
        topk_ids: List of top-k retrieved transaction IDs
        
    Returns:
        Tuple of (is_valid, invalid_ids) where is_valid is True when all 
        evidence_ids are subset of topk_ids
    """
    if evidence_ids is None:
        return False, []
    invalid = [eid for eid in evidence_ids if eid not in set(topk_ids)]
    return (len(invalid) == 0), invalid

def mark_needs_review_if_invalid(con, llm_result_id, evidence_ids, topk_ids):
    """
    Set needs_review flag for an LLM result if evidence is invalid.
    
    Args:
        con: DuckDB connection
        llm_result_id: ID of the LLM result row
        evidence_ids: List of evidence transaction IDs provided by LLM
        topk_ids: List of top-k retrieved transaction IDs
        
    Returns:
        Tuple of (is_valid, invalid_ids)
    """
    is_valid, invalid = verify_evidence(evidence_ids, topk_ids)
    if not is_valid:
        # set needs_review flag for that llm result
        con.execute('UPDATE llm_results SET needs_review = TRUE WHERE id = ?', [llm_result_id])
    return is_valid, invalid
