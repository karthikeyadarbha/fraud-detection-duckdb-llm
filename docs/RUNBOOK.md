# Runbook (short)

Incidents
- Ingestion failure:
  - Action: Stop promotion, notify Payments Data Team, revert to last good snapshot.
  - Recovery: Re-run ingestion job for failed batches, verify DQ checks.
- LLM provider unavailable:
  - Action: Failover to mock/static explanations or cached LLM outputs; revert to rule-based decisioning.
  - Recovery: Re-route calls to local LLM if available; resume queued enrichments.
- Spike in false positives:
  - Action: Pause automated blocking, switch routing to analyst-only workflow, open incident.
  - Recovery: rollback to previous model, analyze drift and retrain.

Daily checklist
- Verify ingestion lag (SLO)
- Run DQ checks (schema, nulls, distribution)
- Check LLM call success rate and average latency
- Inspect `audit_log` for unusual activities

On-call escalation
1. Data owner (payments-data-team)
2. ML Platform (ml-platform)
3. Security (security-team)

Runbook for deploying a schema change
- 1) Create schema migration SQL and tests
- 2) Run migration in dev DuckDB and run DQ tests
- 3) Commit migration and open PR
- 4) Deploy to staging and repeat tests
- 5) Deploy to prod with pre-approved rollback plan