# Security & Privacy Notes

PII handling
- Mask or tokenize PII before sending to any external LLMs.
- Store PII mapping (masked -> raw) only in a secure vault (HashiCorp Vault), not in DuckDB.
- If policy forbids storing prompts, store `prompt_hash` and `prompt_template_id` instead of full prompt.

Encryption & access
- Store DuckDB file on encrypted disk (or use an encrypted filesystem).
- Restrict access to the DuckDB file via OS-level permissions or expose DuckDB behind an API service that enforces RBAC and audit logs.

Provenance & auditing
- For each LLM call store: prompt_hash, prompt_template_id, llm_provider, llm_model+version, call_ts, latency, retrieved_ids, job_id.
- Maintain `audit_log` and `lineage` tables for job-level provenance (input files, commit sha, run id).

Operational safeguards
- Use local/on-prem LLMs when regulatory constraints prevent external calls.
- Use rate-limiting, batching and caching to reduce cost and risk.
- Implement a manual override and require approval for actions that block high-value transactions.