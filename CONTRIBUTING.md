# Contributing & Check-in guidance

Repository rules
- Use the naming conventions in `docs/NAMING_CONVENTION.md`.
- Add new dataset contracts under `data_contracts/` as YAML and commit to git.
- Add any schema changes under `sql/` and include migration steps in the PR description.

Suggested PR template (summary)
- Title: feat(poc): short description
- Description: what changed, files added, how to test
- Checklist:
  - [ ] `sql/schema.sql` updated if schema changed
  - [ ] `data_contracts/*.yml` added or updated with owner and SLOs
  - [ ] Documentation updated in `docs/`
  - [ ] Notebook runs end-to-end locally (if applicable)

CI suggestions
- Add a naming-lint step (Python) to reject table/column names not matching `docs/NAMING_CONVENTION.md`
- Optional: Run `tools/register_dataset.py` in CI against a temporary DuckDB instance to validate contract YAMLs