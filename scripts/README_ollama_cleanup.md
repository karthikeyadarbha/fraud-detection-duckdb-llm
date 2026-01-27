# Ollama Cleanup in GitHub Codespaces

This repo includes a script to remove Ollama from a Codespace/WSL dev container:
- Stops `ollama serve` and active model runs
- Deletes installed models and caches
- Optionally removes the Ollama binary
- Optionally clears Level‑3 artifacts

## Quick start

```bash
# Make executable
chmod +x scripts/cleanup_ollama.sh

# Interactive cleanup (asks before deleting)
./scripts/cleanup_ollama.sh

# Non-interactive (assumes yes)
./scripts/cleanup_ollama.sh --yes

# Remove models/cache only (keep binary)
./scripts/cleanup_ollama.sh --yes --models-only

# Full removal (models + binary; requires sudo in Codespaces)
./scripts/cleanup_ollama.sh --yes --full

# Also remove project L3 artifacts (llm_run/warm/smoke)
./scripts/cleanup_ollama.sh --yes --artifacts
```

## What gets removed

- Running processes: `ollama serve` and active model runs (`ollama stop <name>`)
- Models/caches: `$OLLAMA_MODELS`, `$OLLAMA_HOME/models`, `~/.ollama/models`, and `~/.ollama`
- Temp logs: `/tmp/ollama_serve.log`
- Optional: the `ollama` binary/package (via apt or file removal)
- Optional: project artifacts — `artifacts/level3_llm_run`, `artifacts/level3_llm_warm`, `artifacts/level3_llm_smoke`

## Tips

- Use `--models-only` if you plan to reinstall Ollama later and just want to reclaim disk.
- Keep your source datasets under `data/` to avoid deleting them by accident when cleaning project artifacts.
- If you customize model storage via `OLLAMA_MODELS` or `OLLAMA_HOME`, the script will detect and remove those directories.