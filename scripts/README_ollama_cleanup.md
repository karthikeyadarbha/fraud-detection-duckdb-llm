# Ollama Cleanup Scripts

Use these scripts to remove Ollama from a Codespace/WSL environment, reclaim disk space, and clear Level‑3 artifacts if desired.

Quick start (Linux/WSL/Codespaces)
```bash
# Make executable
chmod +x scripts/cleanup_ollama.sh

# Interactive cleanup (asks before deleting)
./scripts/cleanup_ollama.sh

# Non-interactive (assumes yes)
./scripts/cleanup_ollama.sh --yes

# Remove models/cache only (keep binary)
./scripts/cleanup_ollama.sh --yes --models-only

# Full removal (models + binary; requires sudo)
./scripts/cleanup_ollama.sh --yes --full

# Also remove project L3 artifacts
./scripts/cleanup_ollama.sh --yes --artifacts
```

PowerShell (Windows)
```powershell
# Interactive
./scripts/cleanup_ollama.ps1

# Non-interactive models-only
./scripts/cleanup_ollama.ps1 -Yes -ModelsOnly

# Full removal
./scripts/cleanup_ollama.ps1 -Yes -Full

# Also remove project artifacts
./scripts/cleanup_ollama.ps1 -Yes -Artifacts
```

What gets removed
- Running processes: `ollama serve` and active model runs.
- Models/caches: `$OLLAMA_MODELS`, `$OLLAMA_HOME/models`, `~/.ollama/models`, and `~/.ollama`.
- Temp logs: `/tmp/ollama_serve.log` (Linux) or `%TEMP%\ollama_serve.log` (Windows).
- Optional: the `ollama` binary/package (requires sudo on Linux; admin on Windows).
- Optional: project artifacts — `artifacts/level3_llm_run`, `artifacts/level3_llm_warm`, `artifacts/level3_llm_smoke`.

Notes
- In Codespaces, sudo is typically available; removing the package may be a no‑op if Ollama was installed via direct binary download.
- If you set non‑default model directories (e.g., via `OLLAMA_MODELS`), the script will detect and remove them.
- Use `--models-only` when you plan to reinstall Ollama later and just want to reclaim disk.