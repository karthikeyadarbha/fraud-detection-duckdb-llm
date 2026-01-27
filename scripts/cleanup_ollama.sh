#!/usr/bin/env bash
# Cleanup Ollama from a Codespace/WSL/Linux dev container.
# - Stops any running ollama serve/model processes
# - Removes all installed Ollama models and caches
# - Optionally removes the ollama binary (requires sudo)
# - Optionally removes Level-3 LLM artifacts from this repo
#
# Usage:
#   ./scripts/cleanup_ollama.sh                 # interactive (asks for confirmation)
#   ./scripts/cleanup_ollama.sh --yes           # non-interactive (assumes yes)
#   ./scripts/cleanup_ollama.sh --models-only   # only delete models/caches (keep binary)
#   ./scripts/cleanup_ollama.sh --full          # also remove ollama binary (requires sudo)
#   ./scripts/cleanup_ollama.sh --artifacts     # also clear artifacts/level3_* dirs
#
# Env:
#   OLLAMA_MODELS   If set, used as the models directory root
#   OLLAMA_HOME     Alternative base (we check $OLLAMA_HOME/models)
set -euo pipefail

YES=0
FULL=0
MODELS_ONLY=0
ARTIFACTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) YES=1; shift ;;
    --full) FULL=1; shift ;;
    --models-only) MODELS_ONLY=1; shift ;;
    --artifacts) ARTIFACTS=1; shift ;;
    -h|--help)
      sed -n '1,60p' "$0"
      exit 0
      ;;
    *) echo "[warn] unknown arg: $1"; shift ;;
  esac
done

confirm() {
  if [[ $YES -eq 1 ]]; then
    return 0
  fi
  read -r -p "$1 [y/N]: " ans
  case "${ans,,}" in
    y|yes) return 0 ;;
    *) return 1 ;;
  esac
}

echo "[info] cleanup Ollama starting..."

BIN_PATH="$(command -v ollama || true)"
if [[ -n "$BIN_PATH" ]]; then
  echo "[info] ollama binary: $BIN_PATH"
  echo "[info] ollama version: $(ollama --version || echo 'unknown')"
else
  echo "[info] ollama binary not found in PATH"
fi

# 1) Stop any running ollama serve / model processes
echo "[info] stopping running ollama processes (if any)..."
if pgrep -f "ollama serve" >/dev/null 2>&1; then
  if confirm "Stop ollama serve process(es)?"; then
    pkill -f "ollama serve" || true
  fi
fi

if [[ -n "$BIN_PATH" ]]; then
  # Attempt to stop running models gracefully
  PS=$(ollama ps 2>/dev/null || true)
  if [[ -n "$PS" ]] && [[ "$PS" != "NAME"* ]]; then
    echo "$PS" | awk 'NR>1{print $1}' | while read -r name; do
      [[ -z "$name" ]] && continue
      echo "[info] ollama stop $name"
      ollama stop "$name" || true
    done
  fi
fi

# 2) Remove all installed models via ollama rm (if possible)
if [[ -n "$BIN_PATH" ]]; then
  LIST=$(ollama list 2>/dev/null || true)
  if [[ -n "$LIST" ]] && [[ "$LIST" != "NAME"* ]]; then
    echo "[info] removing models via ollama rm ..."
    echo "$LIST" | awk 'NR>1{print $1}' | while read -r mdl; do
      [[ -z "$mdl" ]] && continue
      echo "  - ollama rm $mdl"
      ollama rm "$mdl" || true
    done
  else
    echo "[info] no models listed by ollama"
  fi
fi

# 3) Remove model directories/caches on disk
CANDIDATES=()
if [[ -n "${OLLAMA_MODELS:-}" ]]; then
  CANDIDATES+=("$OLLAMA_MODELS")
fi
if [[ -n "${OLLAMA_HOME:-}" ]]; then
  CANDIDATES+=("$OLLAMA_HOME/models" "$OLLAMA_HOME")
fi
CANDIDATES+=("$HOME/.ollama/models" "$HOME/.ollama")

echo "[info] candidate model/cache dirs: ${CANDIDATES[*]}"
for d in "${CANDIDATES[@]}"; do
  if [[ -d "$d" ]]; then
    if confirm "Delete directory $d ?"; then
      echo "[info] rm -rf $d"
      rm -rf "$d"
    fi
  fi
done

# 4) Remove temp logs created by project
if [[ -f /tmp/ollama_serve.log ]]; then
  if confirm "Delete /tmp/ollama_serve.log ?"; then
    rm -f /tmp/ollama_serve.log
  fi
fi

# 5) Optionally remove project artifacts
if [[ $ARTIFACTS -eq 1 ]]; then
  for d in artifacts/level3_llm_run artifacts/level3_llm_warm artifacts/level3_llm_smoke; do
    if [[ -d "$d" ]]; then
      if confirm "Delete project artifacts directory $d ?"; then
        rm -rf "$d"
      fi
    fi
  done
fi

# 6) Optionally uninstall/remove ollama binary/package
if [[ $FULL -eq 1 && $MODELS_ONLY -eq 0 ]]; then
  if [[ -n "$BIN_PATH" ]]; then
    echo "[info] attempting binary/package removal (requires sudo)"
    if confirm "Remove ollama package/binary? This may require sudo."; then
      # Try apt (Codespaces often has sudo)
      if command -v sudo >/dev/null 2>&1 && command -v apt >/dev/null 2>&1; then
        sudo apt remove -y ollama || true
        sudo apt purge -y ollama || true
      fi
      # Fallback: remove the binary directly
      if command -v sudo >/dev/null 2>&1; then
        sudo rm -f "$BIN_PATH" || true
      else
        rm -f "$BIN_PATH" || true
      fi
    fi
  else
    echo "[info] ollama binary not in PATH; skipping binary removal"
  fi
fi

echo "[info] cleanup complete."
echo "[info] disk usage after cleanup:"
df -h | sed -n '1,15p'
echo "[info] memory:"
free -h || true