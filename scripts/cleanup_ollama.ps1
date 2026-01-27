#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Cleanup Ollama from a Windows/PowerShell environment:
  - Stops ollama processes
  - Removes models/cache under $HOME\.ollama
  - Optionally removes the ollama.exe if found
  - Optionally cleans Level-3 artifacts in this repo

.PARAMETER Yes
  Non-interactive (assume Yes to prompts)

.PARAMETER Full
  Also remove ollama.exe (if found)

.PARAMETER ModelsOnly
  Only remove models/cache (keep binary)

.PARAMETER Artifacts
  Also delete artifacts/level3_* directories in this repo

.EXAMPLE
  ./scripts/cleanup_ollama.ps1 -Yes -ModelsOnly
#>
param(
  [switch]$Yes = $false,
  [switch]$Full = $false,
  [switch]$ModelsOnly = $false,
  [switch]$Artifacts = $false
)

function Confirm-Action($Message) {
  if ($Yes) { return $true }
  $resp = Read-Host "$Message [y/N]"
  return @('y','yes') -contains $resp.ToLower()
}

Write-Host "[info] cleanup Ollama (PowerShell)"

# 1) Stop processes
$procs = Get-Process -Name ollama -ErrorAction SilentlyContinue
if ($procs) {
  if (Confirm-Action "Stop ollama processes?") {
    $procs | Stop-Process -Force -ErrorAction SilentlyContinue
  }
}

# 2) Remove models/cache
$home = $env:USERPROFILE
$ollamaHome = "$home\.ollama"
$paths = @()
if ($env:OLLAMA_MODELS) { $paths += $env:OLLAMA_MODELS }
$paths += "$ollamaHome\models", $ollamaHome

foreach ($p in $paths) {
  if (Test-Path $p) {
    if (Confirm-Action "Delete directory $p ?") {
      Remove-Item -Recurse -Force $p -ErrorAction SilentlyContinue
    }
  }
}

# 3) Remove temp log
$log = "$env:TEMP\ollama_serve.log"
if (Test-Path $log) {
  if (Confirm-Action "Delete $log ?") {
    Remove-Item -Force $log -ErrorAction SilentlyContinue
  }
}

# 4) Artifacts
if ($Artifacts) {
  $dirs = @("artifacts/level3_llm_run","artifacts/level3_llm_warm","artifacts/level3_llm_smoke")
  foreach ($d in $dirs) {
    if (Test-Path $d) {
      if (Confirm-Action "Delete $d ?") {
        Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue
      }
    }
  }
}

# 5) Optionally remove binary
if ($Full -and -not $ModelsOnly) {
  $bin = (Get-Command ollama -ErrorAction SilentlyContinue).Path
  if ($bin) {
    if (Confirm-Action "Remove ollama binary at $bin ? (may need admin)") {
      try {
        Remove-Item -Force $bin -ErrorAction SilentlyContinue
      } catch {}
    }
  } else {
    Write-Host "[info] ollama not found in PATH"
  }
}

Write-Host "[info] cleanup complete."