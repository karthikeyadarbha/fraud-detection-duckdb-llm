#!/usr/bin/env python3
"""
Safe cleanup for Level-2 (anomaly) artifacts produced by older runs/versions.

Usage:
  # dry-run (shows files that would be removed)
  python scripts/cleanup_level2_artifacts.py --dry-run

  # actually delete
  python scripts/cleanup_level2_artifacts.py --yes
"""
import argparse
from pathlib import Path
import sys

PATTERNS = [
    "artifacts/ae_*",             # old autoencoder artifacts
    "artifacts/vae_ae*",          # old VAE artifacts
    "artifacts/pca_ae*",          # PCA AE artifacts
    "artifacts/isof_*",           # old isoForest artifacts
    "artifacts/*anomaly*.parquet",
    "artifacts/*anomaly*.csv",
    "artifacts/*recon_errors*.parquet",
    "artifacts/results_stream_anomaly_*.parquet",
    "artifacts/results_stream_anomaly_*.csv",
    "artifacts/manifest_ae_*",    # old manifests
    "artifacts/ae_anomaly_*.pkl",
]

def find_files():
    files = []
    for pat in PATTERNS:
        files.extend(Path(".").glob(pat))
    # dedupe and keep only files
    return sorted({p for p in files if p.exists() and p.is_file()})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="List files to be removed but don't delete")
    p.add_argument("--yes", action="store_true", help="Actually delete matched files")
    args = p.parse_args()

    files = find_files()
    if not files:
        print("No Level-2 artifact files matched patterns. Nothing to do.")
        return

    print("Matched files:")
    for f in files:
        print("  ", f)
    if args.dry_run:
        print("\nDry run only; no files deleted.")
        return

    if not args.yes:
        confirm = input("\nProceed to delete these files? Type YES to confirm: ")
        if confirm != "YES":
            print("Aborting - no files deleted.")
            return

    # delete
    deleted = 0
    for f in files:
        try:
            f.unlink()
            deleted += 1
            print("Deleted", f)
        except Exception as e:
            print("Failed to delete", f, ":", e)
    print(f"Deleted {deleted} files.")

if __name__ == "__main__":
    main()