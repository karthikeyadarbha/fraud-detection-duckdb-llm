#!/usr/bin/env python3
"""
DEPRECATED: Use scripts/level1_baseline_train.py instead.

This compatibility shim forwards execution to the new filename while emitting
a deprecation warning, so existing pipelines do not break during migration.
"""
from __future__ import annotations

import runpy
import warnings
from pathlib import Path

def main() -> int:
    warnings.warn(
        "scripts/level1_train.py is deprecated; use scripts/level1_baseline_train.py",
        DeprecationWarning,
        stacklevel=2,
    )
    target = Path(__file__).with_name("level1_baseline_train.py")
    # Execute the target script as if it were run directly
    runpy.run_path(str(target), run_name="__main__")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())