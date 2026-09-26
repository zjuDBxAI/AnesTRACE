#!/usr/bin/env python3
"""Compatibility wrapper; use ``level3/inference/run_ablation.py``."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "level3/inference/run_ablation.py"),
        run_name="__main__",
    )
