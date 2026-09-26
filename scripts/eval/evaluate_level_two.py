#!/usr/bin/env python3
"""Compatibility wrapper; use ``level2/evaluation/evaluate.py``."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "level2/evaluation/evaluate.py"),
        run_name="__main__",
    )
