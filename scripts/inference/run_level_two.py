#!/usr/bin/env python3
"""Compatibility wrapper; use ``level2/inference/run_extended.py``."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "level2/inference/run_extended.py"),
        run_name="__main__",
    )
