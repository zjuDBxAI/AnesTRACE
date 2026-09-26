#!/usr/bin/env python3
"""Compatibility wrapper; use ``level1/inference/run.py``."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "level1/inference/run.py"),
        run_name="__main__",
    )
