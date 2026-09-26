#!/usr/bin/env python3
"""Compatibility entrypoint for the Level 2 text-only runner."""

from level2.inference.run_text import *  # noqa: F401,F403
from level2.inference.run_text import main


if __name__ == "__main__":
    raise SystemExit(main())
