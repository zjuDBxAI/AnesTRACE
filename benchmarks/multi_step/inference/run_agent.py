#!/usr/bin/env python3
"""Canonical launcher for the full AnesTRACE Level Three agent."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGENT_ROOT = PROJECT_ROOT / "benchmarks" / "multi_step" / "agent"
AGENT_SOURCE = AGENT_ROOT / "src"

if not AGENT_SOURCE.is_dir():
    raise RuntimeError(f"Level Three agent source does not exist: {AGENT_SOURCE}")
if str(AGENT_SOURCE) not in sys.path:
    sys.path.insert(0, str(AGENT_SOURCE))

from anestrace_agent.cli import main as agent_main


def main(argv: list[str] | None = None) -> int:
    return agent_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
