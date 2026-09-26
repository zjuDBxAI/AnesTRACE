"""Command-line interface for the English Level Three agent benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .adapters.level_three import DEFAULT_INPUT, episode_count_summary, load_episodes
from .benchmark_runner import run_benchmark
from .config import load_settings


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / f"run-{stamp}"


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"anestrace-{stamp}-{uuid.uuid4().hex[:8]}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anestrace-l3")
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate the agent dataset without model calls")
    validate.add_argument("--input", type=Path, default=DEFAULT_INPUT)

    run = commands.add_parser("run", help="Run sequential Level Three agent inference")
    run.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--run-id")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--limit", type=int)
    run.add_argument("--workers", type=int)
    run.add_argument("--api-base")
    run.add_argument("--model")
    run.add_argument("--api-key-env")
    run.add_argument("--temperature", type=float)
    run.add_argument("--top-p", type=float)
    run.add_argument("--top-k", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--max-tokens", type=int)
    run.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable model thinking mode (disabled in the Level Two-aligned profile).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = load_settings(args.config)
    if args.command == "validate":
        result = episode_count_summary(load_episodes(args.input))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.workers is not None:
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        settings.runner.episode_workers = args.workers
    if args.api_base:
        settings.model.api_base = args.api_base
    if args.model:
        settings.model.model = args.model
    if args.api_key_env:
        settings.model.api_key_env = args.api_key_env
    if args.temperature is not None:
        if args.temperature < 0:
            raise SystemExit("--temperature must be non-negative")
        settings.model.temperature = args.temperature
    if args.top_p is not None:
        if not 0 < args.top_p <= 1:
            raise SystemExit("--top-p must be in (0, 1]")
        settings.model.top_p = args.top_p
    if args.top_k is not None:
        if args.top_k < 0:
            raise SystemExit("--top-k must be non-negative")
        settings.model.top_k = args.top_k
    if args.seed is not None:
        settings.model.seed = args.seed
    if args.max_tokens is not None:
        if args.max_tokens < 1:
            raise SystemExit("--max-tokens must be at least 1")
        settings.model.max_tokens = args.max_tokens
    if args.enable_thinking is not None:
        settings.model.enable_thinking = args.enable_thinking
        if args.enable_thinking:
            if args.temperature is None:
                settings.model.temperature = 0.6
            if args.top_p is None:
                settings.model.top_p = 0.95
            if args.top_k is None:
                settings.model.top_k = 20
    # Re-validate cross-field constraints after applying CLI overrides.
    settings.model = type(settings.model).model_validate(settings.model.model_dump())
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    run_dir = (args.output_dir or _default_output_dir()).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit("--resume requires an existing run_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest["run_id"])
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit("output directory is not empty; use --resume or a new directory")
        run_id = args.run_id or _new_run_id()
    result = asyncio.run(
        run_benchmark(
            settings=settings,
            input_path=args.input,
            run_dir=run_dir,
            run_id=run_id,
            limit=args.limit,
            resume=args.resume,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
