"""Concurrent Episode runner with per-Episode sequential decision points."""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .adapters import load_episodes
from .config import AppSettings, settings_without_secrets
from .graph import build_graph
from .model import build_chat_model
from .persistence import AsyncJsonlStore, load_completed_predictions, write_json_atomic, write_jsonl_atomic
from .schemas import EpisodeInput
from .tools import build_runtime_tools


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _thread_id(run_id: str, source: str, episode_id: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{source}\0{episode_id}".encode("utf-8")).hexdigest()
    return f"anestrace-{digest}"


def _tool_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    class_status: Counter[str] = Counter()
    section_success: Counter[str] = Counter()
    latency: defaultdict[str, float] = defaultdict(float)
    empty_context = 0
    turns_without_context = 0
    failed_after_tools = 0
    model_tokens = 0
    for episode in summaries:
        for turn in episode["turns"]:
            traces = list(turn.get("tool_calls") or [])
            successful_context = [
                item for item in traces
                if item.get("tool_class") == "context" and item.get("status") == "success"
            ]
            if not successful_context:
                turns_without_context += 1
            if traces and not str(turn.get("status", "")).startswith("success"):
                failed_after_tools += 1
            for trace in traces:
                tool_class = str(trace.get("tool_class") or "unknown")
                status = str(trace.get("status") or "unknown")
                class_status[f"{tool_class}:{status}"] += 1
                latency[tool_class] += float(trace.get("elapsed_seconds") or 0)
                if tool_class == "context" and status == "success":
                    section = str(trace.get("information_section") or "unknown")
                    section_success[section] += 1
                    result = trace.get("compact_result") or {}
                    if (
                        result.get("actions") == []
                        or result.get("memory") == []
                        or "No new test results are available" in str(result.get("visible_test_results", ""))
                    ):
                        empty_context += 1
            for call in turn.get("model_calls") or []:
                model_tokens += int(call.get("total_tokens") or 0)
    return {
        "trace_status_by_tool_class": dict(sorted(class_status.items())),
        "successful_context_calls_by_section": dict(sorted(section_success.items())),
        "empty_context_results": empty_context,
        "turns_without_context_tool": turns_without_context,
        "failed_turns_after_any_tool": failed_after_tools,
        "tool_elapsed_seconds_by_class": {
            key: round(value, 6) for key, value in sorted(latency.items())
        },
        "model_total_tokens": model_tokens,
    }


async def run_benchmark(
    *,
    settings: AppSettings,
    input_path: Path,
    run_dir: Path,
    run_id: str,
    limit: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    episodes: list[EpisodeInput] = load_episodes(input_path, limit=limit)
    input_meta = {
        "path": str(input_path.resolve()),
        "sha256": sha256_file(input_path),
        "episodes": len(episodes),
        "turns": sum(len(item.turns) for item in episodes),
        "data_sources": dict(Counter(item.dataset_source for item in episodes)),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    completed = load_completed_predictions(predictions_path, run_id) if resume else {}
    manifest = {
        "schema_version": "anestrace-run-manifest.v2-en-agent",
        "run_id": run_id,
        "status": "running",
        "started_at": _utc_now(),
        "settings": settings_without_secrets(settings),
        "input": input_meta,
        "expected_episodes": len(episodes),
        "expected_turns": sum(len(item.turns) for item in episodes),
        "resume": resume,
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)

    store = AsyncJsonlStore(run_dir)
    await store.start()
    summaries: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(settings.runner.episode_workers)
    model = build_chat_model(settings.model)
    runtime_tools = build_runtime_tools(settings.tools)

    async with AsyncSqliteSaver.from_conn_string(str(run_dir / "checkpoints.sqlite")) as saver:
        graph = build_graph(model, runtime_tools, settings.agent, run_dir, checkpointer=saver)

        async def process_episode(episode: EpisodeInput) -> dict[str, Any]:
            async with semaphore:
                memory: list[dict[str, Any]] = []
                results: list[dict[str, Any]] = []
                for turn in episode.turns:
                    key = (episode.dataset_source, episode.episode_id, turn.ordinal)
                    if key in completed:
                        record = completed[key]
                        results.append(record)
                        memory = list(record.get("memory_after") or memory)
                        continue
                    state = await graph.ainvoke(
                        {
                            "run_id": run_id,
                            "dataset_source": episode.dataset_source,
                            "episode_id": episode.episode_id,
                            "procedure_name": episode.procedure_name,
                            "patient_profile": episode.patient_profile,
                            "procedure_anesthesia_context": episode.procedure_anesthesia_context,
                            "current_turn": turn.model_dump(),
                            "episode_memory": memory,
                        },
                        {
                            "configurable": {
                                "thread_id": _thread_id(
                                    run_id, episode.dataset_source, episode.episode_id
                                )
                            },
                            "recursion_limit": settings.runner.recursion_limit,
                        },
                    )
                    record = state["turn_result"]
                    results.append(record)
                    memory = list(record.get("memory_after") or memory)
                    await store.write("predictions.jsonl", record)
                    for trace in record.get("tool_calls") or []:
                        await store.write(
                            "tool_calls.jsonl",
                            {
                                "run_id": run_id,
                                "dataset_source": episode.dataset_source,
                                "episode_id": episode.episode_id,
                                "turn_id": turn.turn_id,
                                "decision_point": turn.ordinal,
                                **trace,
                            },
                        )
                return {
                    "schema_version": "anestrace-level-three-episode-result.v2-en-agent",
                    "run_id": run_id,
                    "dataset_source": episode.dataset_source,
                    "episode_id": episode.episode_id,
                    "split": episode.split,
                    "turn_count": len(results),
                    "successful_turns": sum(
                        1 for item in results if str(item.get("status", "")).startswith("success")
                    ),
                    "final_memory": memory,
                    "turns": results,
                }

        try:
            summaries = list(await asyncio.gather(*(process_episode(item) for item in episodes)))
        finally:
            await store.close()

    write_jsonl_atomic(run_dir / "episodes.jsonl", summaries)
    statuses: Counter[str] = Counter()
    total_model_calls = 0
    tool_trace_count = 0
    successful_tool_calls = 0
    for summary in summaries:
        for turn in summary["turns"]:
            statuses[str(turn.get("status") or "unknown")] += 1
            total_model_calls += len(turn.get("model_calls") or [])
            traces = list(turn.get("tool_calls") or [])
            tool_trace_count += len(traces)
            successful_tool_calls += sum(item.get("status") == "success" for item in traces)
    manifest.update(
        {
            "status": "completed",
            "completed_at": _utc_now(),
            "completed_episodes": len(summaries),
            "completed_turns": sum(item["turn_count"] for item in summaries),
            "turn_status_distribution": dict(statuses),
            "model_calls": total_model_calls,
            "tool_traces": tool_trace_count,
            "successful_tool_calls": successful_tool_calls,
            "tool_usage": _tool_summary(summaries),
        }
    )
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    return manifest
