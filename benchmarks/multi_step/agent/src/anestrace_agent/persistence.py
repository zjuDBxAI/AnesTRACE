"""Single-writer JSONL output and run manifest helpers."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any


class AsyncJsonlStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None
        self.handles: dict[str, Any] = {}

    async def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.task = asyncio.create_task(self._writer())

    async def write(self, filename: str, record: dict[str, Any]) -> None:
        await self.queue.put((filename, record))

    async def close(self) -> None:
        await self.queue.put(None)
        if self.task is not None:
            await self.task

    async def _writer(self) -> None:
        try:
            while True:
                item = await self.queue.get()
                if item is None:
                    break
                filename, record = item
                handle = self.handles.get(filename)
                if handle is None:
                    path = self.run_dir / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handle = path.open("a", encoding="utf-8")
                    self.handles[filename] = handle
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            for handle in self.handles.values():
                handle.close()


def load_completed_predictions(
    path: Path, run_id: str
) -> dict[tuple[str, str, str], dict[str, Any]]:
    completed: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run_id") != run_id:
                continue
            key = (
                str(record["dataset_source"]),
                str(record["episode_id"]),
                str(record["decision_point"]),
            )
            if key in completed:
                raise ValueError(f"Duplicate completed prediction in resume file: {key}")
            completed[key] = record
    return completed


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)

