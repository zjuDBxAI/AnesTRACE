"""Adapter for the tool-gated English Level Three agent JSONL."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from ..schemas import EpisodeInput, ObservedIntervention, TurnObservation


DEFAULT_INPUT = Path(
    os.environ.get("ANESTRACE_L3_DATA", "data/Level_three_v3_en_agent.jsonl")
).expanduser()
_ORDINAL_RE = re.compile(r"^T(\d+)$")


def _ordinal_number(value: str) -> int:
    match = _ORDINAL_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid decision point ordinal: {value!r}")
    return int(match.group(1))


def _adapt_turn(raw: dict[str, Any], question: str) -> TurnObservation:
    point = raw["decision_point"]
    inputs = raw["input"]
    previous = inputs["previous_intervention"]
    return TurnObservation(
        turn_id=str(raw["turn_id"]),
        ordinal=str(point["ordinal"]),
        timeline_start=point["timeline_start"],
        seconds_after_timeline_start=float(point["seconds_after_timeline_start"]),
        seconds_after_previous=(
            None
            if point["seconds_after_previous"] is None
            else float(point["seconds_after_previous"])
        ),
        anesthesia_medication_state=str(inputs["anesthesia_medication_state"]),
        vital_sign_trends=str(inputs["vital_sign_trends"]),
        observed_intervention=ObservedIntervention(
            source_decision_point=previous["source_decision_point"],
            actions=[str(item) for item in previous["actions"]],
        ),
        visible_test_results=str(inputs["visible_test_results"]["content"]),
        question=question,
    )


def _validate_episode(episode: EpisodeInput) -> None:
    previous_number = -1
    previous_elapsed = -1.0
    expected_timeline = (
        "operation_start"
        if episode.dataset_source == "VitalDB_INSPIRE"
        else "anesthesia_start"
    )
    for index, turn in enumerate(episode.turns):
        number = _ordinal_number(turn.ordinal)
        if number != index or number <= previous_number:
            raise ValueError(f"Decision points are not consecutive: {episode.episode_id}")
        if turn.timeline_start != expected_timeline:
            raise ValueError(f"Incorrect timeline origin: {episode.episode_id}/{turn.ordinal}")
        if turn.seconds_after_timeline_start < previous_elapsed:
            raise ValueError(f"Decision times are not monotonic: {episode.episode_id}")
        if index == 0:
            if turn.seconds_after_previous is not None:
                raise ValueError(f"T0 must not have a previous interval: {episode.episode_id}")
            if turn.observed_intervention.actions:
                raise ValueError(f"T0 must not expose a previous intervention: {episode.episode_id}")
        else:
            if turn.seconds_after_previous is None:
                raise ValueError(f"Missing previous interval: {episode.episode_id}/{turn.ordinal}")
            expected_source = f"T{index - 1}"
            if turn.observed_intervention.source_decision_point not in (None, expected_source):
                raise ValueError(
                    f"Previous intervention source does not match {expected_source}: "
                    f"{episode.episode_id}/{turn.ordinal}"
                )
            expected_elapsed = previous_elapsed + turn.seconds_after_previous
            if abs(expected_elapsed - turn.seconds_after_timeline_start) > 1e-4:
                raise ValueError(f"Inconsistent cumulative time: {episode.episode_id}/{turn.ordinal}")
        previous_number = number
        previous_elapsed = turn.seconds_after_timeline_start


def load_episodes(path: str | Path | None = None, *, limit: int | None = None) -> list[EpisodeInput]:
    """Load only whitelisted observation fields; labels and media are discarded."""

    input_path = Path(path or DEFAULT_INPUT)
    episodes: list[EpisodeInput] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("schema_version") != "anesbench-level-three.atomic-episode.v3-en-agent":
                raise ValueError(f"Unsupported schema version on line {line_number}")
            info = raw["patient_information"]
            question = str(raw["question"])
            episode = EpisodeInput(
                dataset_source=raw["data_source"],
                source_path=str(input_path.resolve()),
                source_line=line_number,
                episode_id=str(raw["episode_id"]),
                split=raw.get("split"),
                procedure_name=str(raw["procedure_name"]),
                patient_profile=str(info["patient_profile"]),
                procedure_anesthesia_context=str(info["procedure_anesthesia_context"]),
                turns=[_adapt_turn(item, question) for item in raw["turns"]],
            )
            _validate_episode(episode)
            episodes.append(episode)
            if limit is not None and len(episodes) >= limit:
                break
    return episodes


def episode_count_summary(episodes: Iterable[EpisodeInput]) -> dict[str, Any]:
    items = list(episodes)
    lengths: dict[str, int] = {}
    sources: dict[str, int] = {}
    for episode in items:
        key = str(len(episode.turns))
        lengths[key] = lengths.get(key, 0) + 1
        sources[episode.dataset_source] = sources.get(episode.dataset_source, 0) + 1
    return {
        "episodes": len(items),
        "turns": sum(len(item.turns) for item in items),
        "data_sources": dict(sorted(sources.items())),
        "episode_length_distribution": dict(sorted(lengths.items())),
    }
