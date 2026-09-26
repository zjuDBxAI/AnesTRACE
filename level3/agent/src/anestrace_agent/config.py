"""Runtime configuration loaded from JSON plus environment-backed secrets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(os.environ.get("ANESTRACE_L3_HOME", Path.cwd())).expanduser().resolve()
DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.json")


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSettings(SettingsModel):
    api_base: str
    model: str
    api_key_env: str = "ANESTRACE_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: int = 42
    enable_thinking: bool = False
    timeout_seconds: float = Field(default=180, gt=0)
    max_retries: int = Field(default=3, ge=0)
    max_tokens: int = Field(default=3072, gt=0)

    @model_validator(mode="after")
    def validate_level_two_sampling_profile(self) -> "ModelSettings":
        if self.temperature == 0 and self.enable_thinking:
            raise ValueError(
                "thinking mode requires a positive temperature; the Level Two "
                "thinking profile uses temperature=0.6, top_p=0.95, and top_k=20"
            )
        return self

    def api_key(self) -> str:
        value = os.getenv(self.api_key_env)
        if value:
            return value
        hostname = (urlparse(self.api_base).hostname or "").casefold()
        if hostname in {"127.0.0.1", "localhost", "::1"}:
            return "EMPTY"
        raise RuntimeError(
            f"Remote model endpoint requires API key environment variable {self.api_key_env}"
        )


class AgentSettings(SettingsModel):
    max_tool_rounds_per_turn: int = Field(default=6, ge=0)
    max_context_tool_calls_per_turn: int = Field(default=6, ge=0)
    max_knowledge_tool_calls_per_turn: int = Field(default=4, ge=0)
    max_total_tool_calls_per_turn: int = Field(default=8, ge=0)
    max_tool_observation_chars: int = Field(default=8000, ge=1000)
    max_actions: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def validate_budget_relationships(self) -> "AgentSettings":
        if self.max_total_tool_calls_per_turn > (
            self.max_context_tool_calls_per_turn + self.max_knowledge_tool_calls_per_turn
        ):
            raise ValueError("total tool budget cannot exceed context plus knowledge budgets")
        return self


class ToolSettings(SettingsModel):
    top_k: int = Field(default=3, ge=1, le=5)
    network_enabled: bool = True
    cache_dir: str = "data/tool_cache"
    guideline_corpus_path: str | None = None
    miller_corpus_path: str | None = None


class RunnerSettings(SettingsModel):
    episode_workers: int = Field(default=4, ge=1)
    recursion_limit: int = Field(default=24, ge=10)


class AppSettings(SettingsModel):
    model: ModelSettings
    agent: AgentSettings
    tools: ToolSettings
    runner: RunnerSettings


def load_settings(path: str | Path | None = None) -> AppSettings:
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return AppSettings.model_validate(json.load(handle))


def settings_without_secrets(settings: AppSettings) -> dict[str, Any]:
    return settings.model_dump()
