from pydantic import ValidationError
import pytest

from anestrace_agent.cli import _parser
from anestrace_agent.config import ModelSettings, PROJECT_ROOT, load_settings


def test_level_two_aligned_qwen35_generation_profile():
    settings = load_settings()
    model = settings.model
    assert model.model == "Qwen3.5-27B"
    assert model.temperature == 0.0
    assert model.top_p == 1.0
    assert model.top_k == 0
    assert model.seed == 42
    assert model.enable_thinking is False
    assert model.max_tokens == 3072


def test_repository_config_matches_packaged_default():
    assert load_settings(PROJECT_ROOT / "configs/default.json") == load_settings()


def test_thinking_requires_nonzero_temperature():
    with pytest.raises(ValidationError):
        ModelSettings(
            api_base="http://127.0.0.1:8002/v1",
            model="test",
            enable_thinking=True,
            temperature=0,
        )


def test_cli_exposes_level_two_generation_overrides():
    args = _parser().parse_args(
        [
            "run",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--top-k",
            "0",
            "--seed",
            "42",
            "--max-tokens",
            "3072",
            "--no-enable-thinking",
        ]
    )
    assert args.temperature == 0
    assert args.top_p == 1
    assert args.top_k == 0
    assert args.seed == 42
    assert args.max_tokens == 3072
    assert args.enable_thinking is False
