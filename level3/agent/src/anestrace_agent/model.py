"""OpenAI-compatible model construction."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import ModelSettings


def build_chat_model(settings: ModelSettings) -> ChatOpenAI:
    # Match the Level Two generation profile. For its default non-thinking
    # run, temperature=0 selects greedy decoding; top-p/top-k therefore do not
    # alter token selection, but remain in the run manifest for reproducibility.
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": settings.enable_thinking}
    }
    if settings.temperature > 0:
        extra_body["top_k"] = settings.top_k
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.api_base,
        api_key=settings.api_key(),
        temperature=settings.temperature,
        top_p=settings.top_p,
        seed=settings.seed,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        max_tokens=settings.max_tokens,
        extra_body=extra_body,
        streaming=False,
    )
