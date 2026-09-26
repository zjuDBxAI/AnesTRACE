"""Single-agent LangGraph for one partially observable Level Three turn."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage,
)
from langgraph.graph import END, START, StateGraph, add_messages

from .config import AgentSettings
from .prompts import FORCE_ANSWER_PROMPT, SYSTEM_PROMPT, render_repair_prompt, render_turn_prompt
from .schemas import DecisionMemory, DecisionOutput, ModelCallTrace, ToolTrace, TurnObservation
from .tools import RuntimeTool, compact_tool_result


class AgentState(TypedDict, total=False):
    run_id: str
    dataset_source: str
    episode_id: str
    procedure_name: str
    patient_profile: str
    procedure_anesthesia_context: str
    current_turn: dict[str, Any]
    episode_memory: list[dict[str, Any]]
    messages: Annotated[list[AnyMessage], add_messages]
    prompt_sha256: str
    memory_before: list[dict[str, Any]]
    tool_rounds: int
    tool_call_count: int
    context_tool_call_count: int
    knowledge_tool_call_count: int
    called_context_tools: list[str]
    current_tool_traces: list[dict[str, Any]]
    current_model_traces: list[dict[str, Any]]
    decision_output: dict[str, Any] | None
    invalid_content: str
    validation_error: str | None
    validation_status: str
    turn_result: dict[str, Any]


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts: list[str] = []
        for item in message.content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(message.content or "")


def _parse_output(content: str) -> DecisionOutput:
    text = content.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    return DecisionOutput.model_validate(payload)


def _usage_trace(response: AIMessage, phase: str, elapsed: float) -> dict[str, Any]:
    usage = dict(response.usage_metadata or {})
    token_usage = response.response_metadata.get("token_usage") or {}
    return ModelCallTrace(
        phase=phase,
        elapsed_seconds=elapsed,
        input_tokens=usage.get("input_tokens") or token_usage.get("prompt_tokens"),
        output_tokens=usage.get("output_tokens") or token_usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens") or token_usage.get("total_tokens"),
        response_id=response.response_metadata.get("id"),
    ).model_dump()


def _safe_component(value: str) -> str:
    cleaned = _SAFE_FILE_RE.sub("_", value).strip("._")
    return cleaned[:100] or "unknown"


def build_graph(
    model: Any,
    runtime_tools: dict[str, RuntimeTool],
    agent_settings: AgentSettings,
    run_dir: Path,
    *,
    checkpointer: Any = None,
):
    """Compile a single-agent graph. One invocation processes one decision point."""

    exposed_tools = [item.tool for item in runtime_tools.values()]
    bound_model = model.bind_tools(exposed_tools, tool_choice="auto")
    tool_results_dir = run_dir / "tool_results"
    tool_results_dir.mkdir(parents=True, exist_ok=True)

    async def prepare_turn(state: AgentState) -> dict[str, Any]:
        turn = TurnObservation.model_validate(state["current_turn"])
        memory = [DecisionMemory.model_validate(item) for item in state.get("episode_memory", [])]
        prompt = render_turn_prompt(state["procedure_name"], turn)
        removals = [
            RemoveMessage(id=message.id)
            for message in state.get("messages", [])
            if getattr(message, "id", None)
        ]
        return {
            "messages": removals + [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "memory_before": [item.model_dump() for item in memory],
            "tool_rounds": 0,
            "tool_call_count": 0,
            "context_tool_call_count": 0,
            "knowledge_tool_call_count": 0,
            "called_context_tools": [],
            "current_tool_traces": [],
            "current_model_traces": [],
            "decision_output": None,
            "invalid_content": "",
            "validation_error": None,
            "validation_status": "pending",
        }

    async def decision_agent(state: AgentState) -> dict[str, Any]:
        started = time.monotonic()
        response = await bound_model.ainvoke(state["messages"])
        elapsed = time.monotonic() - started
        traces = list(state.get("current_model_traces", []))
        traces.append(_usage_trace(response, "decision", elapsed))
        return {"messages": [response], "current_model_traces": traces}

    def route_agent(state: AgentState) -> str:
        message = state["messages"][-1]
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            return "validate_answer"
        if (
            state.get("tool_rounds", 0) >= agent_settings.max_tool_rounds_per_turn
            or state.get("tool_call_count", 0) >= agent_settings.max_total_tool_calls_per_turn
        ):
            return "reject_budget"
        return "tool_executor"

    async def execute_one(
        state: AgentState,
        call: dict[str, Any],
        rejection: Literal["budget_rejected", "duplicate_rejected"] | None = None,
        rejection_reason: str | None = None,
    ) -> tuple[ToolMessage, dict[str, Any]]:
        call_id = str(call.get("id") or "missing-call-id")
        name = str(call.get("name") or "")
        raw_args = call.get("args") or {}
        runtime = runtime_tools.get(name)
        tool_class = runtime.tool_class if runtime else "knowledge"
        section = runtime.information_section if runtime else None
        started_at = _utc_now()
        started = time.monotonic()
        if rejection is not None:
            error = rejection_reason or "Tool call rejected."
            trace = ToolTrace(
                call_id=call_id,
                tool_name=name or "unknown",
                tool_class=tool_class,
                information_section=section,
                arguments=dict(raw_args) if isinstance(raw_args, dict) else {},
                status=rejection,
                started_at=started_at,
                elapsed_seconds=0.0,
                error=error,
            )
            return ToolMessage(content=error, tool_call_id=call_id, name=name), trace.model_dump()
        try:
            if runtime is None:
                raise ValueError(f"Unknown tool: {name}")
            parsed_args = runtime.tool.args_schema.model_validate(raw_args).model_dump()
            full_result = await asyncio.to_thread(runtime.execute, parsed_args, state)
            elapsed = time.monotonic() - started
            serialized = json.dumps(full_result, ensure_ascii=False, default=str, sort_keys=True)
            digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            turn = TurnObservation.model_validate(state["current_turn"])
            episode_dir = tool_results_dir / hashlib.sha256(
                state["episode_id"].encode("utf-8")
            ).hexdigest()[:16]
            episode_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{_safe_component(turn.ordinal)}_{_safe_component(call_id)}_"
                f"{_safe_component(name)}.json"
            )
            result_path = episode_dir / filename
            result_path.write_text(serialized + "\n", encoding="utf-8")
            compact = compact_tool_result(full_result, agent_settings.max_tool_observation_chars)
            trace = ToolTrace(
                call_id=call_id,
                tool_name=name,
                tool_class=runtime.tool_class,
                information_section=runtime.information_section,
                arguments=parsed_args,
                status="success",
                started_at=started_at,
                elapsed_seconds=elapsed,
                result_sha256=digest,
                result_path=str(result_path.relative_to(run_dir)),
                compact_result=compact,
            )
            return (
                ToolMessage(
                    content=json.dumps(compact, ensure_ascii=False, default=str),
                    tool_call_id=call_id,
                    name=name,
                ),
                trace.model_dump(),
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            error = f"{type(exc).__name__}: {exc}"
            trace = ToolTrace(
                call_id=call_id,
                tool_name=name or "unknown",
                tool_class=tool_class,
                information_section=section,
                arguments=dict(raw_args) if isinstance(raw_args, dict) else {},
                status="error",
                started_at=started_at,
                elapsed_seconds=elapsed,
                error=error,
            )
            return ToolMessage(content=f"Tool error: {error}", tool_call_id=call_id, name=name), trace.model_dump()

    async def tool_executor(state: AgentState) -> dict[str, Any]:
        message = state["messages"][-1]
        calls = list(getattr(message, "tool_calls", None) or [])
        total = state.get("tool_call_count", 0)
        context_count = state.get("context_tool_call_count", 0)
        knowledge_count = state.get("knowledge_tool_call_count", 0)
        called_context = set(state.get("called_context_tools", []))
        planned: list[tuple[dict[str, Any], str | None, str | None]] = []
        for call in calls:
            name = str(call.get("name") or "")
            runtime = runtime_tools.get(name)
            tool_class = runtime.tool_class if runtime else "knowledge"
            rejection: str | None = None
            reason: str | None = None
            if tool_class == "context" and name in called_context:
                rejection = "duplicate_rejected"
                reason = "This context section was already retrieved in the current turn."
            elif total >= agent_settings.max_total_tool_calls_per_turn:
                rejection = "budget_rejected"
                reason = "The total per-turn tool-call budget is exhausted."
            elif tool_class == "context" and context_count >= agent_settings.max_context_tool_calls_per_turn:
                rejection = "budget_rejected"
                reason = "The per-turn context-tool budget is exhausted."
            elif tool_class == "knowledge" and knowledge_count >= agent_settings.max_knowledge_tool_calls_per_turn:
                rejection = "budget_rejected"
                reason = "The per-turn knowledge-tool budget is exhausted."
            else:
                total += 1
                if tool_class == "context":
                    context_count += 1
                    called_context.add(name)
                else:
                    knowledge_count += 1
            planned.append((call, rejection, reason))
        results = await asyncio.gather(
            *(execute_one(state, call, rejection, reason) for call, rejection, reason in planned)
        )
        messages, traces = zip(*results) if results else ((), ())
        return {
            "messages": list(messages),
            "current_tool_traces": list(state.get("current_tool_traces", [])) + list(traces),
            "tool_call_count": total,
            "context_tool_call_count": context_count,
            "knowledge_tool_call_count": knowledge_count,
            "called_context_tools": sorted(called_context),
            "tool_rounds": state.get("tool_rounds", 0) + 1,
        }

    async def reject_budget(state: AgentState) -> dict[str, Any]:
        message = state["messages"][-1]
        calls = list(getattr(message, "tool_calls", None) or [])
        results = await asyncio.gather(
            *(
                execute_one(
                    state,
                    call,
                    "budget_rejected",
                    "Tool call rejected because the per-turn budget was exhausted.",
                )
                for call in calls
            )
        )
        messages, traces = zip(*results) if results else ((), ())
        return {
            "messages": list(messages),
            "current_tool_traces": list(state.get("current_tool_traces", [])) + list(traces),
        }

    async def forced_answer(state: AgentState) -> dict[str, Any]:
        started = time.monotonic()
        response = await model.ainvoke(
            list(state["messages"]) + [HumanMessage(content=FORCE_ANSWER_PROMPT)]
        )
        elapsed = time.monotonic() - started
        traces = list(state.get("current_model_traces", []))
        traces.append(_usage_trace(response, "forced_answer", elapsed))
        return {"messages": [response], "current_model_traces": traces}

    async def validate_answer(state: AgentState) -> dict[str, Any]:
        content = _content_text(state["messages"][-1])
        try:
            output = _parse_output(content)
        except Exception as exc:
            return {
                "invalid_content": content,
                "validation_error": f"{type(exc).__name__}: {exc}",
                "validation_status": "needs_repair",
                "decision_output": None,
            }
        return {
            "decision_output": output.public_dict(),
            "validation_error": None,
            "validation_status": "success",
        }

    def route_validation(state: AgentState) -> str:
        return "update_episode_memory" if state["validation_status"] == "success" else "repair_answer"

    async def repair_answer(state: AgentState) -> dict[str, Any]:
        prompt = render_repair_prompt(
            state.get("invalid_content", ""), state.get("validation_error") or "unknown"
        )
        started = time.monotonic()
        response = await model.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        elapsed = time.monotonic() - started
        traces = list(state.get("current_model_traces", []))
        traces.append(_usage_trace(response, "repair", elapsed))
        content = _content_text(response)
        try:
            output = _parse_output(content)
        except Exception as exc:
            return {
                "messages": [response],
                "invalid_content": content,
                "validation_error": f"{type(exc).__name__}: {exc}",
                "validation_status": "failed_schema_validation",
                "decision_output": None,
                "current_model_traces": traces,
            }
        return {
            "messages": [response],
            "decision_output": output.public_dict(),
            "validation_error": None,
            "validation_status": "success_after_repair",
            "current_model_traces": traces,
        }

    async def update_episode_memory(state: AgentState) -> dict[str, Any]:
        memory = [DecisionMemory.model_validate(item) for item in state.get("episode_memory", [])]
        if state.get("decision_output"):
            output = DecisionOutput.model_validate(state["decision_output"])
            turn = TurnObservation.model_validate(state["current_turn"])
            memory.append(DecisionMemory.from_output(turn.ordinal, output))
        return {"episode_memory": [item.model_dump() for item in memory]}

    async def persist_turn_result(state: AgentState) -> dict[str, Any]:
        turn = TurnObservation.model_validate(state["current_turn"])
        direct_input = {
            "procedure_name": state["procedure_name"],
            "decision_point": turn.ordinal,
            "timeline_start": turn.timeline_start,
            "seconds_after_timeline_start": turn.seconds_after_timeline_start,
            "seconds_after_previous": turn.seconds_after_previous,
            "vital_sign_trends": turn.vital_sign_trends,
            "question": turn.question,
        }
        traces = list(state.get("current_tool_traces", []))
        accessed_sections = sorted(
            {
                item["information_section"]
                for item in traces
                if item.get("tool_class") == "context"
                and item.get("status") == "success"
                and item.get("information_section")
            }
        )
        result = {
            "schema_version": "anestrace-level-three-prediction.v2-en-agent",
            "run_id": state["run_id"],
            "dataset_source": state["dataset_source"],
            "episode_id": state["episode_id"],
            "turn_id": turn.turn_id,
            "decision_point": turn.ordinal,
            "input": {"direct_input": direct_input},
            "accessed_context_sections": accessed_sections,
            "prompt_sha256": state["prompt_sha256"],
            "status": state["validation_status"],
            "prediction": state.get("decision_output"),
            "memory_before": state.get("memory_before", []),
            "memory_after": state.get("episode_memory", []),
            "tool_call_count": state.get("tool_call_count", 0),
            "context_tool_call_count": state.get("context_tool_call_count", 0),
            "knowledge_tool_call_count": state.get("knowledge_tool_call_count", 0),
            "tool_rounds": state.get("tool_rounds", 0),
            "tool_calls": traces,
            "model_calls": state.get("current_model_traces", []),
            "validation_error": state.get("validation_error"),
            "created_at": _utc_now(),
        }
        return {"turn_result": result}

    builder = StateGraph(AgentState)
    builder.add_node("prepare_turn", prepare_turn)
    builder.add_node("decision_agent", decision_agent)
    builder.add_node("tool_executor", tool_executor)
    builder.add_node("reject_budget", reject_budget)
    builder.add_node("forced_answer", forced_answer)
    builder.add_node("validate_answer", validate_answer)
    builder.add_node("repair_answer", repair_answer)
    builder.add_node("update_episode_memory", update_episode_memory)
    builder.add_node("persist_turn_result", persist_turn_result)
    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "decision_agent")
    builder.add_conditional_edges(
        "decision_agent",
        route_agent,
        {
            "tool_executor": "tool_executor",
            "reject_budget": "reject_budget",
            "validate_answer": "validate_answer",
        },
    )
    builder.add_edge("tool_executor", "decision_agent")
    builder.add_edge("reject_budget", "forced_answer")
    builder.add_edge("forced_answer", "validate_answer")
    builder.add_conditional_edges(
        "validate_answer",
        route_validation,
        {"update_episode_memory": "update_episode_memory", "repair_answer": "repair_answer"},
    )
    builder.add_edge("repair_answer", "update_episode_memory")
    builder.add_edge("update_episode_memory", "persist_turn_result")
    builder.add_edge("persist_turn_result", END)
    return builder.compile(checkpointer=checkpointer)
