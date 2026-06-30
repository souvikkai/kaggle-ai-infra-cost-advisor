"""
ADK Pipeline Runner
===================
Thin async wrapper around the ADK pipeline that:
  - Manages a process-lifetime InMemorySessionService (sessions survive
    multiple requests, lost on server restart)
  - Exposes run_parse_judge() and run_full_pipeline() for main.py
  - Maps ADK session state → structured API responses

Session lifecycle:
  - Frontend passes session_id (UUID string) on each request
  - First message: pass state_delta with any existing assumptions
  - Subsequent messages: session already has state; just send new_message
  - Session state keys: workload_spec, judge_verdict, judge_issues,
    clarifying_question, pricing_results, final_recommendation
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types

logger = logging.getLogger(__name__)

APP_NAME = "ai-infra-cost-advisor"
_session_service = InMemorySessionService()


def _make_runner(agent_or_workflow) -> Runner:
    return Runner(
        node=agent_or_workflow,
        app_name=APP_NAME,
        session_service=_session_service,
    )


async def _get_or_create_session(
    session_id: str,
    user_id: str,
    initial_state: dict[str, Any] | None = None,
) -> Session:
    try:
        session = await _session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        if session is not None:
            return session
    except Exception:
        pass
    return await _session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state or {},
    )


def _user_content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


async def _drain_events(runner: Runner, user_id: str, session_id: str, message: str) -> list[dict]:
    """Run the pipeline and collect all agent text outputs."""
    outputs = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=_user_content(message),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    outputs.append({"author": event.author, "text": part.text})
    return outputs


async def run_parse_judge(
    user_message: str,
    session_id: str | None = None,
    user_id: str = "default_user",
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run just the ParseJudgeLoop for one user turn.

    Returns a dict with:
      session_id      — use this in subsequent requests
      verdict         — "pass" | "needs_user" | "retry" (retry means circuit-breaker fired)
      workload_spec   — parsed spec dict (populated when verdict == "pass")
      clarifying_question — plain-English question (when verdict == "needs_user")
      judge_issues    — list of issues found by Judge
      agent_outputs   — list of {author, text} for debugging
    """
    from agents.pipeline import parse_judge_workflow

    sid = session_id or str(uuid.uuid4())
    state = {"user_message": user_message, **(initial_state or {})}
    await _get_or_create_session(sid, user_id, state)

    runner = _make_runner(parse_judge_workflow)
    outputs = await _drain_events(runner, user_id, sid, user_message)

    # Read final state
    session = await _session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=sid
    )
    state = session.state if session else {}

    verdict = state.get("judge_verdict", "needs_user")
    workload_spec_raw = state.get("workload_spec")
    workload_spec = _parse_json_field(workload_spec_raw)

    return {
        "session_id": sid,
        "verdict": verdict,
        "workload_spec": workload_spec,
        "clarifying_question": state.get("clarifying_question", ""),
        "judge_issues": state.get("judge_issues", []),
        "agent_outputs": outputs,
    }


async def run_full_pipeline(
    user_message: str,
    session_id: str | None = None,
    user_id: str = "default_user",
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the full pipeline: ParseJudgeLoop, then ONLY IF verdict == "pass",
    CostReasoningPipeline (PricingAgent → CostEngineBridge → ReasoningAgent).

    This is two separate Runner invocations, not one fused graph. That split
    is the actual gate: if the Judge says "retry" or "needs_user", pricing
    and reasoning never run at all — no MCP tool calls, no extra LLM calls,
    no wasted latency on a result the caller is going to discard anyway.

    Returns a dict with all pipeline results. If verdict != "pass", pricing_results,
    cost_scenarios, and final_recommendation are all None and clarifying_question
    is set instead.
    """
    from agents.pipeline import parse_judge_workflow, cost_reasoning_pipeline

    sid = session_id or str(uuid.uuid4())
    state = {"user_message": user_message, **(initial_state or {})}
    await _get_or_create_session(sid, user_id, state)

    # ── Stage 1: Parse + Judge only ──────────────────────────────────────
    parse_runner = _make_runner(parse_judge_workflow)
    parse_outputs = await _drain_events(parse_runner, user_id, sid, user_message)

    session = await _session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=sid
    )
    state = session.state if session else {}
    verdict = state.get("judge_verdict", "needs_user")
    workload_spec = _parse_json_field(state.get("workload_spec"))

    if verdict != "pass":
        # Gate: do NOT run PricingAgent/CostEngineBridge/ReasoningAgent.
        # Saves 3 MCP tool calls + 2 full LLM reasoning passes on every
        # clarification round.
        return {
            "session_id": sid,
            "verdict": verdict,
            "workload_spec": workload_spec,
            "clarifying_question": state.get("clarifying_question", ""),
            "judge_issues": state.get("judge_issues", []),
            "pricing_results": None,
            "cost_scenarios": None,
            "final_recommendation": None,
            "agent_outputs": parse_outputs,
        }

    # ── Stage 2: verdict == "pass" — run pricing + cost + reasoning ─────
    reasoning_runner = _make_runner(cost_reasoning_pipeline)
    reasoning_outputs = await _drain_events(reasoning_runner, user_id, sid, user_message)

    session = await _session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=sid
    )
    state = session.state if session else {}

    pricing_results = _parse_json_field(state.get("pricing_results"))
    final_recommendation = _parse_json_field(state.get("final_recommendation"))
    # cost_scenarios is populated by cost_engine_bridge inside the ADK pipeline
    cost_scenarios = _parse_json_field(state.get("cost_scenarios"))

    return {
        "session_id": sid,
        "verdict": verdict,
        "workload_spec": workload_spec,
        "clarifying_question": state.get("clarifying_question", ""),
        "judge_issues": state.get("judge_issues", []),
        "pricing_results": pricing_results,
        "cost_scenarios": cost_scenarios,
        "final_recommendation": final_recommendation,
        "agent_outputs": parse_outputs + reasoning_outputs,
    }


async def run_with_spec(
    workload_spec: dict[str, Any],
    user_id: str = "default_user",
) -> dict[str, Any]:
    """
    Run cost engine + reasoning on a pre-validated workload spec, bypassing
    the ParseJudgeLoop. Used when the user has corrected the parsed spec.
    """
    from agents.pipeline import cost_reasoning_pipeline

    sid = str(uuid.uuid4())
    initial_state = {
        "workload_spec": json.dumps(workload_spec),
        "judge_verdict": "pass",
    }
    await _get_or_create_session(sid, user_id, initial_state)

    runner = _make_runner(cost_reasoning_pipeline)
    outputs = await _drain_events(runner, user_id, sid, "recalculate with corrected spec")

    session = await _session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=sid
    )
    state = session.state if session else {}

    return {
        "session_id": sid,
        "verdict": "pass",
        "workload_spec": workload_spec,
        "cost_scenarios": _parse_json_field(state.get("cost_scenarios")),
        "final_recommendation": _parse_json_field(state.get("final_recommendation")),
        "agent_outputs": outputs,
    }


def _run_cost_engine(spec: dict[str, Any]) -> list[dict] | None:
    """
    Synchronous helper: run the deterministic cost engine for a given workload spec.
    Returns the list of scenarios, or None if required volume fields are missing.
    Used by cost_engine_bridge (inside the pipeline) and by test_runner.
    """
    monthly_queries = int(spec.get("monthly_queries") or 0)
    input_tokens = int(spec.get("input_tokens_per_query") or 0)
    output_tokens = int(spec.get("output_tokens_per_query") or 0)
    if not (monthly_queries and input_tokens and output_tokens):
        return None
    from backend.cost_engine import calculate_scenarios
    return calculate_scenarios(
        monthly_queries=monthly_queries,
        input_tokens_per_query=input_tokens,
        output_tokens_per_query=output_tokens,
    )


def _parse_json_field(value: Any) -> dict | list | None:
    """Parse a session state field that may be a JSON string or already a dict/list.
    Strips markdown fences (```json ... ```) before parsing."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        # Strip markdown fences — same approach as verdict_router
        start = text.find("{")
        arr_start = text.find("[")
        if arr_start != -1 and (start == -1 or arr_start < start):
            start = arr_start
            end = text.rfind("]")
        else:
            end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, (dict, list)) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None
