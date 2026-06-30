"""
ReAct Agent — AI Infra Cost Simulator
======================================
Implements the Reasoning + Acting (ReAct) loop for workload cost planning.

Architecture
------------
The agent is the agentic backbone of /plan. It follows the ReAct pattern:
  1. THOUGHT  — reason about what information is missing or needs clarifying
  2. ACTION   — call a tool to retrieve or compute something
  3. OBSERVATION — receive the tool result
  4. REPEAT   — until all required fields are resolved
  5. FINAL ANSWER — return structured assumptions ready for /simulate

Tools available to the agent
-----------------------------
  get_provider_pricing(provider_key)   — look up current GPU or API pricing
  estimate_token_volumes(description)  — infer token counts from workload text
  validate_assumptions(assumptions)    — check all required fields are present
  compute_quick_cost(assumptions)      — back-of-envelope cost for a sanity check

Design notes
------------
- The agent does NOT call /simulate directly. It prepares structured assumptions
  that the deterministic simulator (cost_engine.py) then evaluates.
- Tool calls are deterministic. Only the reasoning step calls the LLM.
- Each tool is a pure Python function — no external API required at tool-call time.
- The agent exits after MAX_TURNS to prevent infinite loops (circuit breaker).
- This module is consumed by llm_planner.py when USE_REACT_AGENT=true.

This pattern maps directly to the PM interview answer:
  "I built a ReAct agent where the model decides which tool to call at each
   step — the loop continues until all required assumptions are resolved or
   the circuit breaker fires. Tool execution is deterministic; only the
   reasoning step is probabilistic."
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_TURNS = 8          # circuit breaker — prevents infinite loops
TOOL_TIMEOUT_S = 5     # max seconds a single tool call may take

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pricing_snapshot.json"

REQUIRED_FIELDS = [
    "monthly_queries",
    "input_tokens_per_query",
    "output_tokens_per_query",
    "latency_sla_ms",
    "reasoning_complexity",
    "context_complexity",
    "coding_tool_use_intensity",
    "hallucination_sensitivity",
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations — all deterministic, no LLM calls
# ─────────────────────────────────────────────────────────────────────────────

def tool_get_provider_pricing(provider_key: str) -> dict:
    """
    Return current pricing for a named GPU provider or API model.
    Reads from the local pricing_snapshot.json — no network call.

    Args:
        provider_key: One of the keys in gpu_providers or models in
                      pricing_snapshot.json (e.g. 'coreweave_h100',
                      'claude_sonnet_4_6', 'gpt_5_5').

    Returns:
        Pricing dict with hourly_cost_per_gpu / input_per_million /
        output_per_million depending on provider type, plus metadata.
    """
    try:
        with open(DATA_PATH) as f:
            pricing = json.load(f)
    except FileNotFoundError:
        return {"error": f"pricing_snapshot.json not found at {DATA_PATH}"}

    gpu = pricing.get("gpu_providers", {}).get(provider_key)
    if gpu:
        return {
            "type": "gpu_provider",
            "provider_key": provider_key,
            **gpu,
        }

    model = pricing.get("models", {}).get(provider_key)
    if model:
        return {
            "type": "api_model",
            "model_key": provider_key,
            **model,
        }

    available = list(pricing.get("gpu_providers", {}).keys()) + list(pricing.get("models", {}).keys())
    return {
        "error": f"Unknown provider_key '{provider_key}'",
        "available_keys": available,
    }


def tool_estimate_token_volumes(description: str) -> dict:
    """
    Infer reasonable token volume estimates from a plain-English workload
    description. Returns point estimates with brief rationale.

    This is a deterministic heuristic, not an LLM call.

    Args:
        description: Plain-English workload description.

    Returns:
        Dict with input_tokens_per_query, output_tokens_per_query,
        and latency_sla_ms with rationale strings.
    """
    desc = description.lower()

    # Input token estimation
    if any(k in desc for k in ("codebase", "multi-document", "research", "large doc")):
        input_tokens = 8000
        input_rationale = "Large document / codebase context — 8K tokens typical"
    elif any(k in desc for k in ("rag", "retrieval", "document", "pdf")):
        input_tokens = 3000
        input_rationale = "RAG pipeline with retrieved context — 3K tokens typical"
    elif any(k in desc for k in ("code completion", "copilot", "coding")):
        input_tokens = 2000
        input_rationale = "Code completion with surrounding context — 2K tokens typical"
    elif any(k in desc for k in ("support", "chat", "conversation", "customer")):
        input_tokens = 700
        input_rationale = "Support / chat with short conversation history — 700 tokens typical"
    elif any(k in desc for k in ("image gen", "video gen", "text-to-image", "text-to-video")):
        input_tokens = 250
        input_rationale = "Generation prompt — 250 tokens typical"
    else:
        input_tokens = 700
        input_rationale = "General workload — 700 tokens as conservative baseline"

    # Output token estimation
    if any(k in desc for k in ("report", "summary", "long-form", "long form", "essay")):
        output_tokens = 800
        output_rationale = "Long-form output — 800 tokens typical"
    elif any(k in desc for k in ("code", "script", "function", "snippet")):
        output_tokens = 400
        output_rationale = "Code snippet output — 400 tokens typical"
    elif any(k in desc for k in ("support", "answer", "response")):
        output_tokens = 250
        output_rationale = "Support answer — 250 tokens typical"
    elif any(k in desc for k in ("image gen", "video gen")):
        output_tokens = 100
        output_rationale = "Job ID / metadata — 100 tokens typical"
    else:
        output_tokens = 250
        output_rationale = "General response — 250 tokens as conservative baseline"

    # Latency SLA estimation
    if any(k in desc for k in ("real-time", "sub-second", "autocomplete", "type-ahead")):
        latency_sla_ms = 800
        latency_rationale = "Real-time / autocomplete — 800ms SLA"
    elif any(k in desc for k in ("chat", "support", "interactive", "conversation")):
        latency_sla_ms = 3000
        latency_rationale = "Interactive chat — 3s SLA common"
    elif any(k in desc for k in ("batch", "async", "background", "offline")):
        latency_sla_ms = 30000
        latency_rationale = "Batch / async job — no strict real-time SLA"
    elif any(k in desc for k in ("image gen", "video gen")):
        latency_sla_ms = 15000
        latency_rationale = "Generation task — 15s SLA typical"
    else:
        latency_sla_ms = 3000
        latency_rationale = "General interactive workload — 3s SLA baseline"

    return {
        "input_tokens_per_query": input_tokens,
        "input_rationale": input_rationale,
        "output_tokens_per_query": output_tokens,
        "output_rationale": output_rationale,
        "latency_sla_ms": latency_sla_ms,
        "latency_rationale": latency_rationale,
    }


def tool_validate_assumptions(assumptions: dict) -> dict:
    """
    Check which required fields are present and which are still missing.
    Does NOT fill defaults — that is the agent's job.

    Args:
        assumptions: Current state of the structured assumptions dict.

    Returns:
        Dict with 'complete' bool, 'missing_fields' list, and 'present_fields' list.
    """
    missing = [f for f in REQUIRED_FIELDS if assumptions.get(f) is None]
    present = [f for f in REQUIRED_FIELDS if assumptions.get(f) is not None]
    return {
        "complete": len(missing) == 0,
        "missing_fields": missing,
        "present_fields": present,
        "completion_pct": round(len(present) / len(REQUIRED_FIELDS) * 100),
    }


def tool_compute_quick_cost(assumptions: dict) -> dict:
    """
    Back-of-envelope monthly cost estimate using the cheapest viable API model
    and the cheapest GPU provider. Used as a sanity check during planning.

    Requires: monthly_queries, input_tokens_per_query, output_tokens_per_query.
    Returns zeros if any required field is missing.

    Args:
        assumptions: Current structured assumptions.

    Returns:
        Dict with api_estimate and gpu_estimate in USD/month.
    """
    q = assumptions.get("monthly_queries")
    i = assumptions.get("input_tokens_per_query")
    o = assumptions.get("output_tokens_per_query")

    if not (q and i and o):
        return {
            "api_estimate_usd": None,
            "gpu_estimate_usd": None,
            "note": "Need monthly_queries, input_tokens_per_query, output_tokens_per_query to estimate cost.",
        }

    try:
        with open(DATA_PATH) as f:
            pricing = json.load(f)
    except FileNotFoundError:
        return {"error": "pricing_snapshot.json not found"}

    # Cheapest API model by output cost (dominant term at scale)
    api_models = pricing.get("models", {})
    cheapest_api_cost = None
    cheapest_api_name = None
    for key, model in api_models.items():
        cost = (q * i / 1_000_000) * model["input_per_million"] + (q * o / 1_000_000) * model["output_per_million"]
        if cheapest_api_cost is None or cost < cheapest_api_cost:
            cheapest_api_cost = round(cost, 2)
            cheapest_api_name = model.get("display_name", key)

    # Cheapest GPU provider (simple estimate at 70% utilization)
    gpu_providers = pricing.get("gpu_providers", {})
    cheapest_gpu_cost = None
    cheapest_gpu_name = None
    total_tokens = q * (i + o)
    seconds_per_month = 30 * 24 * 3600
    for key, provider in gpu_providers.items():
        tps_needed = total_tokens / seconds_per_month
        gpus = max(1, round(tps_needed / provider["tokens_per_second_per_gpu"] / 0.70 + 0.5))
        cost = round(gpus * provider["hourly_cost_per_gpu"] * 24 * 30, 2)
        if cheapest_gpu_cost is None or cost < cheapest_gpu_cost:
            cheapest_gpu_cost = cost
            cheapest_gpu_name = provider.get("display_name", key)

    return {
        "api_estimate_usd": cheapest_api_cost,
        "cheapest_api_model": cheapest_api_name,
        "gpu_estimate_usd": cheapest_gpu_cost,
        "cheapest_gpu_provider": cheapest_gpu_name,
        "note": "Quick estimate at 70% GPU utilization, no enterprise discount. Run /simulate for full scenarios.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — maps string names to callable functions
# ─────────────────────────────────────────────────────────────────────────────

TOOLS: dict[str, dict] = {
    "get_provider_pricing": {
        "fn": tool_get_provider_pricing,
        "description": (
            "Look up current pricing for a named GPU provider or API model. "
            "Use this when you need exact $/hr or $/M token costs. "
            "Args: provider_key (str) — e.g. 'coreweave_h100', 'claude_sonnet_4_6'."
        ),
        "required_args": ["provider_key"],
    },
    "estimate_token_volumes": {
        "fn": tool_estimate_token_volumes,
        "description": (
            "Infer reasonable input_tokens_per_query, output_tokens_per_query, and latency_sla_ms "
            "from a plain-English workload description. "
            "Use this when the user has not provided token counts explicitly. "
            "Args: description (str) — the user's workload description."
        ),
        "required_args": ["description"],
    },
    "validate_assumptions": {
        "fn": tool_validate_assumptions,
        "description": (
            "Check which required fields are present and which are still missing. "
            "Use this to decide whether to ask another clarifying question or proceed to simulation. "
            "Args: assumptions (dict) — current structured assumptions."
        ),
        "required_args": ["assumptions"],
    },
    "compute_quick_cost": {
        "fn": tool_compute_quick_cost,
        "description": (
            "Compute a back-of-envelope monthly cost estimate to sanity-check token volume assumptions. "
            "Use this when the user seems surprised by costs or when you want to validate the workload scale. "
            "Args: assumptions (dict) — current structured assumptions."
        ),
        "required_args": ["assumptions"],
    },
}

TOOL_NAMES_FOR_PROMPT = "\n".join(
    f"  {name}: {spec['description']}" for name, spec in TOOLS.items()
)


# ─────────────────────────────────────────────────────────────────────────────
# ReAct step dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReActStep:
    turn: int
    thought: str
    action: str | None = None
    action_input: dict | None = None
    observation: dict | None = None
    final_answer: dict | None = None
    error: str | None = None
    latency_ms: int = 0


@dataclass
class AgentResult:
    """
    Output of the ReAct agent run.
    Consumed by llm_planner.py to build the /plan response.
    """
    assumptions: dict = field(default_factory=dict)
    steps: list[ReActStep] = field(default_factory=list)
    turns_used: int = 0
    circuit_breaker_fired: bool = False
    missing_fields: list[str] = field(default_factory=list)
    ready_to_simulate: bool = False
    summary: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Tool executor — runs a named tool with timeout and structured error handling
# ─────────────────────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict) -> dict:
    """
    Execute a named tool with the given arguments.

    Returns the tool result dict, or an error dict if the tool raises
    or times out. This is the 'O' (Observation) in the ReAct loop.
    """
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": f"Unknown tool '{name}'. Available tools: {list(TOOLS.keys())}"}

    fn = spec["fn"]
    try:
        start = time.monotonic()
        result = fn(**args)
        elapsed_ms = round((time.monotonic() - start) * 1000)
        logger.debug("Tool %s completed in %dms", name, elapsed_ms)
        return result if isinstance(result, dict) else {"result": result}
    except TypeError as exc:
        return {"error": f"Tool '{name}' called with wrong arguments: {exc}"}
    except Exception as exc:
        logger.exception("Tool '%s' raised an unexpected error", name)
        return {"error": f"Tool '{name}' failed: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# ReAct loop (mock / unit-testable version — no LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

def run_react_loop_mock(
    user_message: str,
    existing_assumptions: dict | None = None,
) -> AgentResult:
    """
    Rule-based ReAct loop that mimics the T→A→O→repeat pattern without an LLM.

    Used when USE_REACT_AGENT=false or as a fallback when the LLM call fails.
    Each step follows the same thought/action/observation structure as the
    real LLM-driven loop, making it straightforward to swap in.

    Step sequence:
      Turn 1 — THOUGHT: what fields are missing?
               ACTION: validate_assumptions(current_assumptions)
               OBSERVATION: list of missing fields
      Turn 2 — THOUGHT: if token volumes are missing, estimate from description
               ACTION: estimate_token_volumes(user_message)
               OBSERVATION: token estimates
      Turn 3 — THOUGHT: validate again; if complete, compute quick cost sanity check
               ACTION: compute_quick_cost(assumptions)
               OBSERVATION: rough cost range
      FINAL — return structured assumptions with coverage summary
    """
    steps: list[ReActStep] = []
    assumptions: dict = {}

    # Seed from existing assumptions if provided
    if existing_assumptions:
        workload = existing_assumptions.get("workload") or {}
        complexity = existing_assumptions.get("complexity") or {}
        for fk in ("monthly_queries", "input_tokens_per_query", "output_tokens_per_query", "latency_sla_ms"):
            cell = workload.get(fk)
            val = cell.get("internal_value") if isinstance(cell, dict) else cell
            if val is not None:
                assumptions[fk] = val
        for fk in ("reasoning_complexity", "context_complexity", "coding_tool_use_intensity", "hallucination_sensitivity"):
            cell = complexity.get(fk)
            val = cell.get("internal_value") if isinstance(cell, dict) else cell
            if val is not None:
                assumptions[fk] = val

    # ── Turn 1: Validate current state ──
    t1_start = time.monotonic()
    t1 = ReActStep(
        turn=1,
        thought="I need to check which required fields are already present and which are missing before deciding what to do next.",
        action="validate_assumptions",
        action_input={"assumptions": assumptions},
    )
    t1.observation = execute_tool("validate_assumptions", {"assumptions": assumptions})
    t1.latency_ms = round((time.monotonic() - t1_start) * 1000)
    steps.append(t1)

    missing = t1.observation.get("missing_fields", [])
    logger.debug("Turn 1: %d missing fields: %s", len(missing), missing)

    # ── Turn 2: Estimate token volumes if not provided ──
    token_fields_missing = any(f in missing for f in ("input_tokens_per_query", "output_tokens_per_query", "latency_sla_ms"))
    if token_fields_missing:
        t2_start = time.monotonic()
        t2 = ReActStep(
            turn=2,
            thought=(
                "Token volume fields are missing. I'll call estimate_token_volumes "
                "to infer reasonable values from the workload description rather than "
                "using generic defaults."
            ),
            action="estimate_token_volumes",
            action_input={"description": user_message},
        )
        t2.observation = execute_tool("estimate_token_volumes", {"description": user_message})
        t2.latency_ms = round((time.monotonic() - t2_start) * 1000)
        steps.append(t2)

        obs = t2.observation
        if "input_tokens_per_query" in missing:
            assumptions["input_tokens_per_query"] = obs.get("input_tokens_per_query")
        if "output_tokens_per_query" in missing:
            assumptions["output_tokens_per_query"] = obs.get("output_tokens_per_query")
        if "latency_sla_ms" in missing:
            assumptions["latency_sla_ms"] = obs.get("latency_sla_ms")

    # ── Turn 3: Validate again and compute quick cost ──
    t3_start = time.monotonic()
    t3 = ReActStep(
        turn=3,
        thought=(
            "Fields have been updated. I'll validate again and compute a quick cost "
            "estimate to sanity-check whether the token volumes produce a plausible "
            "cost range before returning to the planner."
        ),
        action="compute_quick_cost",
        action_input={"assumptions": assumptions},
    )
    t3.observation = execute_tool("compute_quick_cost", {"assumptions": assumptions})
    t3.latency_ms = round((time.monotonic() - t3_start) * 1000)
    steps.append(t3)

    # Re-validate after all tool calls
    final_validation = execute_tool("validate_assumptions", {"assumptions": assumptions})
    missing_final = final_validation.get("missing_fields", [])
    ready = final_validation.get("complete", False)

    quick_cost = t3.observation
    api_est = quick_cost.get("api_estimate_usd")
    gpu_est = quick_cost.get("gpu_estimate_usd")
    cost_summary = (
        f"Quick cost estimate: API ~${api_est}/mo, self-hosted ~${gpu_est}/mo."
        if (api_est and gpu_est) else ""
    )

    summary = (
        f"Agent resolved {final_validation.get('completion_pct', 0)}% of required fields "
        f"in {len(steps)} turns. {cost_summary} "
        f"{'Ready for simulation.' if ready else f'Still missing: {missing_final}'}"
    ).strip()

    return AgentResult(
        assumptions=assumptions,
        steps=steps,
        turns_used=len(steps),
        circuit_breaker_fired=False,
        missing_fields=missing_final,
        ready_to_simulate=ready,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM-driven ReAct loop (called when USE_REACT_AGENT=true)
# ─────────────────────────────────────────────────────────────────────────────

def _build_react_system_prompt() -> str:
    return f"""You are an AI infrastructure cost planning agent. Your job is to collect
structured workload assumptions needed to simulate AI API and GPU infrastructure costs.

You follow the ReAct pattern: at each turn you THINK about what to do next, then ACT
by calling one tool, then OBSERVE the result. You repeat until all required fields are
resolved or you have enough to return a FINAL ANSWER.

Required fields: {REQUIRED_FIELDS}

Available tools:
{TOOL_NAMES_FOR_PROMPT}

Output format for each turn (strict JSON, no text outside it):
{{
  "thought": "<your reasoning about what to do next>",
  "action": "<tool_name or 'final_answer'>",
  "action_input": {{<args dict for the tool, or assumptions dict for final_answer>}}
}}

When action is 'final_answer', action_input should be the complete assumptions dict
with all available fields filled in. Set any truly unknown fields to null.

Rules:
- Never ask more than one question at a time in thought.
- Use estimate_token_volumes before defaulting to generic numbers.
- Use validate_assumptions to decide whether to continue or stop.
- Fire the final_answer action as soon as all required fields are present.
- Maximum {MAX_TURNS} turns — fire final_answer before hitting the limit.
"""


def run_react_loop_with_llm(
    user_message: str,
    existing_assumptions: dict | None = None,
    call_llm_fn=None,
) -> AgentResult:
    """
    LLM-driven ReAct loop. Requires a callable `call_llm_fn(prompt: str) -> str`.

    Each turn:
      1. Build a prompt with the current scratchpad (all prior T/A/O steps)
      2. Call the LLM to get the next thought + action
      3. Execute the tool (deterministic)
      4. Append the observation to the scratchpad
      5. Repeat until final_answer or MAX_TURNS

    Args:
        user_message: The user's workload description.
        existing_assumptions: Prior structured assumptions from the planner session.
        call_llm_fn: Callable that takes a prompt string and returns a response string.
                     Must be injected by the caller (llm_planner.py) to keep this
                     module free of direct LLM API dependencies.

    Returns:
        AgentResult with final assumptions and full step trace.
    """
    if call_llm_fn is None:
        logger.warning("call_llm_fn not provided; falling back to mock ReAct loop")
        return run_react_loop_mock(user_message, existing_assumptions)

    steps: list[ReActStep] = []
    scratchpad: list[str] = []
    assumptions: dict = {}
    circuit_breaker_fired = False

    # Seed assumptions from existing session state
    if existing_assumptions:
        workload = existing_assumptions.get("workload") or {}
        complexity = existing_assumptions.get("complexity") or {}
        for fk in ("monthly_queries", "input_tokens_per_query", "output_tokens_per_query", "latency_sla_ms"):
            cell = workload.get(fk)
            val = cell.get("internal_value") if isinstance(cell, dict) else cell
            if val is not None:
                assumptions[fk] = val
        for fk in ("reasoning_complexity", "context_complexity", "coding_tool_use_intensity", "hallucination_sensitivity"):
            cell = complexity.get(fk)
            val = cell.get("internal_value") if isinstance(cell, dict) else cell
            if val is not None:
                assumptions[fk] = val

    system_prompt = _build_react_system_prompt()
    user_context = (
        f"User workload description: {user_message}\n\n"
        f"Current assumptions: {json.dumps(assumptions, indent=2)}"
    )

    for turn in range(1, MAX_TURNS + 1):
        if turn > MAX_TURNS:
            circuit_breaker_fired = True
            logger.warning("ReAct agent circuit breaker fired at turn %d", turn)
            break

        scratchpad_text = "\n".join(scratchpad) if scratchpad else "(No prior steps)"
        prompt = (
            f"{user_context}\n\n"
            f"Prior steps:\n{scratchpad_text}\n\n"
            "What is your next thought and action? Respond with JSON only."
        )

        step = ReActStep(turn=turn, thought="")
        t_start = time.monotonic()

        try:
            raw_response = call_llm_fn(f"{system_prompt}\n\n{prompt}")
            parsed = json.loads(raw_response.strip())
        except (json.JSONDecodeError, Exception) as exc:
            step.error = f"LLM response parse error at turn {turn}: {exc}"
            step.latency_ms = round((time.monotonic() - t_start) * 1000)
            steps.append(step)
            logger.warning("ReAct LLM parse error at turn %d: %s", turn, exc)
            break

        step.thought = parsed.get("thought", "")
        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})
        step.action = action
        step.action_input = action_input

        if action == "final_answer":
            # Merge LLM-provided assumptions with anything already collected
            if isinstance(action_input, dict):
                for k, v in action_input.items():
                    if v is not None:
                        assumptions[k] = v
            step.final_answer = assumptions
            step.latency_ms = round((time.monotonic() - t_start) * 1000)
            steps.append(step)
            scratchpad.append(f"Turn {turn}: THOUGHT={step.thought} | ACTION=final_answer")
            logger.debug("ReAct agent reached final_answer at turn %d", turn)
            break

        # Execute the tool
        observation = execute_tool(action, action_input if isinstance(action_input, dict) else {})
        step.observation = observation
        step.latency_ms = round((time.monotonic() - t_start) * 1000)
        steps.append(step)

        # Merge any field values the tool returned
        for field_name in REQUIRED_FIELDS:
            if observation.get(field_name) is not None and assumptions.get(field_name) is None:
                assumptions[field_name] = observation[field_name]

        scratchpad.append(
            f"Turn {turn}: THOUGHT={step.thought} | ACTION={action}({action_input}) | "
            f"OBSERVATION={json.dumps(observation)[:200]}"
        )

    # Final validation
    final_validation = execute_tool("validate_assumptions", {"assumptions": assumptions})
    missing_final = final_validation.get("missing_fields", [])
    ready = final_validation.get("complete", False)
    completion_pct = final_validation.get("completion_pct", 0)

    summary = (
        f"ReAct agent completed {len(steps)} turns, resolved {completion_pct}% of required fields. "
        f"{'Ready for simulation.' if ready else f'Missing: {missing_final}'}"
        + (" CIRCUIT BREAKER FIRED." if circuit_breaker_fired else "")
    )

    return AgentResult(
        assumptions=assumptions,
        steps=steps,
        turns_used=len(steps),
        circuit_breaker_fired=circuit_breaker_fired,
        missing_fields=missing_final,
        ready_to_simulate=ready,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — called by llm_planner.py
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(
    user_message: str,
    existing_assumptions: dict | None = None,
    call_llm_fn=None,
) -> AgentResult:
    """
    Run the ReAct agent for a workload planning turn.

    Dispatches to the LLM-driven loop if call_llm_fn is provided,
    otherwise falls back to the rule-based mock loop.

    Args:
        user_message: The user's latest message.
        existing_assumptions: Structured assumptions from prior planner turns.
        call_llm_fn: Optional LLM callable injected from llm_planner.py.

    Returns:
        AgentResult with final assumptions, step trace, and readiness flag.
    """
    use_llm = call_llm_fn is not None and os.getenv("USE_REACT_AGENT", "false").lower() in ("true", "1", "yes")

    if use_llm:
        try:
            return run_react_loop_with_llm(user_message, existing_assumptions, call_llm_fn)
        except Exception as exc:
            logger.warning("LLM ReAct loop failed; falling back to mock: %s", exc)

    return run_react_loop_mock(user_message, existing_assumptions)
