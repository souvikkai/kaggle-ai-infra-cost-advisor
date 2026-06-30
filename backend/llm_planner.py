"""
LLM / Agentic planning layer.

Intended architecture:
  user message
    → LLM planner (this module)
    → structured assumptions
    → deterministic simulator (/simulate, cost_engine)
    → explanation layer (recommendations, crossover, UI narrative)

Environment:
  USE_REAL_LLM_PLANNER=true|false  (default: false)
  LLM_PROVIDER=openai|anthropic    (default: openai)
  OPENAI_API_KEY                   (required when provider=openai)
  OPENAI_MODEL or LLM_MODEL        (default: gpt-4o-mini)
  LLM_BASE_URL                     (optional, OpenAI-compatible base URL)
  ANTHROPIC_API_KEY                (required when provider=anthropic)
  ANTHROPIC_MODEL                  (default: claude-haiku-4-5-20251001)
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

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

COMPLEXITY_FIELDS = {
    "reasoning_complexity",
    "context_complexity",
    "coding_tool_use_intensity",
    "hallucination_sensitivity",
}

REASONING_LEVELS = {"low", "medium", "high"}
CONTEXT_LEVELS = {"short-context", "medium-context", "long-context"}

CONTEXT_SIZE_CLARIFICATION = (
    "What is the typical context size for each request? "
    "Examples: short chat, long documents, large code files, multi-document research, "
    "or codebase-scale context."
)

FIELD_CLARIFICATIONS = {
    "monthly_queries": "How many user queries do you expect per month?",
    "latency_sla_ms": "What latency SLA are you targeting?",
    "reasoning_complexity": (
        "Is this workload simple Q&A, multi-step reasoning, coding/tool use, "
        "or high-stakes analysis?"
    ),
    "context_complexity": (
        "Is this workload simple Q&A, multi-step reasoning, coding/tool use, "
        "or high-stakes analysis?"
    ),
    "coding_tool_use_intensity": (
        "Is this workload simple Q&A, multi-step reasoning, coding/tool use, "
        "or high-stakes analysis?"
    ),
    "hallucination_sensitivity": (
        "Is this workload simple Q&A, multi-step reasoning, coding/tool use, "
        "or high-stakes analysis?"
    ),
}

TOKEN_FIELDS = {"input_tokens_per_query", "output_tokens_per_query"}

HUMAN_FIELD_LABELS = {
    "monthly_queries": "Monthly query volume",
    "input_tokens_per_query": "Input context estimate",
    "output_tokens_per_query": "Output length estimate",
    "latency_sla_ms": "Response time target",
    "reasoning_complexity": "Reasoning complexity",
    "context_complexity": "Context size profile",
    "coding_tool_use_intensity": "Coding / tool use intensity",
    "hallucination_sensitivity": "Accuracy sensitivity",
}

CONSERVATIVE_DEFAULTS = {
    "monthly_queries": 100_000,
    "input_tokens_per_query": 700,
    "output_tokens_per_query": 250,
    "latency_sla_ms": 3000,
    "reasoning_complexity": "medium",
    "context_complexity": "medium-context",
    "coding_tool_use_intensity": "low",
    "hallucination_sensitivity": "medium",
}

ASSUMPTION_MESSAGES = {
    "monthly_queries": "I'll assume about 100,000 monthly queries as a conservative early-stage baseline.",
    "input_tokens_per_query": "I'll assume medium input context (~700 tokens) typical for support-style chat.",
    "output_tokens_per_query": "I'll assume ~250 tokens of output per response unless you specify otherwise.",
    "latency_sla_ms": "I'll assume a 3 second response-time target, common for interactive chat.",
    "reasoning_complexity": "I'll assume medium reasoning complexity for now.",
    "context_complexity": "I'll assume medium context size for now. You can edit this later.",
    "coding_tool_use_intensity": "I'll assume low coding / tool-use intensity unless you need agents or code execution.",
    "hallucination_sensitivity": "I'll assume medium accuracy sensitivity — important answers but not life-critical.",
}

DEFAULT_OPERATIONAL_ASSUMPTIONS = {
    "gpu_utilization_pct": 70,
    "enterprise_api_discount_pct": 0,
    "burstiness_factor": "medium",
    "failover_reserve_pct": 15,
}

SYSTEM_INSTRUCTION = """You are a friendly, expert AI infrastructure cost planning assistant.

Your job is to help users estimate the cost of running AI workloads. You do this by having a
natural conversation to collect their workload details, then returning structured JSON.

== RESPONSE FORMAT ==
You must ALWAYS return a single JSON object. No text outside the JSON.

The JSON must have these exact keys:
workload_summary, monthly_queries, input_tokens_per_query, output_tokens_per_query,
latency_sla_ms, reasoning_complexity, context_complexity, coding_tool_use_intensity,
hallucination_sensitivity, missing_fields, clarification_questions, ready_to_simulate

== CONVERSATION FLOW ==
Follow this exact sequence before marking ready_to_simulate = true:

TURN 1 — Acknowledge what the user described. Ask the single most important missing question.
         Priority order: monthly volume → context/output size → latency → complexity profile.

TURN 2 — Acknowledge their answer. Ask the next most important missing question.
         Never skip ahead to assumptions while key facts are still unknown.

TURN 3+ — Once you have monthly volume and at least a sense of context/output size, you MAY
           fill remaining gaps with intelligent assumptions. BUT you must surface every single
           assumption explicitly in workload_summary before setting ready_to_simulate = true.
           The user must see what you assumed and why, so they can correct anything.

Never set ready_to_simulate = true before turn 3 unless the user has explicitly provided
ALL required fields themselves.

== HOW TO WRITE workload_summary ==
This is the message the user sees in the chat. Write it as a helpful, warm assistant.

1. FIRST MESSAGE (no existing assumptions):
   - Acknowledge what the user described in one sentence.
   - Ask the single most important missing question in plain English.
   - Never use technical jargon like "tokens" — say "how long are typical responses?" instead.
   - Example: "A customer support chatbot for a SaaS startup — great starting point! To model
     the costs accurately, I have one key question: roughly how many support conversations do
     you expect per month?"

2. FOLLOW-UP MESSAGES (existing assumptions present, still collecting):
   - Acknowledge what the user just told you in one sentence.
   - Ask the next most important missing question.
   - One question only — never stack multiple questions.
   - Example: "Got it — around 5,000 conversations per month. One more thing: how long do
     typical responses need to be? A short answer (1-2 sentences) or something more detailed?"

3. ASSUMPTION MESSAGE (turn 3+, filling remaining gaps):
   - Open with one sentence summarising what you now know.
   - Then list every assumption you are making, one per line, in plain English.
   - For each assumption, briefly explain WHY that value makes sense for this specific use case.
   - Close with: "These are my best estimates — you can adjust any of them before running
     the simulation."
   - Example format:
     "You're building a text-to-video platform expecting 20,000 generations per month.
     Here's what I'm filling in based on that:

     • Input length: I'll assume around 300 tokens per prompt — typical for detailed
       text-to-video instructions.
     • Output length: I'll assume around 150 tokens — this is mostly metadata and a
       job ID, since the actual video is generated separately.
     • Response time: I'll assume 10 seconds — video generation is slow by nature and
       users expect to wait.
     • Reasoning complexity: Medium — the model needs to interpret creative prompts
       but doesn't need multi-step logical reasoning.
     • Context size: Short — each generation request is self-contained, no long history.
     • Coding / tool use: Low — this is a creative generation workload, not agentic.
     • Accuracy sensitivity: Medium — quality matters but a slightly imperfect video
       is not catastrophic.

     These are my best estimates — you can adjust any of them before running the simulation."

4. CORRECTION COMMANDS ("make X high", "set output to 1000", "change reasoning to high"):
   - Confirm the change warmly and briefly.
   - Example: "Done — I've updated the output length to 1,000 tokens."
   - If still ready to simulate, add: "Everything looks good — click Run Simulation when ready."

5. UNCLEAR OR OFF-TOPIC MESSAGES:
   - Gently redirect back to workload planning.
   - Example: "Happy to help! To estimate your AI infrastructure costs, I need a few details
     about your workload. What kind of AI product are you building?"

6. NEVER:
   - Set ready_to_simulate = true before surfacing all assumptions to the user first
   - Silently fill in defaults without telling the user
   - Repeat back raw field names or JSON structure in the message
   - Ask more than one question at a time
   - Use the same generic defaults for every use case — reason from the actual workload

== INTELLIGENT USE-CASE-AWARE ASSUMPTIONS ==
When filling gaps at turn 3+, reason about what is realistic for THIS specific workload.
Do not use the same numbers for every use case. Use these as calibration anchors:

MONTHLY QUERIES — derive from what the user told you:
   - If they gave visitor/user numbers, multiply by expected generations per session
   - If they gave growth rates, use month-1 numbers and note the growth trajectory
   - Never default to 100,000 without reasoning from their actual numbers

INPUT TOKENS PER QUERY — based on what goes INTO each request:
   - Short text prompt (image/video gen): 100-400 tokens
   - Support chat message + short history: 400-1,000 tokens
   - Code completion with context: 1,000-4,000 tokens
   - RAG with retrieved documents: 2,000-8,000 tokens
   - Document summarisation: 3,000-12,000 tokens
   - Long research / multi-doc: 8,000-50,000 tokens

OUTPUT TOKENS PER QUERY — based on what comes OUT of each request:
   - Image/video job metadata or URL: 50-150 tokens
   - Short chat reply (1-3 sentences): 80-200 tokens
   - Detailed support answer: 150-400 tokens
   - Code snippet: 200-800 tokens
   - Long-form summary or report: 400-1,200 tokens

LATENCY SLA — based on user experience expectations:
   - Real-time chat / autocomplete: 500-1,500 ms
   - Interactive Q&A / support: 2,000-4,000 ms
   - Image generation (async): 5,000-15,000 ms
   - Video generation (async): 10,000-60,000 ms
   - Batch / background jobs: no strict SLA, use 30,000 ms

REASONING COMPLEXITY:
   - Simple lookup / FAQ / classification: low
   - Support chat, general Q&A, creative generation: medium
   - Code generation, multi-step analysis, agent tasks: high

CONTEXT COMPLEXITY:
   - Single-turn, self-contained requests: short-context
   - Multi-turn chat, moderate history: medium-context
   - RAG, long documents, codebase context: long-context

CODING / TOOL USE INTENSITY:
   - Pure generation or conversation: low
   - Copilot with some tool calls: medium
   - Agentic, multi-tool, code execution: high

HALLUCINATION SENSITIVITY:
   - Creative / entertainment / drafts: low
   - Support, general business use: medium
   - Legal, medical, financial, compliance: high

== FIELD EXTRACTION RULES ==
- Extract only information clearly stated or strongly implied by the user.
- When filling assumptions, pick values from the calibration anchors above — not generic defaults.
- Complexity enums:
    reasoning_complexity, coding_tool_use_intensity, hallucination_sensitivity: low | medium | high
    context_complexity: short-context | medium-context | long-context
- monthly_queries, input_tokens_per_query, output_tokens_per_query, latency_sla_ms: integer or null
- ready_to_simulate: true only when ALL required fields are non-null AND you have surfaced all
  assumptions to the user in workload_summary
- missing_fields: list of field names still null
- clarification_questions: array of plain-English questions (no jargon)

== CORRECTION COMMANDS ==
If the user says anything like "set X to high", "make X low", "change X to Y", "increase X to N",
"can you make X high", "update X to N" — you MUST update that field to the exact requested value.
The user's explicit instruction always overrides any prior assumption.

== ONE QUESTION AT A TIME ==
Never ask more than one question per response. Pick the most important missing field and ask
only about that. This keeps the conversation focused and not overwhelming.
"""


def use_real_llm_planner() -> bool:
    return os.getenv("USE_REAL_LLM_PLANNER", "false").strip().lower() in ("true", "1", "yes")


def _parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def _coerce_optional_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if cleaned.isdigit():
            return int(cleaned)
    return None


def _coerce_reasoning_level(value) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in REASONING_LEVELS else None


def _coerce_context_level(value) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in CONTEXT_LEVELS else None


_FIELD_ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"reasoning\s+complexity|reasoning"), "reasoning_complexity"),
    (re.compile(r"context\s+complexity|context\s+size|context\s+window"), "context_complexity"),
    (re.compile(r"coding[\s/]+tool[\s-]?use|tool[\s-]?use|coding"), "coding_tool_use_intensity"),
    (re.compile(r"hallucination\s+sensitivity|accuracy\s+sensitivity|hallucination"), "hallucination_sensitivity"),
    (re.compile(r"monthly\s+queries|query\s+volume|queries\s+per\s+month"), "monthly_queries"),
    (re.compile(r"latency\s+sla|response[\s-]?time"), "latency_sla_ms"),
]

_NUMERIC_FIELD_ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"output\s+(length|tokens?|size)|tokens?\s+(?:per\s+)?(?:response|output|query\s+out)"), "output_tokens_per_query"),
    (re.compile(r"input\s+(context|tokens?|size)|tokens?\s+(?:per\s+)?(?:input|query\s+in)"), "input_tokens_per_query"),
    (re.compile(r"monthly\s+queries|query\s+volume|requests?\s+per\s+month"), "monthly_queries"),
    (re.compile(r"latency\s+(?:sla|target)|response[\s-]?time"), "latency_sla_ms"),
]

_LEVEL_ALIASES: dict[str, str] = {
    "high": "high",
    "medium": "medium",
    "med": "medium",
    "low": "low",
    "short": "short-context",
    "short-context": "short-context",
    "short context": "short-context",
    "medium-context": "medium-context",
    "medium context": "medium-context",
    "long": "long-context",
    "long-context": "long-context",
    "long context": "long-context",
}

_OVERRIDE_VERBS = re.compile(
    r"\b(set|make|change|update|use|switch|put|mark|flag|keep|adjust|"
    r"can you (set|make|change|update|use)|please (set|make|change|use)|"
    r"i want|i'?d like|should be|needs? to be|has? to be)\b",
    re.IGNORECASE,
)


def _detect_explicit_overrides(user_message: str) -> dict:
    msg = user_message.lower()
    overrides: dict = {}
    has_override_signal = bool(_OVERRIDE_VERBS.search(msg))

    # Complexity fields (enum values)
    for field_pattern, field_key in _FIELD_ALIASES:
        match = field_pattern.search(msg)
        if not match:
            continue
        start = max(0, match.start() - 60)
        end = min(len(msg), match.end() + 60)
        surrounding = msg[start:end]
        for level_phrase, canonical in _LEVEL_ALIASES.items():
            if level_phrase in surrounding:
                if field_key in COMPLEXITY_FIELDS:
                    if field_key == "context_complexity":
                        if canonical in CONTEXT_LEVELS and has_override_signal:
                            overrides[field_key] = canonical
                            break
                    else:
                        if canonical in REASONING_LEVELS and has_override_signal:
                            overrides[field_key] = canonical
                            break

    # Numeric fields
    number_re = re.compile(r"\b(\d[\d,]*)\b")
    for field_pattern, field_key in _NUMERIC_FIELD_ALIASES:
        match = field_pattern.search(msg)
        if not match:
            continue
        start = max(0, match.start() - 80)
        end = min(len(msg), match.end() + 80)
        surrounding = msg[start:end]
        num_match = number_re.search(surrounding)
        if not num_match:
            continue
        raw_num = _parse_int(num_match.group(1))
        plausible = (
            (field_key == "output_tokens_per_query" and 10 <= raw_num <= 100_000) or
            (field_key == "input_tokens_per_query" and 10 <= raw_num <= 100_000) or
            (field_key == "monthly_queries" and raw_num >= 100) or
            (field_key == "latency_sla_ms" and 50 <= raw_num <= 60_000)
        )
        if plausible and has_override_signal:
            overrides[field_key] = raw_num

    return overrides


def _is_bare_assignment(text: str, field_pattern: re.Pattern, level: str) -> bool:
    return len(text.strip().split()) <= 5


def _is_numeric_declarative(text: str) -> bool:
    declarative = re.compile(
        r"\b(would be|will be|is|are|should be|estimate[sd]? (?:at|to be)?|"
        r"around|approximately|about|roughly|target[ed]?)\b",
        re.IGNORECASE,
    )
    return bool(declarative.search(text))


def _extract_monthly_queries(user_message: str) -> int | None:
    message_lower = user_message.lower()
    million_match = re.search(r"(\d[\d,]*)\s*(?:million|m)\s+(?:queries|requests)", message_lower)
    if million_match:
        return _parse_int(million_match.group(1)) * 1_000_000
    thousand_match = re.search(r"(\d[\d,]*)\s*(?:thousand|k)\s+(?:queries|requests)", message_lower)
    if thousand_match:
        return _parse_int(thousand_match.group(1)) * 1_000
    per_month_match = re.search(r"(\d[\d,]*)\s+(?:queries|requests)\s+(?:per\s+)?month", message_lower)
    if per_month_match:
        return _parse_int(per_month_match.group(1))
    monthly_match = re.search(r"(\d[\d,]*)\s+monthly\s+(?:queries|requests)", message_lower)
    if monthly_match:
        return _parse_int(monthly_match.group(1))
    return None


def _extract_input_tokens(user_message: str) -> int | None:
    message_lower = user_message.lower()
    patterns = [
        r"(\d[\d,]*)\s*input\s+tokens",
        r"input\s+tokens[:\s]+(\d[\d,]*)",
        r"(\d[\d,]*)\s+tokens?\s+per\s+query\s+in",
    ]
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            return _parse_int(match.group(1))
    range_match = re.search(
        r"(\d[\d,]*)\s*[-–]\s*(\d[\d,]*)\s+tokens?\s+per\s+(?:chat|query|request|message)",
        message_lower,
    )
    if range_match:
        return _parse_int(range_match.group(2))
    return None


def _extract_output_tokens(user_message: str) -> int | None:
    message_lower = user_message.lower()
    patterns = [
        r"(\d[\d,]*)\s*output\s+tokens",
        r"output\s+tokens[:\s]+(\d[\d,]*)",
        r"(\d[\d,]*)\s+tokens?\s+per\s+query\s+out",
        r"make\s+it\s+(\d[\d,]*)\s+tokens?",
        r"(?:set|change|update)\s+(?:it\s+)?to\s+(\d[\d,]*)\s+tokens?",
    ]
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            return _parse_int(match.group(1))
    combined_match = re.search(r"(\d[\d,]*)\s+input\s+and\s+(\d[\d,]*)\s+output\s+tokens", message_lower)
    if combined_match:
        return _parse_int(combined_match.group(2))
    return None


def _extract_latency_sla(user_message: str) -> int | None:
    message_lower = user_message.lower()
    if "real-time" in message_lower or "sub-second" in message_lower:
        return 500
    ms_match = re.search(r"(\d+)\s*ms(?:\s+latency|\s+sla)?", message_lower)
    if ms_match:
        return int(ms_match.group(1))
    second_match = re.search(r"(\d+(?:\.\d+)?)\s*second(?:s)?\s+(?:latency|sla|response)", message_lower)
    if second_match:
        return int(float(second_match.group(1)) * 1000)
    return None


def _extract_reasoning_complexity(user_message: str) -> str | None:
    msg = user_message.lower()
    if any(k in msg for k in ("complex reasoning", "multi-step", "math-heavy", "math heavy", "deep reasoning")):
        return "high"
    if any(k in msg for k in ("simple q&a", "simple qa", "faq", "basic q&a", "low reasoning")):
        return "low"
    if any(k in msg for k in ("multi-step reasoning", "reasoning workload", "analysis")):
        return "medium"
    return None


def _extract_context_complexity(user_message: str) -> str | None:
    msg = user_message.lower()
    if any(k in msg for k in ("long context", "long-context", "large documents", "rag")):
        return "long-context"
    if any(k in msg for k in ("short context", "short-context", "short prompts", "short chat")):
        return "short-context"
    if "medium context" in msg or "medium-context" in msg:
        return "medium-context"
    return None


def _extract_coding_tool_use(user_message: str) -> str | None:
    msg = user_message.lower()
    if any(k in msg for k in ("agent", "tool use", "tools", "codegen", "coding", "copilot")):
        return "high"
    if "no code" in msg or "no coding" in msg:
        return "low"
    if "coding/tool use" in msg:
        return "medium"
    return None


def _extract_hallucination_sensitivity(user_message: str) -> str | None:
    msg = user_message.lower()
    if any(k in msg for k in ("legal", "finance", "medical", "compliance", "high stakes", "high-stakes")):
        return "high"
    if any(k in msg for k in ("draft", "internal", "low risk", "low-risk")):
        return "low"
    if "hallucination" in msg:
        return "medium"
    return None


def _extract_explicit_fields(user_message: str) -> dict:
    combined = re.search(r"(\d[\d,]*)\s+input\s+and\s+(\d[\d,]*)\s+output\s+tokens", user_message.lower())
    input_tokens = _extract_input_tokens(user_message)
    if combined and input_tokens is None:
        input_tokens = _parse_int(combined.group(1))

    base = {
        "monthly_queries": _extract_monthly_queries(user_message),
        "input_tokens_per_query": input_tokens,
        "output_tokens_per_query": _extract_output_tokens(user_message),
        "latency_sla_ms": _extract_latency_sla(user_message),
        "reasoning_complexity": _extract_reasoning_complexity(user_message),
        "context_complexity": _extract_context_complexity(user_message),
        "coding_tool_use_intensity": _extract_coding_tool_use(user_message),
        "hallucination_sensitivity": _extract_hallucination_sensitivity(user_message),
    }

    overrides = _detect_explicit_overrides(user_message)
    for field, value in overrides.items():
        base[field] = value

    return base


def _cell_internal_value(cell) -> object | None:
    if cell is None:
        return None
    if isinstance(cell, dict) and "internal_value" in cell:
        return cell.get("internal_value")
    return cell


def _flatten_structured_assumptions(existing_assumptions: dict | None) -> tuple[dict, dict[str, str]]:
    flat: dict = {}
    sources: dict[str, str] = {}
    if not existing_assumptions:
        return flat, sources
    workload = existing_assumptions.get("workload") or {}
    complexity = existing_assumptions.get("complexity") or {}
    for field in ("monthly_queries", "input_tokens_per_query", "output_tokens_per_query", "latency_sla_ms"):
        cell = workload.get(field)
        value = _cell_internal_value(cell)
        if value is not None:
            flat[field] = value
            sources[field] = str(cell["source"]) if isinstance(cell, dict) and cell.get("source") in ("user_provided", "assumed") else "user_provided"
    for field in COMPLEXITY_FIELDS:
        cell = complexity.get(field)
        value = _cell_internal_value(cell)
        if value is not None:
            flat[field] = value
            sources[field] = str(cell["source"]) if isinstance(cell, dict) and cell.get("source") in ("user_provided", "assumed") else "user_provided"
    return flat, sources


def merge_assumptions(existing: dict | None, extracted: dict) -> dict:
    merged, _sources = merge_assumptions_with_provenance(existing or {}, {}, extracted)
    return merged


def merge_assumptions_with_provenance(
    existing_flat: dict,
    existing_sources: dict[str, str],
    extracted: dict,
    explicit_overrides: set[str] | None = None,
) -> tuple[dict, dict[str, str]]:
    explicit_overrides = explicit_overrides or set()
    merged = {field: None for field in REQUIRED_FIELDS}
    sources = {field: "missing" for field in REQUIRED_FIELDS}

    # Pass 1 — load existing values for fields NOT being overridden
    for field in REQUIRED_FIELDS:
        if field not in explicit_overrides and existing_flat.get(field) is not None:
            merged[field] = existing_flat[field]
            sources[field] = existing_sources.get(field, "user_provided")

    # Pass 2 — apply newly extracted values (fills nulls AND overrides)
    for field in REQUIRED_FIELDS:
        if extracted.get(field) is not None:
            merged[field] = extracted[field]
            sources[field] = "user_provided"

    return merged, sources


def _count_user_turns(conversation_history: list | None) -> int:
    count = 0
    if conversation_history:
        for turn in conversation_history:
            if isinstance(turn, dict) and turn.get("role") == "user":
                count += 1
    return count + 1


def _apply_conservative_assumptions(
    merged: dict, sources: dict[str, str], user_turn_count: int
) -> tuple[dict, dict[str, str], list[str]]:
    assumed_fields: list[str] = []
    if user_turn_count < 3:   # requires at least 3 user turns before falling back to defaults
        return merged, sources, assumed_fields
    for field in REQUIRED_FIELDS:
        if merged.get(field) is None and field in CONSERVATIVE_DEFAULTS:
            merged[field] = CONSERVATIVE_DEFAULTS[field]
            sources[field] = "assumed"
            assumed_fields.append(field)
    return merged, sources, assumed_fields


def _format_display_value(field: str, value) -> str | None:
    if value is None:
        return None
    if field == "monthly_queries":
        return f"{int(value):,}"
    if field in TOKEN_FIELDS:
        return f"~{int(value):,} tokens"
    if field == "latency_sla_ms":
        return f"{int(value)} ms"
    if field == "context_complexity":
        return str(value).replace("-", " ").title()
    if field in COMPLEXITY_FIELDS:
        return str(value).replace("_", " ").title()
    return str(value)


def _build_field_cell(field: str, value, source: str) -> dict:
    return {
        "source": source,
        "display_value": _format_display_value(field, value),
        "internal_value": value,
    }


def _build_provenance_structure(merged: dict, sources: dict[str, str]) -> dict:
    workload = {}
    for field in ("monthly_queries", "input_tokens_per_query", "output_tokens_per_query", "latency_sla_ms"):
        value = merged.get(field)
        source = sources.get(field, "missing" if value is None else "user_provided")
        workload[field] = _build_field_cell(field, value, source)
    complexity = {}
    for field in COMPLEXITY_FIELDS:
        value = merged.get(field)
        source = sources.get(field, "missing" if value is None else "user_provided")
        complexity[field] = _build_field_cell(field, value, source)
    return {"workload": workload, "complexity": complexity}


def _merge_operational_assumptions(existing_assumptions: dict | None, user_message: str) -> dict:
    operational = dict(DEFAULT_OPERATIONAL_ASSUMPTIONS)
    if existing_assumptions:
        existing_operational = existing_assumptions.get("operational") or {}
        operational.update({k: v for k, v in existing_operational.items() if v is not None})
    if "enterprise" in user_message.lower():
        operational["enterprise_api_discount_pct"] = 15
    return operational


def _build_missing_fields(merged: dict) -> list[str]:
    return [field for field in REQUIRED_FIELDS if merged.get(field) is None]


def _build_clarification_questions(missing_fields: list[str]) -> list[str]:
    questions = []
    seen = set()
    missing_tokens = TOKEN_FIELDS.intersection(missing_fields)
    if missing_tokens:
        questions.append(CONTEXT_SIZE_CLARIFICATION)
        seen.update(TOKEN_FIELDS)
    missing_complexity = COMPLEXITY_FIELDS.intersection(missing_fields)
    if missing_complexity:
        questions.append(FIELD_CLARIFICATIONS["reasoning_complexity"])
        seen.update(COMPLEXITY_FIELDS)
    for field in missing_fields:
        if field in seen:
            continue
        questions.append(FIELD_CLARIFICATIONS[field])
        seen.add(field)
    return questions


def _build_assistant_message(
    missing_fields: list[str],
    assumed_fields: list[str],
    clarification_questions: list[str],
    ready_to_simulate: bool,
    is_follow_up: bool,
    corrections_applied: dict[str, str] | None = None,
) -> str:
    correction_ack = ""
    if corrections_applied:
        lines = []
        for field, value in corrections_applied.items():
            label = HUMAN_FIELD_LABELS.get(field, field)
            display = _format_display_value(field, value) or str(value)
            lines.append(f"• {label} updated to {display}.")
        correction_ack = "Got it — I've applied your changes:\n\n" + "\n".join(lines) + "\n\n"

    if ready_to_simulate and assumed_fields:
        intro = "I have enough to run a simulation. I filled a few gaps with conservative estimates you can adjust in the table before simulating:"
        bullets = "\n".join(
            f"• {ASSUMPTION_MESSAGES.get(field, HUMAN_FIELD_LABELS.get(field, field))}"
            for field in assumed_fields
        )
        return correction_ack + f"{intro}\n\n{bullets}\n\nReview the assumptions and run the simulation when ready."

    if ready_to_simulate:
        return correction_ack + "I have enough detail to model this workload. Review the assumption table and click Run Simulation when you're ready."

    if clarification_questions:
        intro = "Thanks — I've captured what you shared so far. " if is_follow_up else "Got it — I'm building a workload profile from your description. "
        return correction_ack + f"{intro}To tighten the cost model, I still need one thing:\n\n{clarification_questions[0]}"

    if missing_fields:
        return correction_ack + "Tell me more about your AI workload in plain English — who uses it, how often, and what kind of answers it needs to produce."

    return correction_ack + "Describe your AI workload in plain English and I'll build structured assumptions for the cost simulator."


def _normalize_llm_field_extraction(raw: dict) -> dict:
    return {
        "monthly_queries": _coerce_optional_int(raw.get("monthly_queries")),
        "input_tokens_per_query": _coerce_optional_int(raw.get("input_tokens_per_query")),
        "output_tokens_per_query": _coerce_optional_int(raw.get("output_tokens_per_query")),
        "latency_sla_ms": _coerce_optional_int(raw.get("latency_sla_ms")),
        "reasoning_complexity": _coerce_reasoning_level(raw.get("reasoning_complexity")),
        "context_complexity": _coerce_context_level(raw.get("context_complexity")),
        "coding_tool_use_intensity": _coerce_reasoning_level(raw.get("coding_tool_use_intensity")),
        "hallucination_sensitivity": _coerce_reasoning_level(raw.get("hallucination_sensitivity")),
    }


def _coerce_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _parse_json_from_llm(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start: end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def _http_post_json(url: str, headers: dict, payload: dict, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {error_body}") from exc


def _build_planner_user_prompt(user_message: str, existing_assumptions: dict | None) -> str:
    overrides = _detect_explicit_overrides(user_message)
    override_note = ""
    if overrides:
        lines = [f"  - {k} → {v}" for k, v in overrides.items()]
        override_note = (
            "\n\nNOTE: The user is explicitly correcting the following fields. "
            "You MUST use these exact values in your JSON response:\n"
            + "\n".join(lines) + "\n"
        )

    if existing_assumptions:
        existing_json = json.dumps(existing_assumptions, indent=2)
        return (
            "Already collected assumptions (preserve unless the user clearly updates them):\n"
            f"{existing_json}\n"
            f"{override_note}\n"
            "Latest user message:\n"
            f"{user_message}\n\n"
            "Extract any new information from the latest message and return the JSON object."
        )

    return (
        f"{override_note}"
        "User workload description:\n"
        f"{user_message}\n\n"
        "Extract workload assumptions and return the JSON object."
    )


def _call_openai_compatible(system_prompt: str, user_prompt: str) -> str:
    # Legacy single-turn wrapper — kept for compatibility, delegates to history-aware version
    return _call_openai_compatible_with_history(system_prompt, user_prompt, None)


def _call_openai_compatible_legacy_impl(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when USE_REAL_LLM_PLANNER=true")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = _http_post_json(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        payload=payload,
    )
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI-compatible response missing choices")
    content = (choices[0].get("message") or {}).get("content")
    if not content:
        raise RuntimeError("OpenAI-compatible response missing message content")
    return str(content)


def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    conversation_history: list | None = None,
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    # Build multi-turn message array from conversation history.
    # This is the core memory fix — Haiku sees the full conversation,
    # not just the latest message, so it never asks for info already given.
    messages: list[dict] = []
    if conversation_history:
        for turn in conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Always append the current user prompt as the final turn
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0,
        "system": system_prompt,
        "messages": messages,
    }
    response = _http_post_json(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        payload=payload,
    )
    content_blocks = response.get("content") or []
    text_parts = [b.get("text", "") for b in content_blocks if isinstance(b, dict) and b.get("type") == "text"]
    content = "".join(text_parts).strip()
    if not content:
        raise RuntimeError("Anthropic response missing text content")
    return content


def _call_openai_compatible_with_history(
    system_prompt: str,
    user_prompt: str,
    conversation_history: list | None = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when USE_REAL_LLM_PLANNER=true")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        for turn in conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    response = _http_post_json(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        payload=payload,
    )
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI-compatible response missing choices")
    content = (choices[0].get("message") or {}).get("content")
    if not content:
        raise RuntimeError("OpenAI-compatible response missing message content")
    return str(content)


def _call_llm_provider(
    system_prompt: str,
    user_prompt: str,
    conversation_history: list | None = None,
) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt, conversation_history)
    if provider in {"openai", "openai_compatible"}:
        return _call_openai_compatible_with_history(system_prompt, user_prompt, conversation_history)
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")


def _extract_with_real_llm(
    user_message: str,
    existing_assumptions: dict | None,
    conversation_history: list | None = None,
) -> dict:
    user_prompt = _build_planner_user_prompt(user_message, existing_assumptions)
    raw_text = _call_llm_provider(SYSTEM_INSTRUCTION, user_prompt, conversation_history)
    raw_json = _parse_json_from_llm(raw_text)

    workload_summary = raw_json.get("workload_summary")
    if not isinstance(workload_summary, str) or not workload_summary.strip():
        workload_summary = None
    else:
        workload_summary = workload_summary.strip()

    return {
        "workload_summary": workload_summary,
        "field_values": _normalize_llm_field_extraction(raw_json),
        "clarification_questions": _coerce_string_list(raw_json.get("clarification_questions")),
    }


def _build_plan_from_merged(
    user_message: str,
    existing_assumptions: dict | None,
    merged: dict,
    field_sources: dict[str, str],
    conversation_history: list | None = None,
    workload_summary_override: str | None = None,
    clarification_override: list[str] | None = None,
    corrections_applied: dict[str, str] | None = None,
) -> dict:
    user_turn_count = _count_user_turns(conversation_history)
    merged, field_sources, assumed_fields = _apply_conservative_assumptions(merged, field_sources, user_turn_count)

    missing_fields = _build_missing_fields(merged)
    clarification_questions = _build_clarification_questions(missing_fields)

    if clarification_override and not assumed_fields:
        llm_questions = [q for q in clarification_override if q]
        if llm_questions:
            clarification_questions = llm_questions

    ready_to_simulate = len(missing_fields) == 0
    is_follow_up = existing_assumptions is not None

    assistant_message = workload_summary_override or _build_assistant_message(
        missing_fields, assumed_fields, clarification_questions,
        ready_to_simulate, is_follow_up, corrections_applied=corrections_applied,
    )

    operational = _merge_operational_assumptions(existing_assumptions, user_message)
    provenance = _build_provenance_structure(merged, field_sources)
    provenance["operational"] = operational

    return {
        "assistant_message": assistant_message,
        "workload_summary": assistant_message,
        **merged,
        "field_sources": field_sources,
        "assumed_fields": assumed_fields,
        "structured_provenance": provenance,
        "assumptions": operational,
        "missing_fields": missing_fields,
        "clarification_questions": clarification_questions,
        "ready_to_simulate": ready_to_simulate,
    }


def _parse_workload_with_mock(
    user_message: str,
    existing_assumptions: dict | None = None,
    conversation_history: list | None = None,
) -> dict:
    existing_flat, existing_sources = _flatten_structured_assumptions(existing_assumptions)
    overrides = _detect_explicit_overrides(user_message)
    extracted = _extract_explicit_fields(user_message)
    merged, field_sources = merge_assumptions_with_provenance(
        existing_flat, existing_sources, extracted,
        explicit_overrides=set(overrides.keys()),
    )
    return _build_plan_from_merged(
        user_message, existing_assumptions, merged, field_sources,
        conversation_history=conversation_history,
        corrections_applied=overrides if overrides else None,
    )


def _parse_workload_with_real_llm(
    user_message: str,
    existing_assumptions: dict | None = None,
    conversation_history: list | None = None,
) -> dict:
    existing_flat, existing_sources = _flatten_structured_assumptions(existing_assumptions)
    overrides = _detect_explicit_overrides(user_message)
    llm_result = _extract_with_real_llm(user_message, existing_assumptions, conversation_history)

    # Defensive patch — force overrides even if LLM ignored them
    for field, value in overrides.items():
        llm_result["field_values"][field] = value

    merged, field_sources = merge_assumptions_with_provenance(
        existing_flat, existing_sources, llm_result["field_values"],
        explicit_overrides=set(overrides.keys()),
    )
    return _build_plan_from_merged(
        user_message, existing_assumptions, merged, field_sources,
        conversation_history=conversation_history,
        workload_summary_override=llm_result.get("workload_summary"),
        clarification_override=llm_result.get("clarification_questions"),
        corrections_applied=overrides if overrides else None,
    )


def parse_workload_with_llm(
    user_message: str,
    existing_assumptions: dict | None = None,
    conversation_history: list | None = None,
) -> dict:
    if use_real_llm_planner():
        try:
            return _parse_workload_with_real_llm(
                user_message,
                existing_assumptions=existing_assumptions,
                conversation_history=conversation_history,
            )
        except Exception as exc:
            logger.warning("LLM planner failed; falling back to rule-based extraction: %s", exc)
            return _parse_workload_with_mock(
                user_message,
                existing_assumptions=existing_assumptions,
                conversation_history=conversation_history,
            )

    return _parse_workload_with_mock(
        user_message,
        existing_assumptions=existing_assumptions,
        conversation_history=conversation_history,
    )


def build_plan_response(plan: dict) -> dict:
    structured = plan.get("structured_provenance") or _build_provenance_structure(
        plan, plan.get("field_sources", {})
    )
    if "operational" not in structured:
        structured["operational"] = plan["assumptions"]

    return {
        "assistant_message": plan["assistant_message"],
        "workload_summary": plan.get("workload_summary", plan["assistant_message"]),
        "structured_assumptions": structured,
        "missing_fields": plan["missing_fields"],
        "clarification_questions": plan["clarification_questions"],
        "assumed_fields": plan.get("assumed_fields", []),
        "ready_to_simulate": plan["ready_to_simulate"],
    }