"""
AI Infra Cost Advisor — ADK Agent Pipeline
==========================================
4-agent pipeline built on Google ADK (Workflow API, ADK 2.3.0+).

Architecture:
  ParseJudgeWorkflow (Workflow with loop-back edges)
    ├── parsing_agent    (LlmAgent — extracts structured workload spec)
    ├── judge_agent      (LlmAgent — validates spec, emits verdict)
    └── verdict_router   (FunctionNode — routes: retry → loop back, pass/needs_user → exit)
  pricing_agent          (LlmAgent + MCP tools — fetches live AWS/GCP prices, reads snapshot for CoreWeave/Lambda)
  cost_engine_bridge     (FunctionNode — deterministic cost calculation using live prices)
  reasoning_agent        (LlmAgent — 1x/2x/5x scenarios, breakeven, confidence)

Full pipeline: parse_judge → pricing_agent → cost_engine_bridge → reasoning_agent.
When verdict is "needs_user", the API layer reads session.state["clarifying_question"]
and surfaces it to the UI without running pricing, cost engine, or reasoning.

MCP server: mcp_servers/gpu_pricing_server.py (stdio transport)
  Tools: get_aws_trainium2_price, get_gcp_tpu_v5e_price, get_snapshot_gpu_prices
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.events import Event, EventActions
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams
from mcp.client.stdio import StdioServerParameters
from google.adk.workflow import Workflow, Edge, START, DEFAULT_ROUTE, node

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

_MODEL = os.getenv("ADK_MODEL", "gemini-2.0-flash")


# ---------------------------------------------------------------------------
# 1. Parsing Agent
# ---------------------------------------------------------------------------

PARSING_SYSTEM_PROMPT = """\
You are a workload specification extractor for an AI infrastructure cost advisor.

FEASIBILITY CHECK — do this BEFORE extracting any fields:
Closed/proprietary models (Claude, GPT, Gemini, and similar API-only models) do
NOT release their model weights. This is an architectural fact, not a pricing
or licensing detail: there is no license, contract, or payment tier that lets
a customer download and run Claude, GPT, or Gemini on their own GPUs. The only
way to use these models is by calling the provider's managed API. Self-hosting
is only possible with open-weight models (e.g. Llama, Qwen, Mistral), where the
weights are actually published.

If the user's message is primarily asking whether they can run, host, deploy,
or license a closed/proprietary model on their OWN infrastructure (their own
GPUs, their own datacenter, "on-prem", etc.) — rather than describing a
workload to size — do NOT treat this as a workload to extract. Instead, output:
{
  "original_description": "the user's exact words, copied verbatim",
  "monthly_queries": 0,
  "input_tokens_per_query": 0,
  "output_tokens_per_query": 0,
  "latency_sla": "interactive",
  "reasoning_complexity": "medium",
  "context_complexity": "low",
  "hallucination_sensitivity": "medium",
  "advisory_notes": "",
  "field_confidence": {},
  "infeasible_request": true,
  "infeasible_reason": "one or two plain-English sentences explaining that the specific named model cannot be self-hosted because the provider does not release its weights, and that self-hosting requires an open-weight model instead (give 1-2 examples: Llama, Qwen)"
}
Only set infeasible_request=true for this specific case (asking to self-host a
named closed model). A normal request to self-host "an open-source model" or to
compare open-weight options is NOT infeasible — proceed with normal extraction.

If the request is feasible, proceed with normal extraction below.

Your job is to read the user's plain-English description of an AI workload and
extract a structured JSON object with these fields:

TIER 1 — core volume & shape (extract from text or estimate from context):
  - monthly_queries (int): total LLM API calls PER MONTH
  - input_tokens_per_query (int): tokens in each request prompt
  - output_tokens_per_query (int): tokens in each response

TIER 2 — workload character (extract if stated, else infer from use-case):
  - latency_sla (str): "real-time" | "interactive" | "batch"
  - reasoning_complexity (str): "low" | "medium" | "high"
  - context_complexity (str): "low" | "medium" | "high"
  - hallucination_sensitivity (str): "low" | "medium" | "high"
  - traffic_pattern (str): "smooth" | "predictable_peaks" | "spiky"
      How evenly traffic arrives across the day/week. This drives GPU capacity
      planning for self-hosted options — spiky workloads require provisioning
      for peak load, not average load, which significantly increases GPU cost.
      Inference rules (use these when the user hasn't stated it explicitly):
        "smooth"            → 24/7 steady load; customer support bots, API
                              services with many users, background processing.
                              GPU capacity multiplier: 1.0x (no headroom needed).
        "predictable_peaks" → Traffic spikes at known times (business hours,
                              daily batch jobs, morning rush). Healthcare,
                              B2B SaaS, scheduling systems, daily pipelines.
                              GPU capacity multiplier: 1.15x headroom.
        "spiky"             → Unpredictable or extreme bursts; HFT signals,
                              fraud detection on payment spikes, viral consumer
                              apps, event-driven systems.
                              GPU capacity multiplier: 1.35x headroom.
      If genuinely unclear after reading the description, default to
      "predictable_peaks" and note it in advisory_notes.

TIME UNIT NORMALISATION — always convert to monthly before writing monthly_queries:
  - "per second" → × 60 × 60 × 24 × 30  (e.g. 10/s = 25,920,000/month)
  - "per minute" → × 60 × 24 × 30
  - "per hour"   → × 24 × 30
  - "per day"    → × 30
  - "per week"   → × 4.3
  - "per year"   → ÷ 12
  Always write the MONTHLY total in monthly_queries.

TEAM-SIZE HEURISTICS (use when no explicit volume is given):
  - Customer support agent handling 30–50 tickets/day → ~1,200–2,000 queries/month each
  - Developer running code reviews → ~8 PRs/week × 4.3 weeks = ~34/month each
  - Knowledge-base / FAQ lookup → 3–10 queries per employee per week

RESOLVING QUALITATIVE ANSWERS — if this message is a reply to a clarifying
question (look for "Additional context:" in the text) and the user's answer
is descriptive rather than numeric, convert it to a concrete number yourself
using these anchors. Do NOT ask the same question again — a qualitative
answer IS enough to commit to an estimate.
  Output length anchors:
    - "quick" / "one-liner" / "short answer"        -> ~30-50 tokens
    - "in between" / "mix of short and long" / "varies" -> ~120-180 tokens
    - "detailed" / "long explanation" / "thorough"  -> ~250-400 tokens
  Input length anchors:
    - "short message" / "quick question"            -> ~30-60 tokens
    - "a few sentences" / "some context"             -> ~150-300 tokens
    - "long document" / "paste in a lot of context"  -> ~1,000+ tokens
  When you resolve a qualitative answer this way, still record it in
  "field_confidence" — the number is a reasonable midpoint estimate, not a
  user-stated figure — but do NOT generate a new clarifying_question for it.
  One round of clarification is enough; after that, commit and move forward.

ADVISORY MODE: If the user gives enough context (team size, use-case type, cadence)
to make a reasonable volume estimate, DO estimate it and record your reasoning in
"advisory_notes". Only leave monthly_queries = 0 if the description is a bare label
with NO context whatsoever (e.g. "AI writing assistant" with no other details).

CONFIDENCE TRACKING — required for every Tier 1 field:
For EACH of monthly_queries, input_tokens_per_query, and output_tokens_per_query,
you must record whether the value was STATED by the user or ESTIMATED by you.
A value is STATED if the user gave a number or a clear quantity you converted
(e.g. "8 million queries a month" -> stated; "30-40 tickets a day for 20 agents"
-> stated, since it's directly computable). A value is ESTIMATED if you picked
a number with no real basis in the text — e.g. the user never mentioned message
length, conversation style, or output format, and you defaulted to a generic
number anyway.

Add a "field_confidence" object listing ONLY the fields you estimated (omit
fields that were stated). For each estimated field, give a one-sentence reason
the estimate is weak. Example:
  "field_confidence": {
    "input_tokens_per_query": "User never described message length or content; defaulted to a generic chat estimate with no basis in the description.",
    "output_tokens_per_query": "Same — no information given about response length or format."
  }
If every Tier 1 field was stated or directly computable, output "field_confidence": {}.

Output ONLY a JSON object. No prose, no markdown fences. Example:
{
  "original_description": "the user's exact words, copied verbatim",
  "monthly_queries": 50000,
  "input_tokens_per_query": 800,
  "output_tokens_per_query": 300,
  "latency_sla": "interactive",
  "reasoning_complexity": "medium",
  "context_complexity": "low",
  "hallucination_sensitivity": "medium",
  "traffic_pattern": "predictable_peaks",
  "advisory_notes": "",
  "field_confidence": {}
}

Always include "original_description" as the first field — copy the user's message verbatim.
"""

parsing_agent = LlmAgent(
    name="ParsingAgent",
    model=_MODEL,
    instruction=PARSING_SYSTEM_PROMPT,
    output_key="workload_spec",
)


# ---------------------------------------------------------------------------
# 2. Judge Agent (Evals / LLM-as-Judge rubric item)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a workload specification validator for an AI infrastructure cost advisor.

The parsed workload spec (produced by the Parsing Agent) is:
{workload_spec}

The spec includes an "original_description" field containing the user's exact words.
Use that field when you need to compare the spec against what the user actually said.

Run these cross-checks — do not invent new checks:

CHECK -1 — Infeasible request: Does the spec have "infeasible_request": true?
  If so, this is NOT a workload to validate — the Parsing Agent already
  determined the user is asking to self-host a closed model, which is not
  possible. Immediately set verdict = "infeasible" and set clarifying_question
  to the spec's "infeasible_reason" text, verbatim or lightly rephrased to read
  naturally as a direct answer to the user (not a request for more info — this
  is an explanation, not a question). Do NOT run any other checks below.

CHECK 0 — Weak estimates: Does the spec include a non-empty "field_confidence"
  object (fields the Parsing Agent flagged as estimated with no real basis in
  the description, e.g. input/output token counts guessed with zero context)?
  If field_confidence is non-empty → verdict = "needs_user". Set
  clarifying_question to ask about the SPECIFIC field(s) listed, in plain
  English (e.g. "About how long are typical responses — a quick one-line
  answer, or more detailed explanations?"). Do not ask about fields the user
  already stated or that were directly computable.

  ONE-ROUND LIMIT: check original_description for the literal text
  "Additional context:". If present, this is already a reply to a prior
  clarifying question — do NOT trigger needs_user again for the SAME
  field(s), even if field_confidence still lists them as estimated (a
  reasonable midpoint estimate from a qualitative answer is good enough on
  round 2). Only re-ask if the user's reply gave literally no signal at all
  for that field (e.g. they ignored the question entirely). Bias toward
  "pass" on round 2.

CHECK 1 — Completeness: Are all three TIER 1 fields present and numeric?
  Required: monthly_queries (int > 0), input_tokens_per_query (int > 0),
  output_tokens_per_query (int > 0).
  If ANY are missing or zero → verdict = "needs_user".
  ALSO: if original_description is fewer than 8 words AND monthly_queries
  was estimated (not stated), treat this as "needs_user" — a bare label
  like "AI writing assistant" does not give enough context for a reliable estimate.

CHECK 2 — Output size plausibility: Does output_tokens_per_query fit the
  stated output type?
  - "quick answer / troubleshooting" → should be 50–400 tokens, NOT 1000+
  - "detailed report / summary" → 400–1200 tokens is fine
  - "short chat reply" → 50–200 tokens
  Only flag if output_tokens is wildly wrong (>3x off), not for minor differences.

CHECK 3 — Internal contradiction: Is any field directly contradicted by the
  plain-English description?
  Example: user said "real-time autocomplete" but latency_sla = "batch" → flag.
  Example: user said "10-person team" but monthly_queries = 5,000,000 → flag.
  Only flag clear contradictions, not mere imprecision.

DEFAULT BEHAVIOR: If no checks fail, verdict = "pass". Most well-described
workloads should pass. Bias strongly toward "pass" when the spec is complete
and internally consistent.

Output ONLY a JSON object — no prose, no markdown:
{
  "verdict": "pass" | "retry" | "needs_user" | "infeasible",
  "issues": [],
  "clarifying_question": ""
}

Verdict rules:
- "pass"       → spec complete and consistent. issues=[], clarifying_question="".
- "retry"      → parser clearly missed a field on a complex/dense description
                 (re-parsing the same text might help). Do NOT use for missing
                 user info.
- "needs_user" → user's description is genuinely ambiguous or contradictory in
                 a way only they can resolve. Set clarifying_question to ONE
                 specific plain-English question. issues = list of what's wrong.
- "infeasible" → spec has infeasible_request=true (user asked to self-host a
                 closed model). Set clarifying_question to the explanation
                 from CHECK -1. This is terminal — there is nothing for the
                 user to answer, just an explanation to deliver.
"""

judge_agent = LlmAgent(
    name="JudgeAgent",
    model=_MODEL,
    instruction=JUDGE_SYSTEM_PROMPT,
    input_schema=None,
    output_key="judge_verdict_raw",
)


# ---------------------------------------------------------------------------
# 3. Verdict Router (FunctionNode — controls loop-back vs exit)
# ---------------------------------------------------------------------------

@node
def verdict_router(ctx: Any) -> Event:
    """
    Reads the judge's verdict from session state and routes accordingly:
      - "retry"      → loop back to ParsingAgent (Edge route="retry")
      - "pass"       → exit loop, continue to PricingAgent (Edge route="pass")
      - "needs_user" → exit loop WITHOUT running pricing, carry clarifying
                       question in state (Edge route="needs_user")
    Also tracks iteration count and forces exit after MAX_ITERATIONS.
    """
    import json

    MAX_ITERATIONS = 3

    # Increment iteration counter
    iteration = ctx.state.get("parse_judge_iterations", 0) + 1
    ctx.state["parse_judge_iterations"] = iteration

    # Parse the judge's JSON output — strip markdown fences if present
    raw = ctx.state.get("judge_verdict_raw", "{}")
    if isinstance(raw, dict):
        verdict_obj = raw
    else:
        text = str(raw).strip()
        # Find the first { and last } to extract bare JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
        try:
            verdict_obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            verdict_obj = {}

    verdict = verdict_obj.get("verdict", "needs_user")
    issues = verdict_obj.get("issues", [])
    clarifying_question = verdict_obj.get("clarifying_question", "")

    # Circuit breaker: force exit after MAX_ITERATIONS
    if iteration >= MAX_ITERATIONS and verdict == "retry":
        verdict = "needs_user"
        clarifying_question = (
            "I was unable to fully resolve the workload details after "
            f"{MAX_ITERATIONS} attempts. Could you clarify: "
            + (clarifying_question or "the monthly query volume and output length for your use case?")
        )

    # Persist structured verdict and any clarifying question to state
    ctx.state["judge_verdict"] = verdict
    ctx.state["judge_issues"] = issues
    ctx.state["clarifying_question"] = clarifying_question

    return Event(actions=EventActions(route=verdict))


# ---------------------------------------------------------------------------
# 4. Parse-Judge Loop (Workflow with conditional cycle)
# ---------------------------------------------------------------------------
# Graph: START → parsing → judge → verdict_router
#                                        │ route="retry"  → back to parsing
#                                        │ route="pass"   → exits (no outgoing edge → terminal)
#                                        │ route="needs_user" → exits
#                                        └ route="infeasible" → exits (no question to answer,
#                                          just an explanation — see CHECK -1 in JUDGE_SYSTEM_PROMPT)

@node
def loop_exit(ctx: Any) -> None:
    """Terminal sink — reached when verdict is 'pass', 'needs_user', or 'infeasible'. No-op."""
    pass


parse_judge_workflow = Workflow(
    name="ParseJudgeLoop",
    edges=[
        Edge(from_node=START, to_node=parsing_agent),
        Edge(from_node=parsing_agent, to_node=judge_agent),
        Edge(from_node=judge_agent, to_node=verdict_router),
        Edge(from_node=verdict_router, to_node=parsing_agent, route="retry"),
        Edge(from_node=verdict_router, to_node=loop_exit, route=DEFAULT_ROUTE),
    ],
)


# ---------------------------------------------------------------------------
# 5. Pricing Agent (LlmAgent + MCP tools — live AWS/GCP + snapshot fallback)
# ---------------------------------------------------------------------------

_MCP_SERVER_PATH = str(
    Path(__file__).resolve().parent.parent / "mcp_servers" / "gpu_pricing_server.py"
)

PRICING_SYSTEM_PROMPT = """\
You are a GPU pricing data agent for an AI infrastructure cost advisor.
Your sole job is to gather current hourly GPU prices by calling the three tools
available to you, then output a single JSON object with those prices.

Steps — call ALL THREE tools before outputting:
1. Call get_aws_trainium2_price()   → live Trainium2 chip price from AWS Pricing API
2. Call get_gcp_tpu_v5e_price()     → live TPU v5e chip price from GCP Billing Catalog
3. Call get_snapshot_gpu_prices()   → CoreWeave H100 + Lambda Labs H100 from snapshot

After all three tool calls complete, output ONLY a valid JSON object with NO prose,
NO markdown fences, and NO extra keys. Use the exact hourly_cost_per_gpu values
returned by the tools — do not round, estimate, or substitute different numbers.

Required output format:
{
  "aws_trainium2":   {"hourly_cost_per_gpu": <float>, "source": "<string>"},
  "gcp_tpu_v5":      {"hourly_cost_per_gpu": <float>, "source": "<string>"},
  "coreweave_h100":  {"hourly_cost_per_gpu": <float>, "source": "<string>"},
  "lambda_labs_h100": {"hourly_cost_per_gpu": <float>, "source": "<string>"}
}

Map the tool results to these exact top-level keys:
  get_aws_trainium2_price()   → aws_trainium2
  get_gcp_tpu_v5e_price()     → gcp_tpu_v5
  get_snapshot_gpu_prices() "coreweave_h100"   → coreweave_h100
  get_snapshot_gpu_prices() "lambda_labs_h100" → lambda_labs_h100
"""

pricing_agent = LlmAgent(
    name="PricingAgent",
    model=_MODEL,
    instruction=PRICING_SYSTEM_PROMPT,
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,
                    args=[_MCP_SERVER_PATH],
                )
            )
        )
    ],
    output_key="live_gpu_prices",
)


# ---------------------------------------------------------------------------
# 7. Cost Engine Bridge (FunctionNode — deterministic, uses live GPU prices)
# ---------------------------------------------------------------------------
# Runs backend/cost_engine.py inside the ADK pipeline so ReasoningAgent
# receives real numbers via {cost_scenarios} state interpolation.
# Reads live_gpu_prices from PricingAgent output and applies as overrides.

@node
def cost_engine_bridge(ctx: Any) -> None:
    """
    Reads workload_spec and live_gpu_prices from session state, runs the
    deterministic cost engine with live price overrides, and stores results
    in 'cost_scenarios'. ReasoningAgent reads this via {cost_scenarios}.
    """
    import json as _json

    raw = ctx.state.get("workload_spec", {})
    if isinstance(raw, str):
        try:
            spec = _json.loads(raw)
        except Exception:
            spec = {}
    else:
        spec = raw or {}

    monthly_queries = int(spec.get("monthly_queries") or 0)
    input_tokens = int(spec.get("input_tokens_per_query") or 0)
    output_tokens = int(spec.get("output_tokens_per_query") or 0)

    if not (monthly_queries and input_tokens and output_tokens):
        ctx.state["cost_scenarios"] = _json.dumps([])
        ctx.state["pricing_results"] = _json.dumps({"error": "missing volume fields"})
        return

    # Parse live GPU prices written by PricingAgent (if present)
    gpu_price_overrides: dict | None = None
    raw_live = ctx.state.get("live_gpu_prices")
    if raw_live:
        if isinstance(raw_live, str):
            text = raw_live.strip()
            # Strip markdown code fences if the LLM wrapped its output
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                text = text[start : end + 1]
            try:
                live_prices = _json.loads(text)
            except Exception:
                live_prices = {}
        else:
            live_prices = raw_live or {}

        gpu_price_overrides = {}
        for provider_key in ("aws_trainium2", "gcp_tpu_v5", "coreweave_h100", "lambda_labs_h100"):
            entry = live_prices.get(provider_key, {})
            if isinstance(entry, dict) and entry.get("hourly_cost_per_gpu"):
                gpu_price_overrides[provider_key] = {
                    "hourly_cost_per_gpu": float(entry["hourly_cost_per_gpu"]),
                    "source": entry.get("source", "live"),
                }

    try:
        from backend.cost_engine import calculate_scenarios
        scenarios = calculate_scenarios(
            monthly_queries=monthly_queries,
            input_tokens_per_query=input_tokens,
            output_tokens_per_query=output_tokens,
            gpu_price_overrides=gpu_price_overrides or None,
        )
        ctx.state["cost_scenarios"] = _json.dumps(scenarios, default=str)
        # Flat summary for downstream inspection / pricing_results field
        api_entries, gpu_entries = [], []
        for s in scenarios:
            if s.get("scenario") in ("1x", "current"):
                api_entries = s.get("api_models", [])
                gpu_entries = s.get("gpu_providers", [])
                break
        ctx.state["pricing_results"] = _json.dumps(
            {
                "api_costs": api_entries,
                "gpu_costs": gpu_entries,
                "gpu_price_sources": {
                    k: v.get("source") for k, v in (gpu_price_overrides or {}).items()
                },
            },
            default=str,
        )
    except Exception as exc:
        ctx.state["cost_scenarios"] = _json.dumps([])
        ctx.state["pricing_results"] = _json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# 8. Reasoning & Recommendation Agent
# ---------------------------------------------------------------------------

REASONING_SYSTEM_PROMPT = """\
You are a cost reasoning and recommendation agent for an AI infrastructure
cost advisor. You receive REAL pricing data — do NOT invent numbers.

Workload specification:
{workload_spec}

TRAFFIC PATTERN NOTE: The workload spec includes a "traffic_pattern" field
("smooth" | "predictable_peaks" | "spiky"). This has already been used to
apply a GPU capacity multiplier to the cost scenarios:
  - smooth            → 1.0x  (flat load, GPU costs reflect average throughput)
  - predictable_peaks → 1.15x (15% headroom for known peak windows)
  - spiky             → 1.35x (35% headroom for unpredictable burst traffic)
If the pattern is "spiky" or "predictable_peaks", briefly mention in your
rationale that GPU costs include peak-load headroom — this is why self-hosted
GPU costs may appear higher than a naive calculation would suggest, and it is
an HONEST reflection of real deployment cost, not an overestimate.
If the pattern is "smooth", you can note that GPU costs assume steady load
with no peak headroom needed.

Cost scenarios (deterministic, from pricing engine):
{cost_scenarios}

The cost_scenarios is a JSON list with entries for "current" (1x), "growth_2x", and "growth_5x" scale.
Each entry has:
  - scenario: "current" | "growth_2x" | "growth_5x"
  - monthly_queries: int
  - cheapest_api_model: { display_name, monthly_cost, provider, model_key }
  - cheapest_gpu_provider: { display_name, monthly_cost, estimated_gpu_count }
    NOTE: cheapest_gpu_provider runs a closed/proprietary model on rented GPU — rarely the cheapest self-host option.
  - cheapest_open_weight_option: { display_name, model_display_name, provider_display_name,
      quality_tier, comparable_api, monthly_cost, estimated_gpu_count,
      toolchain_friction, toolchain_friction_note }
    Open-weight models (Llama, Qwen, etc.) have ZERO per-token cost — you only pay GPU hours.
    This is almost always significantly cheaper than managed API at meaningful scale.
  - open_weight_options: [ all (model x provider) combos sorted by monthly_cost, each with
      toolchain_friction ("none" | "moderate" | "high") and toolchain_friction_note ]
  - all_api_models: [ all managed API options sorted by monthly_cost ]

TOOLCHAIN_FRICTION FIELD — what it means:
  "none"     — standard NVIDIA CUDA (CoreWeave, Lambda Labs). Any open-weight model runs
               out of the box with vLLM, TGI, or similar. Zero extra engineering.
  "moderate" — requires porting to AWS Neuron SDK (Trainium2) or JAX/XLA (GCP TPU v5e).
               Official or strong community support exists for this model family.
               Expect 1–4 weeks of engineering work before production inference is viable.
  "high"     — requires non-trivial porting with limited model-specific support.
               Expect 4–12 weeks of engineering effort. Significant ongoing maintenance.

TOOLCHAIN FRICTION SELECTION RULES — apply when choosing the open-weight recommendation:
  1. Never recommend a "high" friction option as the primary recommendation if a "none"
     or "moderate" friction option exists within 20% of its monthly cost.
     (e.g. if Qwen3 32B on GCP TPU v5e costs $800/mo but Qwen3 32B on CoreWeave H100
     costs $950/mo — that's only 19% more — recommend CoreWeave despite its higher cost.)
  2. A "moderate" friction option may be recommended if it is meaningfully cheaper (>20%)
     than the best "none" friction option, but you must explicitly call out the porting
     effort and its engineering cost in recommendation_rationale.
  3. Always surface the friction level and its practical implication in
     recommendation_rationale when the recommended option has friction != "none".
  4. If the cheapest_open_weight_option has "high" friction and a "none"/"moderate"
     option exists within 20%, override cheapest_open_weight_option in your output
     with the friction-adjusted best option instead.

THREE-WAY COMPARISON you must make:
  A) Managed API          — pay per token, zero ops, instant start
  B) Self-hosted closed   — rent GPU + run proprietary model (rarely wins on cost)
  C) Self-hosted open-weight — rent GPU + Llama/Qwen/etc., no per-token cost (often cheapest at scale)

LATENCY SLA RULES — apply before making a recommendation:

  "real-time" latency_sla:
    Managed API round-trip (network + inference) is typically 300ms–3s at p50,
    and 1s–8s at p99 under load. This disqualifies managed API for workloads
    with hard sub-500ms SLA requirements (e.g. autocomplete, fraud screening,
    real-time audio/video, gaming).

    First decide whether this is a HARD requirement or a SOFT preference, by
    checking original_description for explicit language:
      - HARD signals (explicit): a specific millisecond/second number (e.g.
        "under 50ms", "sub-second"), or words like "required", "must",
        "no exceptions", "critical", "hard requirement".
      - HARD signals (domain-implied): even with NO explicit number or
        mandatory wording, treat as HARD if the use case is one where a slow
        response is itself a failure of the product's core function — e.g.
        fraud/transaction screening that blocks a live payment, autocomplete,
        live trading/bidding signals, real-time audio/video, gaming, or
        anything explicitly screening/blocking/gating a transaction or action
        as it happens. In these domains "real-time" is not decoration — the
        product doesn't work at all if the check arrives after the action it
        was supposed to gate. Do not require an explicit keyword for these
        cases; the use case itself is the signal.
      - SOFT signal: "real-time" used loosely with no explicit number, no
        mandatory language, AND no domain-implied urgency (e.g. "we'd like it
        to feel real-time" for a chat assistant, dashboard, or similar
        use case where a 1-2s delay would be a UX annoyance, not a failure).

    HARD requirement → set latency_flag = "api_latency_risk_hard". Managed
      API is NOT a viable recommendation regardless of cost — a disqualified
      option does not become acceptable just because it is cheaper. Recommend
      "Open-Weight GPU" (or "Hybrid" only if self-hosting cost is so extreme,
      e.g. 50x+ the API cost, that it may not be financially viable — in that
      case recommend "Hybrid" and say plainly in rationale that the hard SLA
      makes API non-viable, and self-hosting cost needs further scoping, but
      NEVER recommend "API" outright for a hard requirement it cannot meet).
      confidence_score should reflect SLA certainty, not cost attractiveness —
      do not lower confidence just because self-hosting is more expensive.

    SOFT preference → set latency_flag = "api_latency_risk_soft". Override
      recommendation to "Open-Weight GPU" UNLESS cost of self-hosting is more
      than 10x the API cost (in which case recommend "API" with the flag and
      note the latency tradeoff explicitly in rationale — this is a genuine
      cost/latency tradeoff the user can reasonably choose to accept).

    → Self-hosted inference in the same datacenter delivers 30–200ms p99.

  "interactive" latency_sla:
    Users tolerate 1–5s for conversational responses. Managed API is fine.
    → No latency flag. Cost drives the recommendation.

  "batch" latency_sla:
    Throughput matters, not per-request latency. Managed API is fine.
    → No latency flag. Cost drives the recommendation.

Your job:
1. Summarise the three scenarios using EXACT numbers from cost_scenarios.
   For each scenario report the cheapest option in each category (A, B, C).
2. Identify the breakeven point: at what monthly query volume does
   cheapest_open_weight_option become cheaper than cheapest_api_model?
   Interpolate linearly between the current and growth_5x data points.
3. Make a recommendation (API | Open-Weight GPU | Hybrid) applying:
   a. LATENCY GATE first (rules above) — latency can override the cost winner.
   b. Cost crossover vs current query volume.
   c. quality_tier of cheapest open-weight option vs workload reasoning_complexity:
      if open-weight is "budget" but reasoning_complexity is "high", flag quality gap.
4. Provide a confidence score 0.0-1.0. Lower confidence when latency_sla is
   "real-time" and the cost gap is large (forced self-hosting despite cost penalty).

Output ONLY a JSON object — no prose, no markdown fences:
{
  "scenarios": {
    "current": {
      "api_winner": "...", "api_monthly_cost": 0.0,
      "open_weight_winner": "...", "open_weight_monthly_cost": 0.0, "open_weight_quality_tier": "...",
      "open_weight_toolchain_friction": "none" | "moderate" | "high",
      "gpu_winner": "...", "gpu_monthly_cost": 0.0
    },
    "growth_2x": {
      "api_winner": "...", "api_monthly_cost": 0.0,
      "open_weight_winner": "...", "open_weight_monthly_cost": 0.0, "open_weight_quality_tier": "...",
      "open_weight_toolchain_friction": "none" | "moderate" | "high",
      "gpu_winner": "...", "gpu_monthly_cost": 0.0
    },
    "growth_5x": {
      "api_winner": "...", "api_monthly_cost": 0.0,
      "open_weight_winner": "...", "open_weight_monthly_cost": 0.0, "open_weight_quality_tier": "...",
      "open_weight_toolchain_friction": "none" | "moderate" | "high",
      "gpu_winner": "...", "gpu_monthly_cost": 0.0
    }
  },
  "breakeven_monthly_queries": null,
  "recommendation": "API" | "Open-Weight GPU" | "Hybrid",
  "recommendation_rationale": "2-3 sentence explanation citing actual $ numbers, quality tier, latency implications, and toolchain friction when relevant",
  "toolchain_friction": "none" | "moderate" | "high",
  "toolchain_friction_note": "" | "one sentence — only populated when friction is moderate or high, explaining the porting effort required",
  "latency_flag": "none" | "api_latency_risk_hard" | "api_latency_risk_soft",
  "latency_note": "" | "one sentence explaining the latency concern and what self-hosting achieves",
  "quality_gap_warning": "",
  "confidence_score": 0.85,
  "confidence_explanation": "one sentence"
}
"""

reasoning_agent = LlmAgent(
    name="ReasoningAgent",
    model=_MODEL,
    instruction=REASONING_SYSTEM_PROMPT,
    output_key="final_recommendation",
)


# ---------------------------------------------------------------------------
# 9. Full Pipeline (parse_judge → pricing_agent → cost_engine_bridge → reasoning)
# ---------------------------------------------------------------------------
# IMPORTANT: The API layer (main.py) MUST check session.state["judge_verdict"]
# after parse_judge_workflow completes. If verdict == "needs_user", it should
# return the clarifying_question to the UI and NOT run pricing or reasoning.

full_pipeline = Workflow(
    name="FullPipeline",
    edges=[
        Edge(from_node=START, to_node=parse_judge_workflow),
        Edge(from_node=parse_judge_workflow, to_node=pricing_agent),
        Edge(from_node=pricing_agent, to_node=cost_engine_bridge),
        Edge(from_node=cost_engine_bridge, to_node=reasoning_agent),
    ],
)


# ---------------------------------------------------------------------------
# 10. Cost+Reasoning Pipeline (skips parsing — used when spec is pre-validated)
# ---------------------------------------------------------------------------
# Used by /adk/recalculate: caller writes a validated workload_spec into session
# state before running, so we go straight to live pricing + cost engine.

cost_reasoning_pipeline = Workflow(
    name="CostReasoningPipeline",
    edges=[
        Edge(from_node=START, to_node=pricing_agent),
        Edge(from_node=pricing_agent, to_node=cost_engine_bridge),
        Edge(from_node=cost_engine_bridge, to_node=reasoning_agent),
    ],
)
