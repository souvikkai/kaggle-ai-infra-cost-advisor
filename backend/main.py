from typing import Any, Literal, Optional
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from backend.workload_parser import parse_workload
from backend.cost_engine import calculate_scenarios, DATA_PATH
from backend.recommendation_engine import generate_recommendation
from backend.llm_planner import build_plan_response, parse_workload_with_llm

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── Pricing refresh state ─────────────────────────────────────────────────────
_last_refresh_summary: dict[str, Any] | None = None

# Architecture:
#
#  ADK pipeline path (active — all frontend traffic routes here):
#    POST /adk/simulate    → full_pipeline (ParseJudgeLoop → PricingAgent →
#                            cost_engine_bridge → ReasoningAgent)
#                            Returns verdict="needs_user" (clarifying question) or
#                            verdict="pass" (workload_spec + cost_scenarios + final_recommendation)
#    POST /adk/recalculate → cost_engine_bridge + ReasoningAgent on a pre-validated spec
#                            (used when user edits assumptions inline)
#    POST /adk/plan        → ParseJudgeLoop only (not called by frontend currently)
#
#  Legacy path (kept for reference, not used by frontend):
#    POST /plan      → llm_planner.py → structured assumptions (multi-turn chat)
#    POST /simulate  → cost_engine.py → scenario numbers
#
# The ADK path requires GOOGLE_API_KEY (Gemini) in the environment.

app = FastAPI(title="AI Infra Cost Advisor")


def _run_pricing_refresh() -> None:
    """Background job: refresh pricing snapshot. Runs bi-weekly via APScheduler."""
    global _last_refresh_summary
    try:
        from backend.pricing_refresh import refresh_pricing
        logger.info("Bi-weekly pricing refresh starting…")
        _last_refresh_summary = refresh_pricing()
        logger.info("Pricing refresh complete: %s", _last_refresh_summary)
    except Exception as exc:
        logger.error("Pricing refresh failed: %s", exc)


@app.on_event("startup")
async def start_scheduler() -> None:
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger

        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            _run_pricing_refresh,
            trigger=IntervalTrigger(weeks=2),
            id="pricing_refresh",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        scheduler.start()
        app.state.scheduler = scheduler
        logger.info("Pricing refresh scheduler started — runs every 2 weeks")
    except ImportError:
        logger.warning(
            "apscheduler not installed — bi-weekly pricing refresh disabled. "
            "Run: pip install apscheduler"
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlanRequest(BaseModel):
    message: str
    existing_assumptions: Optional[dict[str, Any]] = None
    conversation_history: Optional[list[dict[str, Any]]] = None


class WorkloadRequest(BaseModel):
    description: str
    gpu_utilization_pct: float = Field(default=70, ge=30, le=100)
    enterprise_api_discount_pct: float = Field(default=0, ge=0, le=40)
    burstiness_factor: Literal["low", "medium", "high"] = "medium"
    failover_reserve_pct: float = Field(default=15, ge=0, le=50)


@app.post("/plan")
def plan_workload(request: PlanRequest):
    """
    Agentic planning layer: convert chat-style workload intake into structured
    assumptions. Stub implementation — no external LLM API yet.
    """
    plan = parse_workload_with_llm(
        request.message,
        existing_assumptions=request.existing_assumptions,
        conversation_history=request.conversation_history,
    )
    return build_plan_response(plan)


@app.post("/simulate")
def simulate_workload(request: WorkloadRequest):
    parsed = parse_workload(request.description)

    scenarios = calculate_scenarios(
        monthly_queries=parsed["monthly_queries"],
        input_tokens_per_query=parsed["input_tokens_per_query"],
        output_tokens_per_query=parsed["output_tokens_per_query"],
        gpu_utilization_pct=request.gpu_utilization_pct,
        enterprise_api_discount_pct=request.enterprise_api_discount_pct,
        burstiness_factor=request.burstiness_factor,
        failover_reserve_pct=request.failover_reserve_pct,
    )

    recommendation = generate_recommendation(scenarios)

    return {
        "input": request.description,
        "parsed_workload": parsed,
        "scenarios": scenarios,
        "recommendation": recommendation,
    }


# ─── Pricing status ───────────────────────────────────────────────────────────

@app.get("/pricing/status")
def pricing_status():
    """Returns snapshot date, staleness, and last refresh summary."""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
        snapshot_date = snapshot.get("snapshot_date", "unknown")
        source_note = snapshot.get("source_note", "")
    except Exception:
        snapshot_date = "unknown"
        source_note = ""

    return {
        "snapshot_date": snapshot_date,
        "source_note": source_note,
        "last_refresh_summary": _last_refresh_summary,
        "refresh_schedule": "bi-weekly (every 14 days)",
        "manual_refresh_command": "python scripts/refresh_pricing.py",
    }


@app.post("/pricing/refresh")
async def trigger_pricing_refresh(dry_run: bool = False):
    """Manually trigger a pricing refresh (admin use). Pass ?dry_run=true to preview."""
    global _last_refresh_summary
    try:
        from backend.pricing_refresh import refresh_pricing
        summary = refresh_pricing(dry_run=dry_run)
        _last_refresh_summary = summary
        return summary
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─── ADK pipeline endpoints ───────────────────────────────────────────────────

class AdkPlanRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: str = "default_user"


class AdkSimulateRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: str = "default_user"
    gpu_utilization_pct: float = Field(default=70, ge=30, le=100)
    enterprise_api_discount_pct: float = Field(default=0, ge=0, le=40)
    burstiness_factor: Literal["low", "medium", "high"] = "medium"
    failover_reserve_pct: float = Field(default=15, ge=0, le=50)


@app.post("/adk/plan")
async def adk_plan(request: AdkPlanRequest):
    """
    ADK ParseJudgeLoop: parse user's workload description and validate it.

    Returns one of:
      - verdict="pass"       → workload_spec is ready; call /adk/simulate next
      - verdict="needs_user" → clarifying_question must be shown to the user
      - verdict="retry"      → circuit-breaker fired (treat like needs_user)

    Pass session_id back in subsequent requests to maintain conversation state.
    """
    import os
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY or GEMINI_API_KEY must be set to use the ADK pipeline.",
        )

    from agents.runner import run_parse_judge
    try:
        result = await run_parse_judge(
            user_message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class AdkRecalculateRequest(BaseModel):
    workload_spec: dict[str, Any]
    user_id: str = "default_user"


@app.post("/adk/recalculate")
async def adk_recalculate(request: AdkRecalculateRequest):
    """
    Re-run cost engine + ReasoningAgent on a user-corrected workload spec,
    skipping the ParseJudgeLoop entirely.

    Expects workload_spec with at minimum:
      monthly_queries, input_tokens_per_query, output_tokens_per_query
    """
    spec = request.workload_spec
    required = {"monthly_queries", "input_tokens_per_query", "output_tokens_per_query"}
    missing = required - set(spec.keys())
    if missing or not all(int(spec.get(k, 0) or 0) > 0 for k in required):
        raise HTTPException(
            status_code=422,
            detail=f"workload_spec must have positive values for: {', '.join(required)}",
        )

    from agents.runner import run_with_spec
    try:
        result = await run_with_spec(
            workload_spec=spec,
            user_id=request.user_id,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/adk/simulate")
async def adk_simulate(request: AdkSimulateRequest):
    """
    ADK full pipeline: ParseJudgeLoop → PricingAgent → ReasoningAgent,
    plus deterministic cost_engine scenarios.

    If verdict == "needs_user", the response contains only clarifying_question
    and no cost data — the caller must surface the question to the user.

    Pass session_id back in subsequent requests for multi-turn continuity.
    """
    import os
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY or GEMINI_API_KEY must be set to use the ADK pipeline.",
        )

    from agents.runner import run_full_pipeline
    try:
        result = await run_full_pipeline(
            user_message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc