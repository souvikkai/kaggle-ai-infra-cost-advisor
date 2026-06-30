# AI Infra Cost Simulator — P5

**Portfolio project · Souvik Kundu · AI PM Master Curriculum v2.0 · Week 4**

> **Resume line:** Built agentic AI infrastructure cost calculator handling ambiguous workload inputs; agent fetches live pricing across 5 providers and models cost-concurrency tradeoffs with structured recommendation output across CoreWeave, Lambda Labs, AWS Trainium2, and GCP TPU v5e.

---

## What it does

Describe an AI workload in plain English → the planner collects structured assumptions conversationally via a multi-turn Haiku conversation with full memory → the deterministic simulator models API and self-hosted GPU costs across current, 2×, and 5× growth scenarios.

**Key PM decisions demonstrated:**
- Separation of probabilistic planning (LLM) from deterministic cost calculation (cost_engine.py)
- ReAct agent pattern with circuit breaker to prevent infinite loops
- Confidence-threshold routing: planner only enables simulation when all required fields are resolved
- Real provider names throughout: CoreWeave H100, Lambda Labs H100, AWS Trainium2, GCP TPU v5e
- API models: Claude Haiku 4.5, Claude Sonnet 4.6, GPT-4o Mini, GPT-5.5, Gemini Flash, Gemini Pro

---

## Architecture

```
user message
  → POST /plan  (FastAPI)
      → LLM planner (Haiku with full conversation history)
          → ReAct agent (backend/agent.py)
              → tool: validate_assumptions
              → tool: estimate_token_volumes
              → tool: compute_quick_cost
      → structured assumptions (provenance-tagged)
  → POST /simulate  (FastAPI)
      → deterministic cost_engine.py
      → scenarios: current / 2x / 5x growth
      → all_api_models × all_gpu_providers
```

### ReAct agent tools (all deterministic, no LLM calls)

| Tool | Purpose |
|---|---|
| `validate_assumptions` | Check which required fields are present/missing |
| `estimate_token_volumes` | Infer token counts from workload description heuristics |
| `get_provider_pricing` | Look up current GPU or API pricing from snapshot |
| `compute_quick_cost` | Back-of-envelope cost sanity check before simulation |

**Circuit breaker:** `MAX_TURNS = 8` — agent exits and returns partial assumptions if loop does not converge.

---

## Running locally

```bash
# Backend (Python 3.11+)
cd p7-ai-infra-cost-simulator
pip install fastapi uvicorn python-dotenv
uvicorn backend.main:app --reload --port 8000

# Frontend (Node 18+)
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### Environment variables (`.env` in project root)

```
USE_REAL_LLM_PLANNER=true
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=your_key_here

# Optional: enable LLM-driven ReAct loop
USE_REACT_AGENT=true
```

---

## Pricing snapshot

`data/pricing_snapshot.json` — manually maintained, dated `2026-05-25`.

**API models:** Claude Haiku 4.5 ($1/$5), Claude Sonnet 4.6 ($3/$15), GPT-4o Mini ($0.15/$0.60), GPT-5.5 ($5/$30), Gemini 3.1 Flash ($0.50/$3), Gemini 3.1 Pro ($2/$12)

**GPU providers:** CoreWeave H100 ($2.95/hr), Lambda Labs H100 ($2.49/hr), AWS Trainium2 ($1.85/hr), GCP TPU v5e ($2.20/hr)

---

## Curriculum topics covered

Transformer architecture · GPU memory hierarchy · KV cache · inference serving stack · vLLM / PagedAttention · speculative decoding · LangChain / ReAct agents · tool use / function calling · circuit breaker pattern · MLOps observability · cloud production layer (Kubernetes, API gateway, caching) · foundation model economics · neocloud business models

---

## Portfolio context

Part of [souvik-ai-pm-portfolio](https://github.com/souvikkai/souvik-ai-pm-portfolio) — built during the 6-Week AI PM Master Curriculum. P5 of 7 projects. Distinct from P2 (Neocloud Cluster Configurator, hardware/silicon focused) — this project focuses on API economics and cloud cost modeling.
