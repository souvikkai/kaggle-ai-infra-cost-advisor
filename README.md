# AI Infrastructure Cost Advisor

**Track: Agents for Business — Google × Kaggle Agentic AI Bootcamp (July 2026)**

A multi-agent tool that answers one question every engineering team eventually faces:
**should your AI workload call a managed API or self-host an open-weight model on GPU?**

It takes a plain-English workload description, parses it into a structured specification,
runs a deterministic cost model across 6 API providers and 6 GPU/open-weight options,
and returns a ranked recommendation with a 5× growth projection — including latency risk
flags, toolchain friction warnings, and a breakeven analysis.

**Live demo:** https://kaggle-ai-infra-cost-advisor.vercel.app
**Backend API:** https://kaggle-ai-infra-cost-advisor-production.up.railway.app/docs

---

## The Problem

Picking AI infrastructure is a high-stakes, high-confusion decision. A team building
a customer support chatbot at 50,000 queries/month should almost certainly call an API.
The same team at 50 million queries/month should almost certainly self-host. Between
those points lies a breakeven that depends on token counts, traffic patterns, latency
requirements, and toolchain maturity — none of which a spreadsheet handles well.

Most teams either over-engineer (self-hosting before it pencils out) or under-engineer
(sticking with API long past the point where it costs 10× more than self-hosting would).
This tool makes the decision transparent and reproducible.

---

## Why Agents?

This is not a problem a single LLM call can solve reliably. The decision requires:

1. **Structured extraction** from free-form natural language into precise numeric specs
2. **Validation** — catching ambiguous or self-contradictory inputs before running math
3. **Live pricing data** — GPU costs change; hardcoded numbers go stale
4. **Deterministic cost math** — the recommendation must be reproducible and auditable
5. **Narrative synthesis** — translating cost tables into a defensible recommendation

Each of these is a distinct capability. Collapsing them into one prompt produces outputs
that are simultaneously confident and wrong. A pipeline of specialized agents — each with
a narrow, well-defined job — produces outputs that are verifiable at each stage.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STAGE 1: Parse + Judge                    │
│                                                                   │
│  User message                                                     │
│      │                                                            │
│      ▼                                                            │
│  ParsingAgent (Gemini 2.5 Flash via Google ADK)                  │
│      │  Extracts: monthly_queries, input_tokens, output_tokens,   │
│      │  latency_sla, traffic_pattern, reasoning_complexity, etc.  │
│      │  Tracks field provenance (stated vs estimated)             │
│      │  Runs feasibility check (e.g. self-hosting Claude = blocked)│
│      ▼                                                            │
│  JudgeAgent (Gemini 2.5 Flash via Google ADK)                    │
│      │  Validates spec completeness and internal consistency       │
│      │  Advisory Mode: asks ONE targeted clarifying question       │
│      │  rather than silently guessing ambiguous fields             │
│      ▼                                                            │
│  verdict_router (FunctionNode)                                    │
│      ├─ "retry"      → back to ParsingAgent (circuit breaker)    │
│      ├─ "needs_user" → return clarifying question to frontend     │
│      ├─ "infeasible" → return explanation (e.g. can't self-host  │
│      │                  a closed model), no further processing    │
│      └─ "pass"       → proceed to Stage 2                        │
└─────────────────────────────────────────────────────────────────┘
                              │ (pass only)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       STAGE 2: Price + Reason                    │
│                                                                   │
│  PricingAgent (Gemini 2.5 Flash via Google ADK)                  │
│      │  Calls MCP server (stdio subprocess) with 3 tools:        │
│      │  • get_gpu_pricing — live GPU provider costs               │
│      │  • get_api_pricing — managed API token costs               │
│      │  • get_open_weight_options — open-weight model configs     │
│      ▼                                                            │
│  cost_engine_bridge (Python FunctionNode)                        │
│      │  Deterministic cost math across 3 growth scenarios         │
│      │  (1×, 2×, 5×) for all providers simultaneously            │
│      │  Applies traffic_pattern burstiness multiplier to GPU      │
│      │  capacity (smooth=1.0×, predictable_peaks=1.15×,          │
│      │  spiky=1.35×)                                             │
│      ▼                                                            │
│  ReasoningAgent (Claude Haiku 4.5 via Anthropic API)             │
│      │  Synthesises cost data + latency + toolchain friction      │
│      │  3-path latency classification:                            │
│      │  • HARD: explicit ms/s requirement or domain-implied       │
│      │    urgency (HFT, fraud screening) → API disqualified       │
│      │  • SOFT: preference without requirement → cost can         │
│      │    override with explicit tradeoff note                    │
│      │  • Batch: throughput-only → pure cost decision             │
│      └─ Returns structured JSON recommendation                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key design decisions

**Two-stage gating:** Stage 2 (pricing + reasoning, ~40s) only runs if Stage 1
produces a valid, unambiguous spec. Clarification rounds skip Stage 2 entirely,
cutting response time by ~60% for the most common case (first message needs refinement).

**Deterministic cost engine:** `cost_engine.py` is pure math — no LLM involved.
This means cost figures are reproducible, auditable, and consistent. The ReasoningAgent
receives ground-truth numbers and is explicitly instructed not to invent them.

**Traffic pattern modeling:** The system infers whether traffic is smooth (24/7 steady),
predictably peaked (business hours, daily batch), or spiky (HFT, fraud detection).
This drives a GPU capacity multiplier (1.0×–1.35×) because self-hosted GPU must be
provisioned for peak load, not average load — a distinction that significantly affects
the breakeven calculation for high-burstiness workloads.

**Infeasibility detection:** If a user asks to self-host Claude, GPT, or Gemini, the
ParsingAgent short-circuits before any cost calculation, explaining that closed-model
weights aren't released. This is architecturally important: the system knows the
difference between "I don't have enough information" and "this is impossible."

---

## Course Concepts Demonstrated

| Concept | Where |
|---|---|
| **Multi-agent system (ADK)** | `agents/pipeline.py` — 4-agent pipeline using Google ADK 2.x: ParsingAgent, JudgeAgent, PricingAgent, ReasoningAgent |
| **MCP Server** | `mcp_servers/gpu_pricing_server.py` — FastMCP server with 3 tools, called via stdio subprocess from PricingAgent |
| **Security features** | `backend/main.py` — prompt injection guard (12 regex patterns), 4,000-char input limit, CORS origin allowlist, secrets via env vars only |
| **Deployability** | Railway (backend, 4 workers) + Vercel (frontend) — live public URLs, environment variable management, `railway.json` start command |

---

## Security Implementation

Security is implemented at the API boundary before input reaches any LLM:

**1. Prompt injection guard** (`backend/main.py: _validate_message`)
Twelve regex patterns catch the most common override attempts:
- "ignore all previous instructions"
- "act as a [persona]"
- "reveal your system prompt"
- `<script>` injection attempts
- Direct "prompt injection" keyword

Returns HTTP 400 with a plain-English explanation rather than silently processing
or returning a cryptic error.

**2. Input length validation**
4,000-character ceiling on all user messages. Prevents runaway token costs and
timeout failures that would otherwise look like backend crashes.

**3. CORS allowlist**
`ALLOWED_ORIGINS` environment variable restricts which frontend origins the backend
accepts requests from. Hardcoded localhost for development; production Vercel URL
set as a Railway environment variable.

**4. Secrets management**
All API keys in environment variables only. `.env` is gitignored. `.env.example`
ships with placeholder values so contributors know what to configure without
exposing real credentials.

---

## Eval Harness

`evals/run_evals.py` contains 18 test cases across three categories:

- **Happy path (2):** clean, well-specified inputs
- **Edge cases (9):** qualitative answers, volume range inputs, multi-tier workloads,
  traffic pattern inference
- **Adversarial (7):** prompt injection attempts, self-hosting closed models,
  hard latency contradictions, team-size vs volume contradictions

Each test case documents the specific bug or behavior it was written to catch.
Cases TP-01 and TP-02 use the new string-equality parse check format to assert
that categorical fields like `traffic_pattern` are inferred correctly
(`"spiky"`, `"smooth"`) rather than just checking numeric ranges.

```bash
python evals/run_evals.py --url http://localhost:8006
```

---

## Project Structure

```
p7-ai-infra-cost-simulator/
├── agents/
│   ├── pipeline.py          # All 4 ADK agents + FunctionNodes + graph edges
│   ├── runner.py            # Two-stage run_full_pipeline(), session management
│   ├── test_skeleton.py     # Graph structure smoke tests
│   └── test_runner.py       # Unit tests for runner helpers
├── backend/
│   ├── main.py              # FastAPI app, security validation, CORS, endpoints
│   ├── cost_engine.py       # Deterministic cost math (no LLM)
│   ├── recommendation_engine.py
│   └── workload_parser.py
├── mcp_servers/
│   └── gpu_pricing_server.py  # FastMCP server, 3 tools, stdio transport
├── data/
│   └── pricing_snapshot.json  # Dated pricing snapshot (2026-06-27)
├── evals/
│   └── run_evals.py         # 18-case eval harness with LLM-as-judge scoring
├── frontend/
│   ├── app/page.tsx         # Next.js frontend (~1,750 lines, single-page)
│   └── ...
├── scripts/
│   └── refresh_pricing.py   # Manual pricing refresh utility
├── .env.example             # Required env var names (no real values)
├── railway.json             # Railway start command
└── requirements.txt         # Python dependencies
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- A Google API key (Gemini) — get one at https://aistudio.google.com/app/apikey
- An Anthropic API key — get one at https://console.anthropic.com

### 1. Clone and configure

```bash
git clone https://github.com/souvikkai/kaggle-ai-infra-cost-advisor.git
cd kaggle-ai-infra-cost-advisor
cp .env.example .env
# Edit .env and fill in your real API keys
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8006
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. Describe your AI workload in plain English and press Enter.

> **Note:** The frontend's `API_BASE_URL` constant in `frontend/app/page.tsx` points
> to the live Railway backend by default. For local development, change it to
> `http://127.0.0.1:8006`.

---

## Deployment

### Backend (Railway)

1. Connect the GitHub repo to a new Railway project
2. Railway auto-detects Python from `requirements.txt`
3. `railway.json` sets the start command:
   ```json
   {"deploy": {"startCommand": "uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 4"}}
   ```
4. Set environment variables in Railway's Variables tab:

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key — powers all 3 ADK agents |
| `ANTHROPIC_API_KEY` | Anthropic key — powers the ReasoningAgent (Claude Haiku 4.5) |
| `ADK_MODEL` | `gemini-2.5-flash` |
| `LLM_PROVIDER` | `anthropic` |
| `USE_REAL_LLM_PLANNER` | `true` |
| `ALLOWED_ORIGINS` | Your Vercel frontend URL (e.g. `https://your-app.vercel.app`) |
| `GOOGLE_CLOUD_API_KEY` | Optional — enables live GCP TPU v5e pricing; falls back to snapshot |

5. Generate a public domain in Railway Settings → Networking

### Frontend (Vercel)

1. Connect the GitHub repo to a new Vercel project
2. Set **Root Directory** to `frontend`
3. Set **Framework Preset** to Next.js
4. Deploy — no environment variables needed (API URL is hardcoded in `page.tsx`)

---

## GPU Providers Covered

| Provider | Chip | Notes |
|---|---|---|
| CoreWeave | H100 SXM5 | On-demand, CUDA-native, widest model support |
| Lambda Labs | H100 SXM4 | Developer-focused, simple pricing |
| AWS | Trainium2 | ~50% cheaper for compatible workloads; Capacity Block only |
| GCP | TPU v5e | Cheapest per token for JAX/TF workloads; moderate toolchain friction |

Open-weight models covered: Llama 3.1 8B, Llama 3.1 70B, Qwen 2.5 7B, Qwen 2.5 72B,
Mistral 7B — each paired against all GPU providers for 12+ self-host configurations.

---

## Scope Limitations

The following are intentional scope boundaries, not oversights.

**1. GPU sizing uses throughput-based scaling, not min-GPU floor**
`cost_engine.py` computes GPU count from actual throughput requirements:
`ceil(required_tokens_per_second / tokens_per_second_per_gpu)`, adjusted by
traffic pattern burstiness multiplier. Min-GPU constraints (e.g. 2 GPUs minimum
for Llama 3.1 70B due to memory requirements) apply as a floor.

**2. Latency is modeled qualitatively, not quantitatively**
Latency SLAs are handled by the ReasoningAgent's 3-path classification (hard/soft/batch),
not by the deterministic cost engine. The cost engine does not model inference latency
as a function of GPU type or batch size.

**3. This tool answers "API vs. self-hosted inference," not "where to host your app"**
GPU providers here are relevant only in the self-hosting branch. General application
hosting (web servers, databases, app infrastructure) is outside this tool's scope.

**4. AWS Trainium2 has no on-demand SKU**
Trainium2 pricing uses the Capacity Block rate. On-demand Trainium2 availability
is limited; users should verify current availability before committing to this option.

---

## Known Issues

See `evals/run_evals.py` for documented regression cases. Current eval suite:
18 cases, categorized as happy_path (2), edge_case (9), adversarial (7).
Each case documents the specific behavior it was written to verify or the bug
it was written to catch.
