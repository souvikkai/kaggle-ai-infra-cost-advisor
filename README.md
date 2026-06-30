# AI Infrastructure Cost Advisor

A multi-agent tool that answers one question: **should your AI workload call a managed API (OpenAI, Anthropic, Gemini) or self-host an open-weight model on rented GPU capacity?** It takes a plain-English workload description, parses it into a structured spec, runs a deterministic cost model across API providers and GPU providers, and returns a ranked recommendation with a 5× growth projection.

Built with Google ADK 2.x, FastAPI, and Next.js. Submitted to the Google Agent Development Kit Hackathon (July 2026).

## Architecture

```
User message
    │
    ▼
ParseJudgeLoop      — extracts monthly_queries, input_tokens, output_tokens; asks
    │                  clarifying questions if insufficient
    ▼
PricingAgent        — fetches live GPU prices via MCP server (AWS Trainium2 CB rate,
    │                  GCP TPU v5e, CoreWeave H100, Lambda Labs H100)
    ▼
cost_engine_bridge  — deterministic cost calculation across all providers × 3 growth
    │                  scenarios (current, 2×, 5×)
    ▼
ReasoningAgent      — synthesises cost data, latency context, and ops complexity
                       into a structured recommendation
```

## Quickstart

```bash
# Backend
pip install -r requirements.txt
uvicorn backend.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:3000`. Describe your workload in plain English and hit Enter.

## GPU providers covered

| Provider | Chip | $/chip-hr (snapshot) |
|---|---|---|
| CoreWeave | H100 SXM5 | $6.16 |
| Lambda Labs | H100 SXM4 | $3.29 |
| AWS | Trainium2 (Capacity Block) | $1.85 |
| GCP | TPU v5e | $1.20 |

Prices are kept in `data/pricing_snapshot.json`. GCP live pricing requires `GOOGLE_CLOUD_API_KEY`. AWS Trainium2 has no on-demand SKU (see Scope Limitations below).

## Scope limitations

The following are intentional scope boundaries, not oversights.

### 1. GPU sizing uses minimum-GPU counts, not throughput-based scaling

`cost_engine.py` calculates self-hosted GPU costs as `hourly_cost_per_gpu × min_gpus × hours_per_month`. The `min_gpus` figure comes from each open-weight model's minimum hardware requirement (e.g. 2 GPUs for Llama 3.1 70B), not from the actual throughput needed to serve the stated query volume.

At sufficiently high volumes this understates self-hosted costs. For example, at 50 million queries/month the engine recommends API because GPU costs look lower than they are — but throughput-based sizing would require ~30 H100s (~$133K/mo), which flips the recommendation to GPU. This is documented in eval case HP-02 in `evals/run_evals.py`.

A correct model would compute `ceil(monthly_queries × output_tokens / (tokens_per_second_per_gpu × 3600 × hours_per_month))` and multiply by GPU price. That extension is out of scope for this version.

### 2. Latency is not a factor in the deterministic cost model

`cost_engine.py` produces cost figures only. Latency SLAs — which in practice often determine whether API or self-hosted GPU is viable — are handled qualitatively by the Reasoning Agent's prompt, not by the cost engine itself.

A concrete case where this matters: a real-time fraud detection system requiring sub-100ms inference at 10 million checks per day. The cost engine recommends API (short 200-token payloads keep per-query cost low), but co-located GPU would be strongly preferred on latency grounds. This gap is documented in eval case EC-05 in `evals/run_evals.py`.

Incorporating latency economics would require additional inputs (inference latency target, network topology, geographic constraints) and a separate latency model.

### 3. This tool answers "API vs. self-hosted model," not "where do I host my application"

The GPU providers in this tool (AWS Trainium2, GCP TPU v5e, CoreWeave, Lambda Labs) are relevant only in the **self-hosting branch** — where an enterprise downloads open-weight model weights (Llama, Qwen, etc.) and rents GPU capacity to run inference. They are not general-purpose compute recommendations.

Calling a managed API (OpenAI, Anthropic, Gemini, or an enterprise channel like Azure OpenAI Service) requires **zero GPU infrastructure decisions** on the user's part. The model provider handles all hardware.

General application hosting — web servers, databases, user-facing app infrastructure — is a separate decision entirely and outside this tool's scope, regardless of whether the workload ultimately calls an API or self-hosts a model.
