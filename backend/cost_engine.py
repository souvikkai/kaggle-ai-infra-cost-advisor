import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pricing_snapshot.json"

SCENARIOS = {
    "current": 1,
    "growth_2x": 2,
    "growth_5x": 5,
}

BURSTINESS_MULTIPLIERS = {
    "low": 1.0,
    "medium": 1.15,
    "high": 1.35,
}


def load_pricing():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def apply_enterprise_discount(cost: float, discount_pct: float) -> float:
    return round(cost * (1 - discount_pct / 100.0), 2)


# ─── API model costs ──────────────────────────────────────────────────────────

def calculate_api_cost(
    model_pricing: dict,
    monthly_queries: int,
    input_tokens_per_query: int,
    output_tokens_per_query: int,
    enterprise_api_discount_pct: float = 0,
) -> dict:
    input_cost = (monthly_queries * input_tokens_per_query / 1_000_000) * model_pricing["input_per_million"]
    output_cost = (monthly_queries * output_tokens_per_query / 1_000_000) * model_pricing["output_per_million"]
    monthly_cost = apply_enterprise_discount(input_cost + output_cost, enterprise_api_discount_pct)
    cost_per_query = monthly_cost / monthly_queries if monthly_queries else 0
    return {
        "monthly_cost": round(monthly_cost, 2),
        "cost_per_query": round(cost_per_query, 6),
        "cost_per_1k_queries": round(cost_per_query * 1000, 4),
    }


def calculate_all_api_costs(
    monthly_queries: int,
    input_tokens_per_query: int,
    output_tokens_per_query: int,
    enterprise_api_discount_pct: float = 0,
) -> list[dict]:
    pricing = load_pricing()
    results = []
    for model_key, model_pricing in pricing["models"].items():
        cost = calculate_api_cost(
            model_pricing=model_pricing,
            monthly_queries=monthly_queries,
            input_tokens_per_query=input_tokens_per_query,
            output_tokens_per_query=output_tokens_per_query,
            enterprise_api_discount_pct=enterprise_api_discount_pct,
        )
        results.append({
            "model_key": model_key,
            "display_name": model_pricing.get("display_name", model_key),
            "provider": model_pricing["provider"],
            "tier": model_pricing.get("tier", "unknown"),
            **cost,
        })
    return sorted(results, key=lambda x: x["monthly_cost"])


# ─── GPU provider costs ───────────────────────────────────────────────────────

def calculate_gpu_provider_cost(
    provider_key: str,
    provider_config: dict,
    monthly_queries: int,
    input_tokens_per_query: int,
    output_tokens_per_query: int,
    gpu_utilization_pct: float = 70,
    burstiness_factor: str = "medium",
    failover_reserve_pct: float = 15,
) -> dict:
    total_tokens = monthly_queries * (input_tokens_per_query + output_tokens_per_query)
    seconds_per_month = 30 * 24 * 3600
    tokens_per_sec_per_gpu = provider_config["tokens_per_second_per_gpu"]
    hourly_cost = provider_config["hourly_cost_per_gpu"]

    required_tps = total_tokens / seconds_per_month
    utilization = max(gpu_utilization_pct / 100.0, 0.01)
    burstiness = BURSTINESS_MULTIPLIERS.get(burstiness_factor, 1.15)
    failover = 1 + failover_reserve_pct / 100.0

    base_gpus = required_tps / tokens_per_sec_per_gpu
    gpu_count = max(1, round((base_gpus / utilization) * burstiness * failover + 0.5))

    monthly_cost = round(gpu_count * hourly_cost * 24 * 30, 2)
    cost_per_query = monthly_cost / monthly_queries if monthly_queries else 0

    return {
        "provider_key": provider_key,
        "display_name": provider_config["display_name"],
        "provider": provider_config["provider"],
        "chip": provider_config["chip"],
        "notes": provider_config.get("notes", ""),
        "estimated_gpu_count": gpu_count,
        "monthly_cost": monthly_cost,
        "cost_per_query": round(cost_per_query, 6),
        "cost_per_1k_queries": round(cost_per_query * 1000, 4),
    }


def calculate_all_gpu_provider_costs(
    monthly_queries: int,
    input_tokens_per_query: int,
    output_tokens_per_query: int,
    gpu_utilization_pct: float = 70,
    burstiness_factor: str = "medium",
    failover_reserve_pct: float = 15,
    gpu_price_overrides: dict | None = None,
) -> list[dict]:
    pricing = load_pricing()
    # Shallow-copy provider configs so overrides don't mutate the loaded dict
    gpu_providers = {k: dict(v) for k, v in pricing["gpu_providers"].items()}
    if gpu_price_overrides:
        for provider_key, override in gpu_price_overrides.items():
            if provider_key in gpu_providers and "hourly_cost_per_gpu" in override:
                gpu_providers[provider_key]["hourly_cost_per_gpu"] = override["hourly_cost_per_gpu"]
                gpu_providers[provider_key]["_price_source"] = override.get("source", "live")
    results = []
    for provider_key, provider_config in gpu_providers.items():
        cost = calculate_gpu_provider_cost(
            provider_key=provider_key,
            provider_config=provider_config,
            monthly_queries=monthly_queries,
            input_tokens_per_query=input_tokens_per_query,
            output_tokens_per_query=output_tokens_per_query,
            gpu_utilization_pct=gpu_utilization_pct,
            burstiness_factor=burstiness_factor,
            failover_reserve_pct=failover_reserve_pct,
        )
        if gpu_price_overrides and provider_key in gpu_price_overrides:
            cost["_price_source"] = gpu_price_overrides[provider_key].get("source", "live")
        results.append(cost)
    return sorted(results, key=lambda x: x["monthly_cost"])


# ─── Open-weight model costs on GPU ──────────────────────────────────────────

# Toolchain friction for each (model_family, provider) pair.
# CoreWeave/Lambda Labs = NVIDIA CUDA → any model "just works" → "none".
# AWS Trainium2 requires Neuron SDK; Meta/AWS have official Neuron support for
# Llama models, making that pairing "moderate". All other models on Trainium2
# require non-trivial porting → "high".
# GCP TPU v5e targets JAX/XLA; no open-weight model has official JAX support
# in the same sense, so all pairings are "moderate" for Llama (community
# ports exist) and "high" for everything else.
_TOOLCHAIN_FRICTION: dict[tuple[str, str], str] = {
    # AWS Trainium2 — Neuron SDK required
    ("llama_31_8b",   "aws_trainium2"): "moderate",
    ("llama_31_70b",  "aws_trainium2"): "moderate",
    ("llama_31_405b", "aws_trainium2"): "moderate",
    ("qwen3_32b",     "aws_trainium2"): "high",
    # GCP TPU v5e — JAX/XLA required
    ("llama_31_8b",   "gcp_tpu_v5"): "moderate",
    ("llama_31_70b",  "gcp_tpu_v5"): "moderate",
    ("llama_31_405b", "gcp_tpu_v5"): "high",
    ("qwen3_32b",     "gcp_tpu_v5"): "high",
}
_FRICTION_NOTES: dict[str, str] = {
    "none":     "Standard NVIDIA CUDA — no porting required.",
    "moderate": (
        "Requires toolchain porting (Neuron SDK for AWS Trainium2, JAX/XLA for GCP TPU v5e). "
        "Community or official support exists for this model, but expect 1–4 weeks of engineering work."
    ),
    "high": (
        "Requires non-trivial toolchain porting with limited community support for this model. "
        "Expect 4–12 weeks of engineering effort before production inference is viable."
    ),
}


def _toolchain_friction(model_key: str, provider_key: str) -> str:
    """Return 'none' | 'moderate' | 'high' for a model × provider pair."""
    return _TOOLCHAIN_FRICTION.get((model_key, provider_key), "none")


def calculate_open_weight_option_cost(
    model_key: str,
    model_config: dict,
    provider_key: str,
    provider_config: dict,
    monthly_queries: int,
    input_tokens_per_query: int,
    output_tokens_per_query: int,
    gpu_utilization_pct: float = 70,
    burstiness_factor: str = "medium",
    failover_reserve_pct: float = 15,
) -> dict:
    total_tokens = monthly_queries * (input_tokens_per_query + output_tokens_per_query)
    seconds_per_month = 30 * 24 * 3600

    tps_multiplier = model_config.get("tps_multiplier", 1.0)
    min_gpus = model_config.get("min_gpus", 1)
    effective_tps_per_gpu = provider_config["tokens_per_second_per_gpu"] * tps_multiplier

    required_tps = total_tokens / seconds_per_month
    utilization = max(gpu_utilization_pct / 100.0, 0.01)
    burstiness = BURSTINESS_MULTIPLIERS.get(burstiness_factor, 1.15)
    failover = 1 + failover_reserve_pct / 100.0

    raw_gpus = (required_tps / effective_tps_per_gpu / utilization) * burstiness * failover
    # Round up to nearest multiple of min_gpus (model parallelism floor)
    import math
    gpu_count = max(min_gpus, int(math.ceil(raw_gpus / min_gpus)) * min_gpus)

    monthly_cost = round(gpu_count * provider_config["hourly_cost_per_gpu"] * 24 * 30, 2)
    cost_per_query = monthly_cost / monthly_queries if monthly_queries else 0

    friction = _toolchain_friction(model_key, provider_key)
    return {
        "option_key": f"{model_key}__{provider_key}",
        "model_key": model_key,
        "provider_key": provider_key,
        "display_name": f"{model_config['display_name']} on {provider_config['display_name']}",
        "model_display_name": model_config["display_name"],
        "provider_display_name": provider_config["display_name"],
        "quality_tier": model_config.get("quality_tier", "unknown"),
        "comparable_api": model_config.get("comparable_api", ""),
        "estimated_gpu_count": gpu_count,
        "monthly_cost": monthly_cost,
        "cost_per_query": round(cost_per_query, 6),
        "cost_per_1k_queries": round(cost_per_query * 1000, 4),
        "toolchain_friction": friction,
        "toolchain_friction_note": _FRICTION_NOTES[friction],
    }


def calculate_all_open_weight_costs(
    monthly_queries: int,
    input_tokens_per_query: int,
    output_tokens_per_query: int,
    gpu_utilization_pct: float = 70,
    burstiness_factor: str = "medium",
    failover_reserve_pct: float = 15,
    gpu_price_overrides: dict | None = None,
) -> list[dict]:
    pricing = load_pricing()
    gpu_providers = {k: dict(v) for k, v in pricing["gpu_providers"].items()}
    if gpu_price_overrides:
        for provider_key, override in gpu_price_overrides.items():
            if provider_key in gpu_providers and "hourly_cost_per_gpu" in override:
                gpu_providers[provider_key]["hourly_cost_per_gpu"] = override["hourly_cost_per_gpu"]
    results = []
    for model_key, model_config in pricing.get("open_weight_models", {}).items():
        for provider_key, provider_config in gpu_providers.items():
            cost = calculate_open_weight_option_cost(
                model_key=model_key,
                model_config=model_config,
                provider_key=provider_key,
                provider_config=provider_config,
                monthly_queries=monthly_queries,
                input_tokens_per_query=input_tokens_per_query,
                output_tokens_per_query=output_tokens_per_query,
                gpu_utilization_pct=gpu_utilization_pct,
                burstiness_factor=burstiness_factor,
                failover_reserve_pct=failover_reserve_pct,
            )
            results.append(cost)
    return sorted(results, key=lambda x: x["monthly_cost"])


# ─── Reference model tiers for frontend display ───────────────────────────────

REFERENCE_MODEL_KEYS = {
    "budget_api": "gpt_4o_mini",
    "premium_api": "claude_sonnet_4_6",
    "frontier_api": "gpt_5_5",
    "google_flash": "gemini_flash_35",
}


def get_reference_models(model_costs: list[dict]) -> dict:
    costs_by_key = {item["model_key"]: item for item in model_costs}
    return {
        label: costs_by_key[key]
        for label, key in REFERENCE_MODEL_KEYS.items()
        if key in costs_by_key
    }


# ─── Scenario engine ──────────────────────────────────────────────────────────

def calculate_scenarios(
    monthly_queries: int,
    input_tokens_per_query: int,
    output_tokens_per_query: int,
    gpu_utilization_pct: float = 70,
    enterprise_api_discount_pct: float = 0,
    burstiness_factor: str = "medium",
    failover_reserve_pct: float = 15,
    gpu_price_overrides: dict | None = None,
) -> list[dict]:
    scenario_results = []

    for scenario_name, multiplier in SCENARIOS.items():
        q = monthly_queries * multiplier

        api_costs = calculate_all_api_costs(
            monthly_queries=q,
            input_tokens_per_query=input_tokens_per_query,
            output_tokens_per_query=output_tokens_per_query,
            enterprise_api_discount_pct=enterprise_api_discount_pct,
        )

        gpu_costs = calculate_all_gpu_provider_costs(
            monthly_queries=q,
            input_tokens_per_query=input_tokens_per_query,
            output_tokens_per_query=output_tokens_per_query,
            gpu_utilization_pct=gpu_utilization_pct,
            burstiness_factor=burstiness_factor,
            failover_reserve_pct=failover_reserve_pct,
            gpu_price_overrides=gpu_price_overrides,
        )

        open_weight_costs = calculate_all_open_weight_costs(
            monthly_queries=q,
            input_tokens_per_query=input_tokens_per_query,
            output_tokens_per_query=output_tokens_per_query,
            gpu_utilization_pct=gpu_utilization_pct,
            burstiness_factor=burstiness_factor,
            failover_reserve_pct=failover_reserve_pct,
            gpu_price_overrides=gpu_price_overrides,
        )

        # Cheapest GPU provider this scenario
        cheapest_gpu = gpu_costs[0]
        cheapest_open_weight = open_weight_costs[0] if open_weight_costs else None

        # Keep backward-compat self_hosted_h100 key pointing to CoreWeave H100
        coreweave = next((g for g in gpu_costs if g["provider_key"] == "coreweave_h100"), cheapest_gpu)

        scenario_results.append({
            "scenario": scenario_name,
            "monthly_queries": q,
            "cheapest_api_model": api_costs[0],
            "cheapest_gpu_provider": cheapest_gpu,
            "cheapest_open_weight_option": cheapest_open_weight,
            "reference_models": get_reference_models(api_costs),
            "all_api_models": api_costs,
            "gpu_providers": gpu_costs,
            "open_weight_options": open_weight_costs,
            # backward-compat for frontend components not yet updated
            "self_hosted_h100": {
                "self_hosted_monthly_cost": coreweave["monthly_cost"],
                "estimated_gpu_count": coreweave["estimated_gpu_count"],
                "self_hosted_cost_per_query": coreweave["cost_per_query"],
            },
        })

    return scenario_results
