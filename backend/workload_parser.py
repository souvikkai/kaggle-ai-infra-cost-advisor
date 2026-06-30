def parse_workload(description: str):

    description_lower = description.lower()

    workload = {
        "monthly_queries": 100000,
        "input_tokens_per_query": 1000,
        "output_tokens_per_query": 300,
        "latency_sla_ms": 3000,
    }

    # Enterprise scale
    if "enterprise" in description_lower:
        workload["monthly_queries"] = 5_000_000

    # Startup scale
    elif "startup" in description_lower:
        workload["monthly_queries"] = 50_000

    # Research / copilot workloads
    if "research" in description_lower or "copilot" in description_lower:
        workload["input_tokens_per_query"] = 2500
        workload["output_tokens_per_query"] = 800

    # Customer support chatbot
    if "support" in description_lower or "chatbot" in description_lower:
        workload["input_tokens_per_query"] = 700
        workload["output_tokens_per_query"] = 250

    # Low latency requirement
    if "real-time" in description_lower:
        workload["latency_sla_ms"] = 500

    return workload