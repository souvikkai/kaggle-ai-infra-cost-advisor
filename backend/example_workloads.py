EXAMPLE_WORKLOADS = {
    "startup_support_bot": {
        "description": "Customer support chatbot for a seed-stage SaaS startup",
        "monthly_queries": 50000,
        "input_tokens_per_query": 500,
        "output_tokens_per_query": 200,
        "latency_sla_ms": 3000,
    },

    "series_a_research_assistant": {
        "description": "AI research copilot for a growing B2B platform",
        "monthly_queries": 500000,
        "input_tokens_per_query": 2000,
        "output_tokens_per_query": 800,
        "latency_sla_ms": 5000,
    },

    "enterprise_customer_agent": {
        "description": "Enterprise-scale AI support assistant for 10,000+ users",
        "monthly_queries": 5000000,
        "input_tokens_per_query": 3000,
        "output_tokens_per_query": 1200,
        "latency_sla_ms": 2000,
    },
}