def generate_recommendation(scenario_results):

    current = scenario_results[0]
    growth_5x = scenario_results[-1]

    current_api = current["cheapest_api_model"]
    current_h100 = current["self_hosted_h100"]

    growth_api = growth_5x["cheapest_api_model"]
    growth_h100 = growth_5x["self_hosted_h100"]

    if current_api["monthly_cost"] < current_h100["self_hosted_monthly_cost"]:

        recommended = "API"

        rationale = (
            f"API inference is cheaper at current scale. "
            f"The cheapest API option is "
            f"{current_api['model_key']} "
            f"at ${current_api['monthly_cost']}/month."
        )

    else:

        recommended = "Self-hosted H100"

        rationale = (
            f"Self-hosting is cheaper at current scale. "
            f"H100 monthly cost is "
            f"${current_h100['self_hosted_monthly_cost']}/month."
        )

    if growth_api["monthly_cost"] < growth_h100["self_hosted_monthly_cost"]:

        migration_trigger = (
            "No migration trigger in modeled range. "
            "API inference remains economically favorable."
        )

    else:

        migration_trigger = (
            "At 5x growth, self-hosting becomes attractive. "
            "Evaluate dedicated GPU infrastructure."
        )

    return {
        "recommended_option": recommended,
        "rationale": rationale,
        "migration_trigger": migration_trigger,
    }