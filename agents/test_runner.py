"""
Runner smoke tests — validates session management and cost_engine wiring
WITHOUT making real LLM calls.

Run with: python -m agents.test_runner
"""

import asyncio
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Windows cp1252 can't encode ✓ / ⚠ — force UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


async def test_session_create_and_retrieve():
    from agents.runner import _session_service, _get_or_create_session, APP_NAME

    sid = "test-session-001"
    uid = "test-user"
    state = {"workload_spec": '{"monthly_queries": 50000}'}

    session = await _get_or_create_session(sid, uid, initial_state=state)
    assert session is not None, "Session should be created"
    assert session.id == sid

    # Retrieve the same session
    retrieved = await _session_service.get_session(
        app_name=APP_NAME, user_id=uid, session_id=sid
    )
    assert retrieved is not None
    assert retrieved.state.get("workload_spec") is not None
    print("  ✓ Session create and retrieve")


def test_parse_json_field():
    from agents.runner import _parse_json_field

    assert _parse_json_field('{"a": 1}') == {"a": 1}
    assert _parse_json_field({"a": 1}) == {"a": 1}
    assert _parse_json_field(None) is None
    assert _parse_json_field("not-json") is None
    assert _parse_json_field([1, 2]) == [1, 2]
    print("  ✓ _parse_json_field handles all cases")


def test_cost_engine_integration():
    from agents.runner import _run_cost_engine

    spec = {
        "monthly_queries": 50000,
        "input_tokens_per_query": 800,
        "output_tokens_per_query": 300,
    }
    scenarios = _run_cost_engine(spec)
    assert scenarios is not None, "Should return scenarios"
    assert len(scenarios) == 3, f"Expected 3 scenarios (1x/2x/5x), got {len(scenarios)}"
    names = [s["scenario"] for s in scenarios]
    assert names == ["current", "growth_2x", "growth_5x"], f"Unexpected scenario names: {names}"
    print(f"  ✓ cost_engine returned {len(scenarios)} scenarios")
    for s in scenarios:
        print(f"    {s['scenario']}: {s['monthly_queries']:,} queries, "
              f"cheapest API ${s['cheapest_api_model']['monthly_cost']:,.0f}/mo, "
              f"cheapest GPU ${s['cheapest_gpu_provider']['monthly_cost']:,.0f}/mo")


def test_cost_engine_missing_fields():
    from agents.runner import _run_cost_engine

    result = _run_cost_engine({"monthly_queries": 50000})  # missing token fields
    assert result is None, "Should return None when volume fields are missing"
    print("  ✓ cost_engine returns None when fields missing")


async def main():
    print("=== Runner Smoke Tests ===\n")
    await test_session_create_and_retrieve()
    test_parse_json_field()
    test_cost_engine_integration()
    test_cost_engine_missing_fields()
    print("\n✓ All runner tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
