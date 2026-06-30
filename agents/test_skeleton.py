"""
Minimal smoke test for the ADK skeleton.

Validates that:
1. ParseJudgeLoop can be instantiated and graph edges are valid
2. The verdict_router FunctionNode is correctly wired
3. Full pipeline can be instantiated

Run with: python -m agents.test_skeleton
"""

import asyncio
import io
import json
import sys

# Windows cp1252 can't encode → / ✓ / ⚠ — force UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from agents.pipeline import parse_judge_workflow, full_pipeline, verdict_router


def test_graph_edges():
    """Validate that parse_judge_workflow has the expected edges."""
    wf = parse_judge_workflow
    assert wf.graph is not None, "Graph should be compiled on init"

    edge_pairs = [(e.from_node.name, e.to_node.name, e.route) for e in wf.graph.edges]
    print("ParseJudgeLoop edges:")
    for src, dst, route in edge_pairs:
        print(f"  {src} → {dst}  (route={route!r})")

    # START → ParsingAgent
    assert any(
        src == "__START__" and dst == "ParsingAgent"
        for src, dst, _ in edge_pairs
    ), "Expected START → ParsingAgent edge"

    # ParsingAgent → JudgeAgent
    assert any(
        src == "ParsingAgent" and dst == "JudgeAgent"
        for src, dst, _ in edge_pairs
    ), "Expected ParsingAgent → JudgeAgent edge"

    # JudgeAgent → verdict_router
    assert any(
        src == "JudgeAgent" and "verdict" in dst.lower()
        for src, dst, _ in edge_pairs
    ), "Expected JudgeAgent → verdict_router edge"

    # verdict_router → ParsingAgent on retry
    assert any(
        "verdict" in src.lower() and dst == "ParsingAgent" and route == "retry"
        for src, dst, route in edge_pairs
    ), "Expected verdict_router →(retry)→ ParsingAgent loop-back edge"

    print("  ✓ All expected edges found")


def test_full_pipeline_structure():
    """Validate that full_pipeline has parse_judge → pricing → reasoning."""
    wf = full_pipeline
    assert wf.graph is not None
    edge_pairs = [(e.from_node.name, e.to_node.name) for e in wf.graph.edges]
    print("\nFullPipeline edges:")
    for src, dst in edge_pairs:
        print(f"  {src} → {dst}")

    assert any(dst == "ParseJudgeLoop" for _, dst in edge_pairs), \
        "Expected START → ParseJudgeLoop"
    assert any(src == "ParseJudgeLoop" and dst == "PricingAgent" for src, dst in edge_pairs), \
        "Expected ParseJudgeLoop → PricingAgent"
    assert any(src == "PricingAgent" and dst == "cost_engine_bridge" for src, dst in edge_pairs), \
        "Expected PricingAgent → cost_engine_bridge"
    assert any(src == "cost_engine_bridge" and dst == "ReasoningAgent" for src, dst in edge_pairs), \
        "Expected cost_engine_bridge → ReasoningAgent"

    print("  ✓ Full pipeline structure correct")


def test_verdict_router_event_shape():
    """Validate verdict_router returns an Event with route set."""
    from google.adk.events import Event

    class MockState(dict):
        pass

    class MockCtx:
        def __init__(self, verdict_raw):
            self.state = MockState({
                "judge_verdict_raw": json.dumps({
                    "verdict": verdict_raw,
                    "issues": [],
                    "clarifying_question": "How many users?" if verdict_raw == "needs_user" else "",
                }),
                "parse_judge_iterations": 0,
            })

    for verdict in ("pass", "retry", "needs_user"):
        ctx = MockCtx(verdict)
        # verdict_router is wrapped by @node decorator; get the underlying fn
        fn = verdict_router._func if hasattr(verdict_router, "_func") else None
        if fn is None:
            print(f"  ⚠ Cannot access underlying function of verdict_router (ADK wraps it); skipping direct call test for '{verdict}'")
            continue
        result = fn(ctx)
        assert isinstance(result, Event), f"Expected Event, got {type(result)}"
        assert result.actions.route == verdict, f"Expected route={verdict!r}, got {result.actions.route!r}"
        assert ctx.state["judge_verdict"] == verdict
        print(f"  ✓ verdict_router correctly emits route={verdict!r}")


if __name__ == "__main__":
    print("=== ADK Skeleton Smoke Tests ===\n")
    test_graph_edges()
    test_full_pipeline_structure()
    test_verdict_router_event_shape()
    print("\n✓ All skeleton tests passed.")
