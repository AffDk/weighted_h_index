"""Tests for cost/token tracker."""

from src.llm.tracker import CostTracker


def test_tracker_record_and_totals():
    tracker = CostTracker()
    tracker.record("qwen", "qwen3.7-flash", 1000, 500)
    tracker.record("deepseek", "deepseek-chat", 2000, 1000)
    totals = tracker.totals()
    assert totals["calls"] == 2
    assert totals["input_tokens"] == 3000
    assert totals["output_tokens"] == 1500
    assert totals["total_tokens"] == 4500
    assert totals["estimated_cost_usd"] >= 0


def test_tracker_unknown_model_zero_cost():
    tracker = CostTracker()
    tracker.record("unknown", "unknown-model", 100, 100)
    totals = tracker.totals()
    assert totals["estimated_cost_usd"] == 0.0
