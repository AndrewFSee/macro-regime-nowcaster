"""Tests for asset allocation logic and backtester metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.allocation.regime_allocator import RegimeAllocator
from src.allocation.backtester import Backtester, BacktestResult, _max_drawdown, _average_turnover


# ---------------------------------------------------------------------------
# RegimeAllocator tests
# ---------------------------------------------------------------------------


def test_allocator_weights_sum_to_one():
    allocator = RegimeAllocator()
    probs = {"expansion": 0.7, "slowdown": 0.2, "recession": 0.05, "recovery": 0.05}
    allocation = allocator.get_allocation(probs)
    total = sum(allocation.values())
    assert abs(total - 1.0) < 1e-10


def test_allocator_pure_regime_matches_config():
    allocator = RegimeAllocator()
    # 100% recession → should match recession config exactly
    probs = {"expansion": 0.0, "slowdown": 0.0, "recession": 1.0, "recovery": 0.0}
    allocation = allocator.get_allocation(probs)
    expected = allocator.get_regime_weights("recession")
    for asset in expected:
        assert abs(allocation[asset] - expected[asset]) < 1e-10


def test_allocator_blending_math():
    """Test that blending is a proper weighted average."""
    weights = {
        "r1": {"a": 1.0, "b": 0.0},
        "r2": {"a": 0.0, "b": 1.0},
    }
    allocator = RegimeAllocator(custom_weights=weights)
    probs = {"r1": 0.3, "r2": 0.7}
    allocation = allocator.get_allocation(probs)
    assert abs(allocation["a"] - 0.3) < 1e-10
    assert abs(allocation["b"] - 0.7) < 1e-10


def test_allocator_unknown_regime_ignored():
    allocator = RegimeAllocator()
    probs = {"expansion": 0.9, "unknown_regime": 0.1}
    allocation = allocator.get_allocation(probs)
    # Should not raise; unknown regime contributes zero weight
    assert sum(allocation.values()) == pytest.approx(1.0)


def test_allocator_zero_probs_raises():
    allocator = RegimeAllocator()
    with pytest.raises(ValueError):
        allocator.get_allocation({"expansion": 0.0, "recession": 0.0})


# ---------------------------------------------------------------------------
# Backtester tests
# ---------------------------------------------------------------------------


def test_backtester_run_returns_result(regime_probs_df, asset_returns_df):
    bt = Backtester()
    result = bt.run(regime_probs_df, asset_returns_df)
    assert isinstance(result, BacktestResult)


def test_backtester_equity_curve_starts_near_one(regime_probs_df, asset_returns_df):
    bt = Backtester()
    result = bt.run(regime_probs_df, asset_returns_df)
    # First valid value (after shift-1 lag) should be near 1.0
    valid = result.equity_curve.dropna()
    assert abs(valid.iloc[0] - 1.0) < 0.1


def test_backtester_max_drawdown_nonpositive(regime_probs_df, asset_returns_df):
    bt = Backtester()
    result = bt.run(regime_probs_df, asset_returns_df)
    assert result.max_drawdown <= 0.0


def test_max_drawdown_simple():
    curve = pd.Series([1.0, 1.2, 0.9, 1.1])
    dd = _max_drawdown(curve)
    # Peak is 1.2, trough is 0.9 → drawdown = (0.9 - 1.2) / 1.2 = -0.25
    assert abs(dd - (-0.25)) < 1e-10


def test_average_turnover_zero_for_constant():
    weights = pd.DataFrame({"a": [0.5] * 10, "b": [0.5] * 10})
    to = _average_turnover(weights)
    # First row diff is NaN → sum = 0; then all zeros
    assert to == pytest.approx(0.0)


def test_backtester_benchmarks(regime_probs_df, asset_returns_df):
    bt = Backtester()
    benchmarks = bt.run_benchmarks(asset_returns_df)
    assert "buy_and_hold" in benchmarks
    assert "60_40" in benchmarks
    assert isinstance(benchmarks["buy_and_hold"], BacktestResult)
