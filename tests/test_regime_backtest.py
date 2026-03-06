"""Tests for the NBER regime backtest framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.regime_backtest import (
    RegimeBacktester,
    RegimeBacktestResult,
    get_nber_recession_indicator,
)


# ---------------------------------------------------------------------------
# NBER indicator tests
# ---------------------------------------------------------------------------


def test_nber_indicator_returns_series():
    nber = get_nber_recession_indicator(start="2005-01-01", end="2015-12-31")
    assert isinstance(nber, pd.Series)
    assert nber.dtype == int


def test_nber_indicator_has_gfc_recession():
    """The 2007-12 to 2009-06 recession should be marked."""
    nber = get_nber_recession_indicator(start="2007-01-01", end="2010-12-31")
    # Some months in 2008-2009 must be recession=1
    assert nber.sum() > 0
    # Dec 2007 should be in recession
    dec07 = nber.loc["2007-12":"2008-01"]
    assert dec07.sum() > 0


def test_nber_indicator_mostly_expansion():
    nber = get_nber_recession_indicator(start="2010-01-01", end="2019-12-31")
    # The 2010s had no recessions → all zeros
    assert nber.sum() == 0


def test_nber_indicator_covid_recession():
    nber = get_nber_recession_indicator(start="2020-01-01", end="2020-12-31")
    # Feb-Apr 2020 recession
    assert nber.sum() >= 2


# ---------------------------------------------------------------------------
# RegimeBacktestResult tests
# ---------------------------------------------------------------------------


def test_backtest_result_summary():
    result = RegimeBacktestResult(
        accuracy=0.85,
        precision_recession=0.70,
        recall_recession=0.60,
        f1_recession=0.647,
        confusion=np.array([[90, 5], [8, 12]]),
        regime_history=pd.DataFrame(),
        n_months=115,
        detection_lag_months=2.0,
        false_alarm_rate=0.05,
    )
    summary = result.summary()
    assert "85.0%" in summary
    assert "NBER VALIDATION" in summary


# ---------------------------------------------------------------------------
# RegimeBacktester — full-sample with synthetic data
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_backtest_panel():
    """Create a synthetic panel that mimics recession/expansion dynamics.

    Uses varied factor loadings so the DFM can extract a clear signal.
    """
    rng = np.random.default_rng(42)
    T = 240  # 20 years monthly
    dates = pd.date_range("2000-01-31", periods=T, freq="ME")

    # Build NBER-aligned synthetic data: negative during recessions
    nber = get_nber_recession_indicator(start="2000-01-01", end="2020-12-31")
    nber_aligned = nber.reindex(dates, method="ffill").fillna(0).astype(int)

    # Create a latent factor that tracks the business cycle
    cycle_factor = np.where(nber_aligned.values == 1, -3.0, 1.0)
    # Add AR(1) dynamics
    for t in range(1, T):
        cycle_factor[t] = 0.7 * cycle_factor[t] + 0.3 * cycle_factor[t - 1]

    # 8 series with varied loadings on the cycle factor
    N = 8
    loadings = rng.uniform(0.5, 2.0, size=N)
    panel = pd.DataFrame(index=dates)
    for i in range(N):
        noise = rng.standard_normal(T) * 0.3
        panel[f"series_{i}"] = loadings[i] * cycle_factor + noise

    return panel, nber_aligned


def test_backtest_full_sample(synthetic_backtest_panel):
    """Full-sample backtest on synthetic data should achieve decent accuracy."""
    panel, nber = synthetic_backtest_panel

    bt = RegimeBacktester(
        n_factors=1,
        n_regimes=2,
        regime_labels=["expansion", "recession"],
        recession_labels=["recession"],
    )
    result = bt.run(
        start="2000-01-01",
        end="2019-12-31",
        panel=panel,
        nber=nber,
    )
    assert isinstance(result, RegimeBacktestResult)
    assert result.n_months > 50
    assert result.accuracy > 0.5  # should be well above chance


def test_backtest_confusion_matrix_shape(synthetic_backtest_panel):
    panel, nber = synthetic_backtest_panel
    bt = RegimeBacktester(
        n_factors=1, n_regimes=2,
        regime_labels=["expansion", "recession"],
        recession_labels=["recession"],
    )
    result = bt.run(
        start="2000-01-01", end="2019-12-31",
        panel=panel, nber=nber,
    )
    assert result.confusion.shape == (2, 2)
    assert result.confusion.sum() == result.n_months


def test_backtest_metrics_in_range(synthetic_backtest_panel):
    panel, nber = synthetic_backtest_panel
    bt = RegimeBacktester(
        n_factors=1, n_regimes=2,
        regime_labels=["expansion", "recession"],
        recession_labels=["recession"],
    )
    result = bt.run(
        start="2000-01-01", end="2019-12-31",
        panel=panel, nber=nber,
    )
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision_recession <= 1.0
    assert 0.0 <= result.recall_recession <= 1.0
    assert 0.0 <= result.f1_recession <= 1.0
    assert 0.0 <= result.false_alarm_rate <= 1.0


def test_backtest_regime_history_columns(synthetic_backtest_panel):
    panel, nber = synthetic_backtest_panel
    bt = RegimeBacktester(
        n_factors=1, n_regimes=2,
        regime_labels=["expansion", "recession"],
        recession_labels=["recession"],
    )
    result = bt.run(
        start="2000-01-01", end="2019-12-31",
        panel=panel, nber=nber,
    )
    hist = result.regime_history
    assert "nber_recession" in hist.columns
    assert "model_pred" in hist.columns
    assert "model_recession_prob" in hist.columns
