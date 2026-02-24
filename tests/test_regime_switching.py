"""Tests for the Markov-Switching regime model.

Checks that the custom EM Hamilton filter runs without error and that
the API returns consistently shaped outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.regime_switching import RegimeSwitchingModel, _gaussian_density


# ---------------------------------------------------------------------------
# RegimeSwitchingModel tests
# ---------------------------------------------------------------------------


def test_rsm_fit_returns_self(synthetic_regime_series):
    dfm_like = pd.DataFrame(
        {"f1": synthetic_regime_series.values,
         "f2": synthetic_regime_series.values * 0.5},
        index=synthetic_regime_series.index,
    )
    rsm = RegimeSwitchingModel(n_regimes=2, use_statsmodels=False)
    result = rsm.fit(dfm_like)
    assert result is rsm


def test_rsm_regime_probs_shape(synthetic_regime_series):
    T = len(synthetic_regime_series)
    dfm_like = pd.DataFrame({"f1": synthetic_regime_series}, index=synthetic_regime_series.index)
    rsm = RegimeSwitchingModel(n_regimes=2, use_statsmodels=False)
    rsm.fit(dfm_like)
    probs = rsm.get_regime_probabilities()
    assert probs.shape == (T, 2)


def test_rsm_regime_probs_sum_to_one(synthetic_regime_series):
    dfm_like = pd.DataFrame({"f1": synthetic_regime_series}, index=synthetic_regime_series.index)
    rsm = RegimeSwitchingModel(n_regimes=2, use_statsmodels=False)
    rsm.fit(dfm_like)
    probs = rsm.get_regime_probabilities()
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_rsm_current_regime_returns_dict(synthetic_regime_series):
    dfm_like = pd.DataFrame({"f1": synthetic_regime_series}, index=synthetic_regime_series.index)
    rsm = RegimeSwitchingModel(n_regimes=2, use_statsmodels=False)
    rsm.fit(dfm_like)
    info = rsm.get_current_regime()
    assert "regime" in info
    assert "probabilities" in info
    assert info["regime"] in rsm.regime_labels


def test_rsm_transition_matrix_shape(synthetic_regime_series):
    dfm_like = pd.DataFrame({"f1": synthetic_regime_series}, index=synthetic_regime_series.index)
    n = 2
    rsm = RegimeSwitchingModel(n_regimes=n, use_statsmodels=False)
    rsm.fit(dfm_like)
    tm = rsm.get_transition_matrix()
    assert tm.shape == (n, n)


def test_gaussian_density_positive():
    y = 0.0
    means = np.array([0.0, 1.0, -1.0])
    variances = np.array([1.0, 1.0, 1.0])
    dens = _gaussian_density(y, means, variances)
    assert (dens > 0).all()


def test_rsm_before_fit_raises():
    rsm = RegimeSwitchingModel(n_regimes=2)
    with pytest.raises(RuntimeError, match="fit"):
        rsm.get_regime_probabilities()
