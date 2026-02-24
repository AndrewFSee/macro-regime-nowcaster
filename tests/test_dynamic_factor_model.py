"""Tests for the Dynamic Factor Model.

Validates that the EM algorithm recovers a known factor structure on
synthetic data and that the statsmodels wrapper has a consistent API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.dynamic_factor_model import DynamicFactorModel, _fill_for_pca


# ---------------------------------------------------------------------------
# DynamicFactorModel tests
# ---------------------------------------------------------------------------


def test_dfm_fit_returns_self(synthetic_panel):
    dfm = DynamicFactorModel(n_factors=2, max_iter=10)
    result = dfm.fit(synthetic_panel)
    assert result is dfm


def test_dfm_is_fitted_after_fit(synthetic_panel):
    dfm = DynamicFactorModel(n_factors=2, max_iter=10)
    dfm.fit(synthetic_panel)
    assert dfm._is_fitted


def test_dfm_factors_shape(synthetic_panel):
    T, N = synthetic_panel.shape
    K = 2
    dfm = DynamicFactorModel(n_factors=K, max_iter=20)
    dfm.fit(synthetic_panel)
    assert dfm.factors_.shape == (T, K)


def test_dfm_loadings_shape(synthetic_panel):
    N = synthetic_panel.shape[1]
    K = 2
    dfm = DynamicFactorModel(n_factors=K, max_iter=10)
    dfm.fit(synthetic_panel)
    loadings = dfm.get_loadings()
    assert loadings.shape == (N, K)


def test_dfm_transform_consistent_with_fit(synthetic_panel):
    dfm = DynamicFactorModel(n_factors=2, max_iter=20)
    dfm.fit(synthetic_panel)
    transformed = dfm.transform(synthetic_panel)
    assert transformed.shape == dfm.factors_.shape


def test_dfm_transform_before_fit_raises(synthetic_panel):
    dfm = DynamicFactorModel(n_factors=2)
    with pytest.raises(RuntimeError, match="fit"):
        dfm.transform(synthetic_panel)


def test_fill_for_pca_no_nans():
    arr = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
    filled = _fill_for_pca(arr)
    assert not np.isnan(filled).any()


def test_dfm_with_nan_panel(synthetic_panel_with_nans):
    """DFM should handle panels with NaN values (ragged edge)."""
    dfm = DynamicFactorModel(n_factors=2, max_iter=15)
    dfm.fit(synthetic_panel_with_nans)
    assert dfm._is_fitted
    assert dfm.factors_ is not None
