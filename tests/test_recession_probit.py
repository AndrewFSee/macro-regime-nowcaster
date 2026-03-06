"""Tests for the probit recession probability model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.recession_probit import RecessionProbit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probit_data():
    """Synthetic labelled data: recession when composite < -0.5."""
    rng = np.random.default_rng(123)
    T = 300
    dates = pd.date_range("2000-01-31", periods=T, freq="ME")

    # Two features that dip during "recessions"
    cycle = np.sin(np.linspace(0, 6 * np.pi, T))  # 3 full cycles
    x1 = cycle + rng.normal(0, 0.3, T)
    x2 = 0.6 * cycle + rng.normal(0, 0.3, T)
    X = pd.DataFrame({"activity": x1, "credit": x2}, index=dates)

    # Label: recession when cycle < -0.5
    y = pd.Series((cycle < -0.5).astype(int), index=dates, name="recession")
    return X, y


# ---------------------------------------------------------------------------
# Construction / API
# ---------------------------------------------------------------------------


def test_probit_init():
    m = RecessionProbit(add_lags=2, regularization=0.05)
    assert m.add_lags == 2
    assert m.regularization == 0.05
    assert not m._is_fitted


def test_probit_fit_returns_self(probit_data):
    X, y = probit_data
    m = RecessionProbit()
    result = m.fit(X, y)
    assert result is m
    assert m._is_fitted


def test_probit_predict_proba_shape(probit_data):
    X, y = probit_data
    m = RecessionProbit().fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X),)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_probit_predict_binary(probit_data):
    X, y = probit_data
    m = RecessionProbit().fit(X, y)
    pred = m.predict(X, threshold=0.5)
    assert set(np.unique(pred)).issubset({0, 1})


def test_probit_accuracy_above_chance(probit_data):
    """With a clean cyclic signal, accuracy should be well above 50 %."""
    X, y = probit_data
    m = RecessionProbit().fit(X, y)
    pred = m.predict(X, threshold=0.5)
    acc = float((pred == y.values).mean())
    assert acc > 0.70


def test_probit_coefficients_dict(probit_data):
    X, y = probit_data
    m = RecessionProbit(add_lags=1).fit(X, y)
    coefs = m.get_coefficients()
    assert isinstance(coefs, dict)
    # 2 features + 2 lagged features + 1 intercept = 5
    assert len(coefs) == 5
    assert "intercept" in coefs


def test_probit_before_fit_raises():
    m = RecessionProbit()
    with pytest.raises(RuntimeError, match="fit"):
        m.predict_proba(np.zeros((5, 2)))


def test_probit_handles_nan_in_predict(probit_data):
    X, y = probit_data
    m = RecessionProbit(add_lags=0).fit(X, y)
    X_nan = X.copy()
    X_nan.iloc[0, 0] = np.nan  # inject NaN
    proba = m.predict_proba(X_nan)
    assert proba[0] == 0.5  # NaN row → 0.5
    assert 0 < proba[1] < 1  # other rows normal


def test_probit_no_lags_mode(probit_data):
    X, y = probit_data
    m = RecessionProbit(add_lags=0).fit(X, y)
    coefs = m.get_coefficients()
    # 2 features + 1 intercept = 3
    assert len(coefs) == 3


def test_probit_numpy_input():
    """Probit should accept plain numpy arrays."""
    rng = np.random.default_rng(42)
    T = 100
    X = rng.standard_normal((T, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    m = RecessionProbit(add_lags=0).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (T,)
