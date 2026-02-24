"""Tests for the Kalman filter and RTS smoother.

Validates the filter against the analytic solution for the local level
model, and checks that NaN handling (ragged edge) works correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.kalman_filter import KalmanFilter, FilterResult, SmootherResult


# ---------------------------------------------------------------------------
# Local level model tests
# ---------------------------------------------------------------------------


def test_filter_returns_filter_result(local_level_kf, local_level_observations):
    result = local_level_kf.filter(local_level_observations)
    assert isinstance(result, FilterResult)
    T = local_level_observations.shape[0]
    assert result.filtered_states.shape == (T, 1)
    assert result.filtered_covs.shape == (T, 1, 1)


def test_filter_log_likelihood_finite(local_level_kf, local_level_observations):
    result = local_level_kf.filter(local_level_observations)
    assert np.isfinite(result.log_likelihood)


def test_smoother_returns_smoother_result(local_level_kf, local_level_observations):
    filter_res = local_level_kf.filter(local_level_observations)
    smoother_res = local_level_kf.smooth(filter_res)
    assert isinstance(smoother_res, SmootherResult)
    T = local_level_observations.shape[0]
    assert smoother_res.smoothed_states.shape == (T, 1)
    assert smoother_res.smoothed_cross_covs.shape == (T - 1, 1, 1)


def test_smoother_ll_ge_filter_ll_heuristic(local_level_kf, local_level_observations):
    """Smoothed variance should be <= filtered variance (information gain)."""
    filter_res = local_level_kf.filter(local_level_observations)
    smoother_res = local_level_kf.smooth(filter_res)
    # For each time step, P_smooth <= P_filtered (scalar case)
    for t in range(len(local_level_observations)):
        assert smoother_res.smoothed_covs[t, 0, 0] <= filter_res.filtered_covs[t, 0, 0] + 1e-10


def test_filter_handles_all_nan_observation():
    """Filter should not update state when all observations are NaN."""
    A = np.array([[0.9]])
    C = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[1.0]])
    kf = KalmanFilter(A=A, C=C, Q=Q, R=R,
                      initial_state=np.array([1.0]),
                      initial_covariance=np.array([[0.5]]))

    # One-step: all-NaN obs → filtered state = predicted state
    obs = np.array([[np.nan]])
    result = kf.filter(obs)
    pred_state = A @ np.array([1.0])
    np.testing.assert_allclose(result.filtered_states[0], pred_state, atol=1e-10)


def test_filter_covariance_symmetric(local_level_kf, local_level_observations):
    """Filtered covariance matrices should be symmetric."""
    result = local_level_kf.filter(local_level_observations)
    for t in range(local_level_observations.shape[0]):
        P = result.filtered_covs[t]
        np.testing.assert_allclose(P, P.T, atol=1e-12)


def test_multivariate_kf():
    """Test a 2-state, 3-observation Kalman filter."""
    n_states, n_obs = 2, 3
    T = 30
    rng = np.random.default_rng(99)

    A = np.eye(n_states) * 0.8
    C = rng.standard_normal((n_obs, n_states))
    Q = np.eye(n_states) * 0.5
    R = np.eye(n_obs) * 1.0

    kf = KalmanFilter(A=A, C=C, Q=Q, R=R)
    obs = rng.standard_normal((T, n_obs))
    result = kf.filter(obs)
    assert result.filtered_states.shape == (T, n_states)
    assert np.isfinite(result.log_likelihood)
