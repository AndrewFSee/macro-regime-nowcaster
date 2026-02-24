"""Integration tests for the Nowcaster pipeline with mock data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.models.nowcaster import Nowcaster, NowcastResult


# ---------------------------------------------------------------------------
# NowcastResult tests
# ---------------------------------------------------------------------------


def test_nowcast_result_to_dict(sample_nowcast_result):
    d = sample_nowcast_result.to_dict()
    assert isinstance(d, dict)
    assert "gdp_nowcast" in d
    assert "current_regime" in d
    assert "regime_probabilities" in d


def test_nowcast_result_to_json(sample_nowcast_result):
    import json
    js = sample_nowcast_result.to_json()
    parsed = json.loads(js)
    assert parsed["current_regime"] == "expansion"


# ---------------------------------------------------------------------------
# Nowcaster integration tests (with mocked pipeline)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline(synthetic_panel):
    pipeline = MagicMock()
    pipeline.run.return_value = synthetic_panel
    return pipeline


def test_nowcaster_run_returns_result(mock_pipeline):
    nowcaster = Nowcaster(pipeline=mock_pipeline, n_factors=2, n_regimes=2)
    result = nowcaster.run()
    assert isinstance(result, NowcastResult)


def test_nowcaster_result_regime_probs_sum_to_one(mock_pipeline):
    nowcaster = Nowcaster(pipeline=mock_pipeline, n_factors=2, n_regimes=2)
    result = nowcaster.run()
    total = sum(result.regime_probabilities.values())
    assert abs(total - 1.0) < 1e-6


def test_nowcaster_result_gdp_in_reasonable_range(mock_pipeline):
    nowcaster = Nowcaster(pipeline=mock_pipeline, n_factors=2, n_regimes=2)
    result = nowcaster.run()
    assert -20.0 < result.gdp_nowcast < 20.0


def test_nowcaster_get_summary_string(mock_pipeline):
    nowcaster = Nowcaster(pipeline=mock_pipeline, n_factors=2, n_regimes=2)
    nowcaster.run()
    summary = nowcaster.get_summary()
    assert isinstance(summary, str)
    assert "Regime" in summary


def test_nowcaster_get_summary_before_run():
    pipeline = MagicMock()
    nowcaster = Nowcaster(pipeline=pipeline, n_factors=2, n_regimes=2)
    summary = nowcaster.get_summary()
    assert "No nowcast" in summary
