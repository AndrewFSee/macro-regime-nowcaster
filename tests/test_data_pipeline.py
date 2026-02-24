"""Tests for data ingestion, transformations, and ragged-edge handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.transformations import (
    log_difference,
    first_difference,
    percent_change,
    standardize,
    apply_transform,
    apply_all_transforms,
)
from src.data.data_pipeline import DataPipeline


# ---------------------------------------------------------------------------
# Transformation tests
# ---------------------------------------------------------------------------


def test_log_difference_returns_series():
    s = pd.Series([100.0, 110.0, 121.0])
    result = log_difference(s)
    assert isinstance(result, pd.Series)
    assert np.isnan(result.iloc[0])
    # ln(110/100) ≈ 0.0953
    assert abs(result.iloc[1] - np.log(1.1)) < 1e-10


def test_first_difference():
    s = pd.Series([1.0, 3.0, 6.0, 10.0])
    result = first_difference(s)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(4.0)


def test_percent_change():
    s = pd.Series([100.0, 110.0, 121.0])
    result = percent_change(s)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.10)


def test_standardize_zero_mean_unit_variance():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = standardize(s)
    assert abs(result.mean()) < 1e-10
    assert abs(result.std(ddof=1) - 1.0) < 1e-10


def test_apply_transform_unknown_raises():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Unknown transform"):
        apply_transform(s, "gibberish")


def test_apply_all_transforms_handles_nans():
    dates = pd.date_range("2020-01-01", periods=10, freq="ME")
    df = pd.DataFrame(
        {"a": [1.0] * 10, "b": [2.0] * 5 + [np.nan] * 5},
        index=dates,
    )
    transform_map = {"a": "diff", "b": "log_diff"}
    result = apply_all_transforms(df, transform_map, standardize_after=False)
    # Column 'a' should be all zeros except first NaN
    assert np.isnan(result["a"].iloc[0])
    assert (result["a"].iloc[1:].abs() < 1e-10).all()


def test_pipeline_get_series_list_non_empty(tmp_path):
    """DataPipeline loads series from config/fred_series.yaml."""
    # Use the real config file from the project
    pipeline = DataPipeline.__new__(DataPipeline)
    series_cfg = pipeline._load_series_config("config/fred_series.yaml")
    assert len(series_cfg) > 0
    codes = [s["code"] for s in series_cfg]
    assert "PAYEMS" in codes
    assert "GDPC1" in codes
