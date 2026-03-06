"""Tests for date and frequency alignment utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.date_utils import (
    to_business_day_end,
    align_to_monthly,
    get_publication_date,
    business_days_between,
    ragged_edge_mask,
)


# ---------------------------------------------------------------------------
# to_business_day_end
# ---------------------------------------------------------------------------


def test_business_day_end_weekday_unchanged():
    """A Wednesday should stay as-is."""
    dt = pd.Timestamp("2024-01-10")  # Wednesday
    assert to_business_day_end(dt) == dt


def test_business_day_end_saturday_to_friday():
    dt = pd.Timestamp("2024-01-13")  # Saturday
    result = to_business_day_end(dt)
    assert result == pd.Timestamp("2024-01-12")
    assert result.weekday() == 4  # Friday


def test_business_day_end_sunday_to_friday():
    dt = pd.Timestamp("2024-01-14")  # Sunday
    result = to_business_day_end(dt)
    assert result == pd.Timestamp("2024-01-12")
    assert result.weekday() == 4  # Friday


# ---------------------------------------------------------------------------
# align_to_monthly
# ---------------------------------------------------------------------------


def test_align_to_monthly_last():
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    s = pd.Series(range(90), index=dates, dtype=float)
    result = align_to_monthly(s, method="last")
    assert result.index.freqstr == "ME"
    assert len(result) == 3  # Jan, Feb, Mar


def test_align_to_monthly_mean():
    dates = pd.date_range("2024-01-01", periods=31, freq="D")
    s = pd.Series(np.ones(31), index=dates)
    result = align_to_monthly(s, method="mean")
    assert result.iloc[0] == pytest.approx(1.0)


def test_align_to_monthly_unknown_raises():
    s = pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1))
    with pytest.raises(ValueError, match="Unknown method"):
        align_to_monthly(s, method="median")


# ---------------------------------------------------------------------------
# get_publication_date
# ---------------------------------------------------------------------------


def test_publication_date_offset():
    obs = pd.Timestamp("2024-01-31")
    result = get_publication_date(obs, publication_lag_days=35)
    assert result == pd.Timestamp("2024-03-06")


def test_publication_date_zero_lag():
    obs = pd.Timestamp("2024-06-15")
    assert get_publication_date(obs, 0) == obs


# ---------------------------------------------------------------------------
# business_days_between
# ---------------------------------------------------------------------------


def test_business_days_between_same_week():
    # Mon to Fri of the same week = 4 business days (Tue, Wed, Thu, Fri)
    start = pd.Timestamp("2024-01-08")  # Monday
    end = pd.Timestamp("2024-01-12")    # Friday
    assert business_days_between(start, end) == 4


def test_business_days_between_across_weekend():
    start = pd.Timestamp("2024-01-12")  # Friday
    end = pd.Timestamp("2024-01-15")    # Monday
    assert business_days_between(start, end) == 1


# ---------------------------------------------------------------------------
# ragged_edge_mask
# ---------------------------------------------------------------------------


def test_ragged_edge_mask_masks_unreleased_series():
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    df = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]},
        index=dates,
    )
    # As of Feb 20: A has 10-day lag (Jan obs published by Feb 10 → OK),
    # B has 60-day lag (Jan obs published by Mar 31 → only Jan is available if lag is generous)
    as_of = pd.Timestamp("2024-02-20")
    lags = {"A": 10, "B": 60}
    result = ragged_edge_mask(df, lags, as_of_date=as_of)

    # A: Jan lag 10 → pub Feb 10 ≤ Feb 20 ✓; Feb lag 10 → pub Mar 10 > Feb 20 ✗
    assert not np.isnan(result.loc[dates[0], "A"])
    assert np.isnan(result.loc[dates[1], "A"])

    # B: Jan lag 60 → pub Apr 1 > Feb 20 ✗
    assert np.isnan(result.loc[dates[0], "B"])


def test_ragged_edge_mask_no_lags_unchanged():
    dates = pd.date_range("2024-01-31", periods=2, freq="ME")
    df = pd.DataFrame({"X": [1.0, 2.0]}, index=dates)
    result = ragged_edge_mask(df, {}, as_of_date=pd.Timestamp("2099-01-01"))
    pd.testing.assert_frame_equal(result, df)
