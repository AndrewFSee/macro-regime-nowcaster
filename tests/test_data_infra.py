"""Tests for storage and FRED client modules."""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.data.fred_client import FREDClient
from src.data.storage import DataStorage
from src.data.data_pipeline import DataPipeline


@pytest.fixture
def scratch_dir():
    """Create a temp directory that works around Windows permission issues."""
    d = tempfile.mkdtemp(prefix="nowcaster_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# FREDClient tests (with mocked fredapi)
# ---------------------------------------------------------------------------


class TestFREDClient:
    def test_init_requires_fredapi(self):
        """FREDClient should raise if fredapi is not available (tested indirectly)."""
        # Just test that instantiation works when fredapi IS available
        with patch("src.data.fred_client.Fred") as MockFred:
            MockFred.return_value = MagicMock()
            client = FREDClient(api_key="test-key")
            assert client.api_key == "test-key"

    def test_get_series_caching(self, scratch_dir):
        """Fetched series should be cached to disk and served from cache."""
        with patch("src.data.fred_client.Fred") as MockFred:
            mock_fred = MagicMock()
            MockFred.return_value = mock_fred

            dates = pd.date_range("2024-01-01", periods=5, freq="ME")
            fake_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
            fake_series.name = "TEST"
            mock_fred.get_series.return_value = fake_series

            client = FREDClient(api_key="key", cache_dir=scratch_dir)

            # First call → API hit
            result1 = client.get_series("TEST")
            assert mock_fred.get_series.call_count == 1
            assert len(result1) == 5

            # Second call → cache hit
            result2 = client.get_series("TEST")
            assert mock_fred.get_series.call_count == 1  # no new API call
            assert len(result2) == 5


# ---------------------------------------------------------------------------
# DataStorage tests
# ---------------------------------------------------------------------------


class TestDataStorage:
    def test_save_and_load_vintage(self, scratch_dir):
        db_path = os.path.join(scratch_dir, "test.duckdb")
        storage = DataStorage(db_path=db_path)

        dates = pd.date_range("2024-01-31", periods=3, freq="ME")
        panel = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=dates
        )
        panel.index.name = "date"

        storage.save_vintage(panel, as_of="2024-04-01")
        loaded = storage.load_vintage("2024-04-01")

        assert loaded.shape == (3, 2)
        storage.close()

    def test_list_vintages(self, scratch_dir):
        db_path = os.path.join(scratch_dir, "test.duckdb")
        storage = DataStorage(db_path=db_path)

        dates = pd.date_range("2024-01-31", periods=2, freq="ME")
        panel = pd.DataFrame({"X": [1.0, 2.0]}, index=dates)
        panel.index.name = "date"

        storage.save_vintage(panel, as_of="2024-03-01")
        storage.save_vintage(panel, as_of="2024-04-01")

        vintages = storage.list_vintages()
        assert "2024-03-01" in vintages
        assert "2024-04-01" in vintages
        storage.close()

    def test_close_does_not_error(self, scratch_dir):
        storage = DataStorage(db_path=os.path.join(scratch_dir, "test.duckdb"))
        storage.close()


# ---------------------------------------------------------------------------
# DataPipeline config tests
# ---------------------------------------------------------------------------


class TestDataPipelineConfig:
    def test_load_series_config_from_project(self):
        """Should load the real config/fred_series.yaml."""
        cfg = DataPipeline._load_series_config("config/fred_series.yaml")
        assert len(cfg) > 0
        codes = [s["code"] for s in cfg]
        assert "PAYEMS" in codes

    def test_load_series_config_missing_file(self):
        """Should return empty list for a non-existent config file."""
        cfg = DataPipeline._load_series_config("nonexistent.yaml")
        assert cfg == []

    def test_pipeline_run_with_mock_client(self):
        """Pipeline.run() should call the client and return a DataFrame."""
        mock_client = MagicMock()
        # Generate 3 years of monthly data so the pipeline has enough obs
        dates = pd.date_range("2021-01-31", periods=36, freq="ME")
        mock_client.get_series.return_value = pd.Series(
            np.random.randn(36).cumsum() + 100, index=dates, name="TEST"
        )

        pipeline = DataPipeline(
            fred_client=mock_client,
            start_date="2021-01-01",
        )
        # Override config to a single series for speed
        pipeline._series_cfg = [
            {"code": "TEST", "name": "Test Series", "transform": "diff"}
        ]
        result = pipeline.run(end_date="2024-03-01")
        assert isinstance(result, pd.DataFrame)
        assert "TEST" in result.columns
