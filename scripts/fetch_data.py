"""CLI script to fetch and update all FRED data.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --start-date 1990-01-01 --end-date 2024-12-31
    python scripts/fetch_data.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from loguru import logger

from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and update all configured FRED economic series"
    )
    parser.add_argument(
        "--start-date",
        default="1980-01-01",
        help="Earliest observation date (ISO-8601). Default: 1980-01-01",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Latest observation date (ISO-8601). Default: today",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings.yaml. Default: config/settings.yaml",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing cache and re-fetch all data",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    setup_logging(level=args.log_level)

    import os
    import yaml

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        logger.error("FRED_API_KEY not set. Copy .env.example → .env and add your key.")
        return 1

    # Load settings
    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open() as fh:
            settings = yaml.safe_load(fh)
    else:
        settings = {}

    cache_dir = settings.get("data", {}).get("cache_dir", "data/cache")
    db_path = settings.get("data", {}).get("database_path", "data/nowcaster.duckdb")

    from src.data.fred_client import FREDClient
    from src.data.data_pipeline import DataPipeline
    from src.data.storage import DataStorage

    logger.info("Initialising FRED client")
    client = FREDClient(api_key=api_key, cache_dir=cache_dir)

    try:
        storage = DataStorage(db_path=db_path)
    except Exception as exc:
        logger.warning(f"Could not initialise storage: {exc}. Proceeding without persistence.")
        storage = None

    pipeline = DataPipeline(
        fred_client=client,
        start_date=args.start_date,
        storage=storage,
    )

    logger.info(f"Fetching data from {args.start_date} to {args.end_date or 'today'}")
    panel = pipeline.run(end_date=args.end_date, save_vintage=(storage is not None))

    logger.info(f"Done! Panel shape: {panel.shape}")
    logger.info(f"Date range: {panel.index[0].date()} — {panel.index[-1].date()}")
    logger.info(f"Series: {list(panel.columns)[:5]} … ({panel.shape[1]} total)")

    if storage:
        storage.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
