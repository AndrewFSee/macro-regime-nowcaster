"""CLI script to run a single nowcast and print results.

Usage:
    python scripts/run_nowcast.py
    python scripts/run_nowcast.py --output-format json
    python scripts/run_nowcast.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from loguru import logger

from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single macro regime nowcast and display results"
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format (text or JSON). Default: text",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Nowcast as-of date (ISO-8601). Default: latest available",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
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
        logger.error("FRED_API_KEY not set.")
        return 1

    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open() as fh:
            settings = yaml.safe_load(fh)
    else:
        settings = {}

    model_cfg = settings.get("model", {})
    n_factors = model_cfg.get("n_factors", 4)
    n_regimes = model_cfg.get("n_regimes", 4)
    regime_labels = model_cfg.get("regime_labels", None)
    factor_names = model_cfg.get("factor_names", None)
    start_date = settings.get("data", {}).get("start_date", "1980-01-01")
    cache_dir = settings.get("data", {}).get("cache_dir", "data/cache")

    from src.data.fred_client import FREDClient
    from src.data.data_pipeline import DataPipeline
    from src.models.nowcaster import Nowcaster

    client = FREDClient(api_key=api_key, cache_dir=cache_dir)
    pipeline = DataPipeline(fred_client=client, start_date=start_date)
    nowcaster = Nowcaster(
        pipeline=pipeline,
        n_factors=n_factors,
        n_regimes=n_regimes,
        regime_labels=regime_labels,
        factor_names=factor_names,
    )

    logger.info("Running nowcast…")
    result = nowcaster.run(end_date=args.end_date)

    if args.output_format == "json":
        print(result.to_json())
    else:
        print(nowcaster.get_summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
