"""CLI script to estimate model parameters on historical data.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --n-factors 4 --n-regimes 4
    python scripts/train_model.py --output-dir models/
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
        description="Estimate Dynamic Factor Model and Regime Switching parameters"
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--n-factors",
        type=int,
        default=None,
        help="Number of latent factors (overrides config)",
    )
    parser.add_argument(
        "--n-regimes",
        type=int,
        default=None,
        help="Number of regimes (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/",
        help="Directory to save fitted model parameters",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    setup_logging(level=args.log_level)

    import os
    import pickle
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
    n_factors = args.n_factors or model_cfg.get("n_factors", 4)
    n_regimes = args.n_regimes or model_cfg.get("n_regimes", 4)
    regime_labels = model_cfg.get("regime_labels", None)
    factor_names = model_cfg.get("factor_names", None)
    start_date = settings.get("data", {}).get("start_date", "1980-01-01")
    cache_dir = settings.get("data", {}).get("cache_dir", "data/cache")

    from src.data.fred_client import FREDClient
    from src.data.data_pipeline import DataPipeline
    from src.models.dynamic_factor_model import DynamicFactorModel
    from src.models.regime_switching import RegimeSwitchingModel

    client = FREDClient(api_key=api_key, cache_dir=cache_dir)
    pipeline = DataPipeline(fred_client=client, start_date=start_date)

    logger.info("Fetching data for model training…")
    panel = pipeline.run(save_vintage=False)
    logger.info(f"Panel shape: {panel.shape}")

    logger.info(f"Fitting DynamicFactorModel (K={n_factors})…")
    dfm = DynamicFactorModel(
        n_factors=n_factors,
        factor_names=factor_names,
    )
    dfm.fit(panel.dropna(how="all"))
    logger.info("DFM fitted successfully")

    factors = dfm.factors_
    logger.info(f"Fitting RegimeSwitchingModel (K={n_regimes})…")
    rsm = RegimeSwitchingModel(n_regimes=n_regimes, regime_labels=regime_labels)
    rsm.fit(factors)
    logger.info("Regime model fitted successfully")

    regime_info = rsm.get_current_regime()
    logger.info(f"Current regime: {regime_info['regime'].upper()}")
    for label, prob in regime_info["probabilities"].items():
        logger.info(f"  {label}: {prob:.1%}")

    # Save model artifacts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "dfm.pkl").open("wb") as fh:
        pickle.dump(dfm, fh)
    with (output_dir / "rsm.pkl").open("wb") as fh:
        pickle.dump(rsm, fh)

    logger.info(f"Model artifacts saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
