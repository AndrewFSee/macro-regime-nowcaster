"""Run the NBER-based regime backtest.

Usage
-----
    python scripts/run_backtest.py                    # full-sample
    python scripts/run_backtest.py --expanding        # expanding-window (slow but no look-ahead)
    python scripts/run_backtest.py --start 2000-01-01 # custom start date
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="NBER Regime Backtest")
    parser.add_argument(
        "--start", default="1990-01-01", help="Evaluation start date"
    )
    parser.add_argument(
        "--end", default=None, help="Evaluation end date (default: today)"
    )
    parser.add_argument(
        "--expanding", action="store_true",
        help="Use expanding-window evaluation (slower, no look-ahead bias)"
    )
    parser.add_argument(
        "--n-factors", type=int, default=4, help="Number of DFM factors"
    )
    parser.add_argument(
        "--n-regimes", type=int, default=4, help="Number of regimes"
    )
    parser.add_argument(
        "--step", type=int, default=3,
        help="Step size in months for expanding-window mode"
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use ensemble recession detection (RSM + probit + CFNAI)"
    )
    args = parser.parse_args()

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.error("FRED_API_KEY not set.  Add it to .env or environment.")
        sys.exit(1)

    from src.data.fred_client import FREDClient
    from src.data.data_pipeline import DataPipeline
    from src.models.regime_backtest import RegimeBacktester, cfnai_baseline_backtest

    logger.info("Initialising FRED client and data pipeline...")
    client = FREDClient(api_key=api_key, cache_dir="data/cache")
    pipeline = DataPipeline(
        fred_client=client,
        start_date="1980-01-01",
        series_config_path="config/fred_series.yaml",
    )

    regime_labels = ["expansion", "recession"]
    factor_names = ["real_activity", "labor_market", "inflation", "financial_conditions"]

    bt = RegimeBacktester(
        pipeline=pipeline,
        n_factors=args.n_factors,
        n_regimes=args.n_regimes,
        factor_names=factor_names,
        regime_labels=regime_labels,
        recession_labels=["recession"],
        use_ensemble=args.ensemble,
    )

    if args.expanding:
        logger.info(
            f"Running expanding-window backtest "
            f"(start_eval={args.start}, step={args.step}mo)..."
        )
        report = bt.run_expanding(
            start_eval=args.start,
            end=args.end,
            step_months=args.step,
        )
    else:
        logger.info(f"Running full-sample backtest (start={args.start})...")
        report = bt.run(start=args.start, end=args.end)

    print()
    print(report.summary())

    # --- CFNAI baseline comparison ---
    try:
        logger.info("Running CFNAI baseline comparison...")
        panel = pipeline.run(end_date=args.end)
        panel = panel.dropna(how="all")
        baseline = cfnai_baseline_backtest(
            panel=panel, start=args.start, end=args.end,
        )
        print()
        print("=" * 50)
        print("  CFNAI BASELINE (upper-bound reference)")
        print("=" * 50)
        print(f"  Accuracy       : {baseline.accuracy:.1%}")
        print(f"  Precision      : {baseline.precision_recession:.1%}")
        print(f"  Recall         : {baseline.recall_recession:.1%}")
        print(f"  F1 score       : {baseline.f1_recession:.3f}")
        print(f"  Detection lag  : {baseline.detection_lag_months:.1f} months")
        print(f"  False alarm    : {baseline.false_alarm_rate:.1%}")
        print("=" * 50)
    except Exception as exc:
        logger.warning(f"CFNAI baseline failed: {exc}")

    print()

    # Save detailed history
    out_path = Path("data") / "backtest_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.regime_history.to_csv(out_path)
    logger.info(f"Detailed history saved to {out_path}")


if __name__ == "__main__":
    main()
