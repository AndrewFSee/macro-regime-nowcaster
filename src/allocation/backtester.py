"""Historical backtest engine for regime-based portfolio strategies.

Given a time series of regime probabilities and asset returns, computes
portfolio weights at each period and evaluates portfolio performance
vs. benchmark strategies (buy-and-hold equities, 60/40).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.allocation.regime_allocator import RegimeAllocator


@dataclass
class BacktestResult:
    """Aggregated performance metrics from a backtest run.

    Attributes:
        total_return: Cumulative return over the full period (decimal).
        annual_return: Annualised return (decimal).
        sharpe_ratio: Annualised Sharpe ratio (assuming zero risk-free rate).
        max_drawdown: Maximum peak-to-trough drawdown (decimal, negative).
        turnover: Average monthly portfolio turnover (decimal).
        equity_curve: Series of cumulative portfolio value (starts at 1.0).
        weights_history: DataFrame of portfolio weights over time.
    """

    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    equity_curve: pd.Series
    weights_history: pd.DataFrame

    def summary(self) -> str:
        """Return a human-readable performance summary."""
        return (
            f"Total Return  : {self.total_return:.1%}\n"
            f"Annual Return : {self.annual_return:.1%}\n"
            f"Sharpe Ratio  : {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown  : {self.max_drawdown:.1%}\n"
            f"Avg Turnover  : {self.turnover:.1%}"
        )


class Backtester:
    """Regime-based portfolio backtester.

    Args:
        allocator: :class:`RegimeAllocator` instance that maps regime
            probabilities to portfolio weights.
        rebalance_freq: Rebalancing frequency (``"M"`` = monthly, etc.).
    """

    def __init__(
        self,
        allocator: Optional[RegimeAllocator] = None,
        rebalance_freq: str = "ME",
    ) -> None:
        self._allocator = allocator or RegimeAllocator()
        self._rebalance_freq = rebalance_freq

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        regime_history: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> BacktestResult:
        """Run the backtest over the common sample period.

        Args:
            regime_history: DataFrame (T × n_regimes) of regime probabilities
                with DatetimeIndex.
            asset_returns: DataFrame (T × n_assets) of simple period returns
                with DatetimeIndex.  Columns must match the asset classes in
                the allocator config.

        Returns:
            :class:`BacktestResult` with performance metrics and equity curve.
        """
        # Align on common dates
        common = regime_history.index.intersection(asset_returns.index)
        if len(common) < 12:
            raise ValueError(f"Insufficient common dates: {len(common)}")

        probs = regime_history.loc[common]
        rets = asset_returns.loc[common]

        # Compute weights for each period
        weights_df = self._allocator.get_allocation_dataframe(probs)

        # Align asset columns to weights columns (fill missing assets with 0)
        asset_cols = weights_df.columns.tolist()
        rets_aligned = rets.reindex(columns=asset_cols, fill_value=0.0)

        # Portfolio returns: shift weights by 1 (invest at start of period)
        port_rets = (weights_df.shift(1) * rets_aligned).sum(axis=1)

        # Compute metrics
        equity_curve = (1 + port_rets).cumprod()
        equity_curve.name = "regime_strategy"

        total_return = equity_curve.iloc[-1] - 1
        n_periods = len(port_rets)
        periods_per_year = _infer_periods_per_year(port_rets.index)

        annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        sharpe_ratio = (
            port_rets.mean() / port_rets.std(ddof=1) * np.sqrt(periods_per_year)
            if port_rets.std() > 0
            else 0.0
        )
        max_drawdown = _max_drawdown(equity_curve)
        turnover = _average_turnover(weights_df)

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            turnover=turnover,
            equity_curve=equity_curve,
            weights_history=weights_df,
        )

    def run_benchmarks(
        self,
        asset_returns: pd.DataFrame,
    ) -> dict[str, BacktestResult]:
        """Compute benchmark portfolio performance.

        Args:
            asset_returns: DataFrame (T × n_assets) with columns including
                at least ``"equities"`` and ``"bonds"``.

        Returns:
            Dict with keys ``"buy_and_hold"`` and ``"60_40"``, each mapping
            to a :class:`BacktestResult`.
        """
        results: dict[str, BacktestResult] = {}

        # Buy-and-hold equities
        if "equities" in asset_returns.columns:
            eq_rets = asset_returns["equities"]
            eq_curve = (1 + eq_rets).cumprod()
            n = len(eq_rets)
            ppy = _infer_periods_per_year(eq_rets.index)
            tr = eq_curve.iloc[-1] - 1
            results["buy_and_hold"] = BacktestResult(
                total_return=tr,
                annual_return=(1 + tr) ** (ppy / n) - 1,
                sharpe_ratio=eq_rets.mean() / eq_rets.std(ddof=1) * np.sqrt(ppy) if eq_rets.std() > 0 else 0.0,
                max_drawdown=_max_drawdown(eq_curve),
                turnover=0.0,
                equity_curve=eq_curve.rename("buy_and_hold"),
                weights_history=pd.DataFrame({"equities": 1.0}, index=asset_returns.index),
            )

        # 60/40 portfolio
        if "equities" in asset_returns.columns and "bonds" in asset_returns.columns:
            bal_rets = 0.6 * asset_returns["equities"] + 0.4 * asset_returns["bonds"]
            bal_curve = (1 + bal_rets).cumprod()
            n = len(bal_rets)
            ppy = _infer_periods_per_year(bal_rets.index)
            tr = bal_curve.iloc[-1] - 1
            results["60_40"] = BacktestResult(
                total_return=tr,
                annual_return=(1 + tr) ** (ppy / n) - 1,
                sharpe_ratio=bal_rets.mean() / bal_rets.std(ddof=1) * np.sqrt(ppy) if bal_rets.std() > 0 else 0.0,
                max_drawdown=_max_drawdown(bal_curve),
                turnover=0.0,
                equity_curve=bal_curve.rename("60_40"),
                weights_history=pd.DataFrame(
                    {"equities": 0.6, "bonds": 0.4}, index=asset_returns.index
                ),
            )

        return results

    def plot_equity_curves(
        self,
        strategy_result: BacktestResult,
        benchmark_results: Optional[dict[str, BacktestResult]] = None,
    ) -> None:
        """Plot equity curves using matplotlib.

        Args:
            strategy_result: Backtest result for the regime strategy.
            benchmark_results: Optional dict of benchmark results.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — cannot plot equity curves")
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        strategy_result.equity_curve.plot(ax=ax, label="Regime Strategy", linewidth=2)

        if benchmark_results:
            for name, res in benchmark_results.items():
                res.equity_curve.plot(ax=ax, label=name.replace("_", " ").title(), linestyle="--")

        ax.set_title("Portfolio Equity Curves")
        ax.set_ylabel("Cumulative Value (start = 1.0)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum peak-to-trough drawdown."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def _average_turnover(weights_df: pd.DataFrame) -> float:
    """Compute average one-way portfolio turnover per period."""
    diff = weights_df.diff().abs().sum(axis=1)
    return float(diff.mean())


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer approximate number of periods per year from a DatetimeIndex."""
    if len(index) < 2:
        return 12.0
    median_days = pd.Series(index).diff().dt.days.median()
    if median_days is None or median_days <= 0:
        return 12.0
    return 365.25 / median_days
