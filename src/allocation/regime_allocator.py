"""Regime-conditional asset allocation.

Maps probabilistic regime output to portfolio weights using a
probability-weighted blend of pre-configured regime allocations.

    blended_weight[asset] = Σ_j P(regime=j) × allocation[regime=j][asset]

Regime-conditional allocations are loaded from ``config/settings.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger


class RegimeAllocator:
    """Maps regime probabilities to blended portfolio weights.

    Args:
        settings_path: Path to ``config/settings.yaml``.
        custom_weights: Override dict ``{regime: {asset: weight}}``.
            If provided, *settings_path* is ignored.
    """

    # Default allocations used when no config is available
    _DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
        "expansion": {"equities": 0.60, "bonds": 0.20, "commodities": 0.15, "cash": 0.05},
        "slowdown":  {"equities": 0.40, "bonds": 0.40, "commodities": 0.10, "cash": 0.10},
        "recession": {"equities": 0.20, "bonds": 0.50, "commodities": 0.05, "cash": 0.25},
        "recovery":  {"equities": 0.55, "bonds": 0.25, "commodities": 0.15, "cash": 0.05},
    }

    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        custom_weights: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        if custom_weights is not None:
            self._regime_weights = custom_weights
        else:
            self._regime_weights = self._load_from_config(settings_path)

        self._asset_classes = sorted(
            {asset for weights in self._regime_weights.values() for asset in weights}
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_allocation(self, regime_probs: dict[str, float]) -> dict[str, float]:
        """Compute probability-weighted blended portfolio allocation.

        Args:
            regime_probs: Mapping ``regime_label → probability``.  Probabilities
                need not sum to exactly 1 — they will be normalised internally.

        Returns:
            Mapping ``asset_class → weight`` that sums to 1.0.
        """
        total_prob = sum(regime_probs.values())
        if total_prob <= 0:
            raise ValueError("Regime probabilities must be positive")

        blended: dict[str, float] = {asset: 0.0 for asset in self._asset_classes}

        for regime, prob in regime_probs.items():
            norm_prob = prob / total_prob
            weights = self._regime_weights.get(regime, {})
            for asset in self._asset_classes:
                blended[asset] += norm_prob * weights.get(asset, 0.0)

        # Renormalise in case of floating-point drift
        total_weight = sum(blended.values())
        if total_weight > 0:
            blended = {k: v / total_weight for k, v in blended.items()}

        return blended

    def get_regime_weights(self, regime: str) -> dict[str, float]:
        """Return the unconditional allocation for a single regime.

        Args:
            regime: Regime label string.

        Returns:
            Dict mapping asset class → weight.

        Raises:
            KeyError: If *regime* is not found in the configuration.
        """
        if regime not in self._regime_weights:
            raise KeyError(f"Regime '{regime}' not found. Available: {list(self._regime_weights)}")
        return self._regime_weights[regime].copy()

    def get_allocation_dataframe(self, regime_probs_df: pd.DataFrame) -> pd.DataFrame:
        """Compute blended allocations for each row of a probability DataFrame.

        Args:
            regime_probs_df: DataFrame (T × n_regimes) of regime probabilities.

        Returns:
            DataFrame (T × n_assets) of portfolio weights.
        """
        rows = []
        for _, row in regime_probs_df.iterrows():
            rows.append(self.get_allocation(row.to_dict()))
        return pd.DataFrame(rows, index=regime_probs_df.index)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_config(self, path: str) -> dict[str, dict[str, float]]:
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Settings file not found at {path}; using default weights")
            return self._DEFAULT_WEIGHTS

        with config_path.open() as fh:
            config = yaml.safe_load(fh)

        weights = (
            config.get("allocation", {}).get("regime_conditional_weights", {})
        )
        if not weights:
            logger.warning("No allocation weights found in config; using defaults")
            return self._DEFAULT_WEIGHTS

        return weights
