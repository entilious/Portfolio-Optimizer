# wRebalancing.py
# Rebalanced strategies compatible with benchmark.py (same compute_strategies() interface).
# Uses existing PortfolioOptimizer from playground.py and applies rolling rebalancing.

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional

from playground import PortfolioOptimizer


class PeriodicRebalanceBacktester:
    """
    Rolling rebalancing backtester:
      - At each rebalance date, use a trailing lookback window to estimate μ and Σ
      - Optimize (max sharpe or min variance)
      - Hold those weights until the next rebalance date
      - Optional sector cap and transaction cost
    """

    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        freq: str = "M",             # 'M' monthly, 'Q' quarterly, etc.
        lookback_days: int = 252,    # ~1Y trading days
        transaction_cost: float = 0.0,   # e.g. 0.001 = 10 bps per unit turnover, applied once at rebalance
        max_sector_weight: Optional[float] = None,
    ):
        if optimizer.returns_data is None or optimizer.returns_data.empty:
            raise ValueError("Call optimizer.load_asset_data() before using the backtester.")
        self.optimizer = optimizer
        self.freq = freq
        self.lookback_days = int(lookback_days)
        self.transaction_cost = float(transaction_cost)
        self.max_sector_weight = max_sector_weight

        # Pre-build sector constraints if requested
        self._constraints = (
            optimizer.add_sector_constraints(max_sector_weight=self.max_sector_weight)
            if self.max_sector_weight is not None else None
        )

    def _set_window_estimates(self, window_df: pd.DataFrame):
        """Refresh μ (expected_returns) and Σ (cov_matrix) from the window."""
        self.optimizer.expected_returns = window_df.mean() * 252.0
        self.optimizer.cov_matrix = window_df.cov() * 252.0
        self.optimizer.correlation_matrix = window_df.corr()

    def _turnover_cost(self, new_w: np.ndarray, prev_w: Optional[np.ndarray]) -> float:
        if prev_w is None or self.transaction_cost <= 0.0:
            return 0.0
        turnover = float(np.sum(np.abs(new_w - prev_w)))
        return self.transaction_cost * turnover

    def _optimize_once(self, mode: str):
        """
        mode: 'max_sharpe' or 'min_variance'
        """
        if mode == "max_sharpe":
            return self.optimizer.optimize_max_sharpe(constraints=self._constraints)
        elif mode == "min_variance":
            return self.optimizer.optimize_min_variance()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def run(self, mode: str = "max_sharpe") -> pd.Series:
        """
        Returns a daily return series for the rebalanced strategy.
        """
        rets = self.optimizer.returns_data.sort_index()
        # Rebalance on last trading day of each period
        rbd = rets.resample(self.freq).last().index
        if len(rbd) < 2:
            raise ValueError("Not enough data to form multiple rebalance periods.")

        port_series = pd.Series(index=rets.index, dtype=float)
        prev_w: Optional[np.ndarray] = None

        for i in range(1, len(rbd)):
            reb_date = rbd[i - 1]
            next_date = rbd[i]

            # Trailing lookback window strictly BEFORE/UP TO rebalance date
            start = reb_date - pd.Timedelta(days=self.lookback_days)
            window = rets.loc[(rets.index > start) & (rets.index <= reb_date)].dropna(how="any")
            if window.empty:
                continue

            # Update μ, Σ from window and optimize
            self._set_window_estimates(window)
            res = self._optimize_once(mode=mode)
            if not res.get("success"):
                continue

            w = np.asarray(res["weights"]).reshape(-1)

            # Apply weights forward AFTER rebalance date until next_date
            mask = (rets.index > reb_date) & (rets.index <= next_date)
            period = rets.loc[mask]
            if period.empty:
                prev_w = w
                continue

            period_port = period.dot(w)

            # One-time turnover cost at first bar after rebalance
            c = self._turnover_cost(w, prev_w)
            if c > 0.0:
                first_idx = period_port.index[0]
                period_port.loc[first_idx] = period_port.loc[first_idx] - c

            port_series.loc[period_port.index] = period_port.values
            prev_w = w

        return port_series.dropna()


# -----------------------------------------------------------------------------------
# Public API expected by your existing benchmark.py
# -----------------------------------------------------------------------------------

def compute_strategies(
    trade_universe: Optional[Dict[str, list]] = None,
    freq: str = "M",
    lookback_days: int = 252,
    transaction_cost: float = 0.001,
) -> Dict[str, pd.Series]:
    """
    Build rebalanced strategies matching the names used in benchmark.py:
      - "Max_Sharpe"
      - "Min_Variance"
      - "Sector_Constrained_Max_Sharpe_30pct"

    Returns a dict: {strategy_name: daily_return_series}
    """
    # If not provided, you can set your default universe here
    if trade_universe is None:
        trade_universe = {
            "TECH": ["SNPS", "CDNS", "TER", "MCHP", "MPWR", "ANSS", "KEYS", "FTNT", "SMTC", "NTNX"],
            "PHARMA": ["BMRN", "VTRS", "NBIX", "TECH", "INSM", "HRMY", "AMGN", "REGN", "VRTX", "BIIB"],
            "ENERGY": ["FANG", "PXD", "HES", "OXY", "MUR", "APA", "DVN", "SM", "EQT", "AR"],
            "DEFENSE": ["HII", "TDG", "CW", "HEI", "KTOS", "LDOS", "BWXT", "MRCY", "AXON", "AVAV"],
        }

    opt = PortfolioOptimizer(trade_universe)
    if not opt.load_asset_data():
        raise RuntimeError("Failed to load market data for rebalanced strategies.")

    strategies: Dict[str, pd.Series] = {}

    # 1) Max Sharpe (rebalanced)
    bt_ms = PeriodicRebalanceBacktester(
        optimizer=opt,
        freq=freq,
        lookback_days=lookback_days,
        transaction_cost=transaction_cost,
        max_sector_weight=None
    )
    strategies["Max_Sharpe"] = bt_ms.run(mode="max_sharpe").rename("Max_Sharpe")

    # 2) Min Variance (rebalanced)
    bt_mv = PeriodicRebalanceBacktester(
        optimizer=opt,
        freq=freq,
        lookback_days=lookback_days,
        transaction_cost=transaction_cost,
        max_sector_weight=None
    )
    strategies["Min_Variance"] = bt_mv.run(mode="min_variance").rename("Min_Variance")

    # 3) Sector-Constrained Max Sharpe (30% per sector, rebalanced)
    bt_sc = PeriodicRebalanceBacktester(
        optimizer=opt,
        freq=freq,
        lookback_days=lookback_days,
        transaction_cost=transaction_cost,
        max_sector_weight=0.30
    )
    strategies["Sector_Constrained_Max_Sharpe_30pct"] = (
        bt_sc.run(mode="max_sharpe").rename("Sector_Constrained_Max_Sharpe_30pct")
    )

    # Clean any NaNs
    for k in list(strategies.keys()):
        strategies[k] = strategies[k].dropna()

    return strategies


# Optional: quick self-test
if __name__ == "__main__":
    s = compute_strategies()
    for k, v in s.items():
        print(k, v.index.min(), "→", v.index.max(), f"{len(v)} points")
