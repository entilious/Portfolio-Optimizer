# ex_ante_backtest.py
# Truly ex-ante backtester: point-in-time estimates & as-of weight snapshots.

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import quantstats as qs
import yfinance as yf
from typing import Dict, Optional

from playground import PortfolioOptimizer  # uses your existing class

# ------------ Config ------------
REPORT_DIR = "globalreports/quantstats_reports"
PIT_DIR = "globalreports/pit"  # point-in-time artifact store
RISK_FREE_ANNUAL = 0.02
FREQ = "M"               # decision cadence: 'M' monthly, 'Q' quarterly, etc.
LOOKBACK_DAYS = 252      # trailing window for μ/Σ estimation (no future data)
TX_COST = 0.001          # turnover cost per unit (applied at decision date)
UNIVERSE: Dict[str, list] = {
    "TECH": [
        "SNPS", "CDNS", "TER", "MCHP", "MPWR",
        "KEYS", "FTNT", "SMTC", "NTNX", "ON"
    ],
    "PHARMA": [
        "BMRN", "VTRS", "NBIX", "TECH", "INSM",
        "HRMY", "AMGN", "REGN", "VRTX", "BIIB"
    ],
    "ENERGY": [
        "FANG", "OXY", "MUR", "APA", "DVN",
        "SM", "EQT", "AR", "MRO", "CTRA"
    ],
    "DEFENSE": [
        "HII", "TDG", "CW", "HEI", "KTOS",
        "LDOS", "BWXT", "MRCY", "AXON", "AVAV"
    ],
    "INDUSTRIALS": [
        "EMR", "ROK", "XYL", "IEX", "DOV",
        "ALLE", "AME", "LECO", "PNR", "ITT"
    ],
    "CONSUMER_DISCRETIONARY": [
        "YETI", "CROX", "DKS", "RH", "BC",
        "DECK", "PVH", "SKX", "TPR", "COLM"
    ],
    "CONSUMER_STAPLES": [
        "MKC", "HRL", "CPB", "CAG", "SJM",
        "CHD", "CLX", "KMB", "GIS", "KHC"
    ],
    "FINANCIALS": [
        "MTB", "FITB", "PNC", "HBAN", "CMA",
        "KEY", "TROW", "BEN", "IVZ", "RJF"
    ],
    "UTILITIES": [
        "NEE", "DUK", "SO", "AEP", "EXC",
        "SRE", "XEL", "ED", "PEG", "WEC"
    ],
    "MATERIALS": [
        "APD", "ECL", "ALB", "CF", "MOS",
        "NEM", "NUE", "VMC", "MLM", "FCX"
    ]
}
BENCHMARK_TICKER = "SPY"  # ex-ante benchmark (ETF)

WEIGHT_SAVE_THRESHOLD = 1e-6  # keep all non-trivial weights in snapshots


# ------------ Utilities ------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def last_trade_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    Return decision dates as the last trading day of each period (M/Q/A).
    """
    return index.to_series().resample(freq).last().dropna().index


def fetch_benchmark_returns(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    px = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if px.empty or "Close" not in px.columns:
        return pd.Series(dtype=float)
    ret = px["Close"].pct_change().dropna()
    ret.name = ticker
    return ret


def save_weights_snapshot(
    out_csv: str,
    as_of_date: pd.Timestamp,
    tickers: list[str],
    weights: np.ndarray,
    sectors_map: Dict[str, str],
    meta_by_ticker: Dict[str, dict]
):
    """
    Append one decision-date snapshot of weights to CSV (point-in-time).
    """
    ensure_dir(os.path.dirname(out_csv))
    w = np.asarray(weights).reshape(-1)
    rows = []
    for i, tkr in enumerate(tickers):
        if float(w[i]) <= WEIGHT_SAVE_THRESHOLD:
            continue
        md = meta_by_ticker.get(tkr, {})
        rows.append({
            "as_of": pd.Timestamp(as_of_date).normalize(),  # decision date (as-of)
            "ticker": tkr,
            "sector": sectors_map.get(tkr, md.get("sector", "")),
            "weight": float(w[i]),
            "ann_return_sample": md.get("annualized_return", np.nan),
            "vol_sample": md.get("volatility", np.nan),
        })
    df = pd.DataFrame(rows)
    # Append (create if missing)
    header = not os.path.exists(out_csv)
    df.to_csv(out_csv, mode="a", index=False, header=header)


# ------------ Ex-Ante Backtester ------------
class ExAnteBacktester:
    """
    Ex-ante backtester:
      - For each decision date t, estimates μ/Σ using ONLY data up to t
      - Optimizes weights
      - Applies those weights forward (t, t_next]
      - Saves weights snapshots (as-of t)
    """

    def __init__(
        self,
        trade_universe: Dict[str, list],
        freq: str = FREQ,
        lookback_days: int = LOOKBACK_DAYS,
        transaction_cost: float = TX_COST,
    ):
        self.freq = freq
        self.lookback_days = int(lookback_days)
        self.tc = float(transaction_cost)

        self.optimizer = PortfolioOptimizer(trade_universe)
        ok = self.optimizer.load_asset_data()
        if not ok:
            raise RuntimeError("Failed to load market data.")

        self.returns = self.optimizer.returns_data.sort_index()
        self.tickers = list(self.returns.columns)

        # map ticker -> sector for snapshots
        self.ticker_to_sector = {}
        for tkr, md in self.optimizer.asset_data.items():
            self.ticker_to_sector[tkr] = md.get("sector", "")

    def _set_window_estimates(self, window_df: pd.DataFrame, decision_date: pd.Timestamp):
        """
        Compute μ and Σ from trailing window strictly <= decision_date (no leakage).
        Hook here to blend in ex-ante news signals when you add them later.
        """
        # Historical estimates (price-only)
        mu_hist = window_df.mean() * 252.0
        Sigma_hist = window_df.cov() * 252.0

        # ----- EX-ANTE NEWS HOOK (optional) -----
        # Example: if you have daily_scores table with mu_news_hat as of decision_date, blend here.
        # mu_news_hat = load_mu_hat_asof(decision_date, self.tickers)  # your PIT loader
        # alpha = 0.6  # shrinkage toward historical mean
        # mu_final = alpha * mu_hist + (1 - alpha) * mu_news_hat
        # For now (no news yet), use historical only:
        mu_final = mu_hist
        Sigma_final = Sigma_hist
        # ----------------------------------------

        self.optimizer.expected_returns = mu_final
        self.optimizer.cov_matrix = Sigma_final
        self.optimizer.correlation_matrix = window_df.corr()

    def _optimize(self, mode: str, sector_cap: Optional[float]) -> Optional[np.ndarray]:
        if mode == "max_sharpe":
            constraints = self.optimizer.add_sector_constraints(max_sector_weight=sector_cap) if sector_cap else None
            res = self.optimizer.optimize_max_sharpe(constraints=constraints)
        elif mode == "min_variance":
            res = self.optimizer.optimize_min_variance()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if not res.get("success"):
            return None
        return np.asarray(res["weights"]).reshape(-1)

    def run_strategy(
        self,
        mode: str = "max_sharpe",
        sector_cap: Optional[float] = None,
        snapshot_csv: Optional[str] = None
    ) -> pd.Series:
        """
        Returns daily portfolio returns for the ex-ante strategy and writes PIT snapshots.
        """
        rets = self.returns
        decisions = last_trade_dates(rets.index, self.freq)
        if len(decisions) < 2:
            raise ValueError("Not enough periods for ex-ante backtest.")

        port_series = pd.Series(index=rets.index, dtype=float)
        prev_w = None

        if snapshot_csv is None:
            snapshot_csv = os.path.join(PIT_DIR, f"weights_exante_{mode}{'_cap'+str(int(sector_cap*100)) if sector_cap else ''}.csv")

        for i in range(1, len(decisions)):
            t = decisions[i - 1]    # decision date (as-of)
            t_next = decisions[i]   # next decision date

            # Trailing window strictly up to t (no future data)
            start = t - pd.Timedelta(days=self.lookback_days)
            window = rets.loc[(rets.index > start) & (rets.index <= t)].dropna(how="any")
            if window.empty:
                continue

            # Set μ/Σ and optimize
            self._set_window_estimates(window, t)
            w = self._optimize(mode=mode, sector_cap=sector_cap)
            if w is None:
                continue

            # Save PIT snapshot of weights at t
            save_weights_snapshot(
                snapshot_csv, t, self.tickers, w, self.ticker_to_sector, self.optimizer.asset_data
            )

            # Apply weights only after t, through t_next
            mask = (rets.index > t) & (rets.index <= t_next)
            period = rets.loc[mask]
            if period.empty:
                prev_w = w
                continue

            period_port = period.dot(w)

            # Turnover cost charged at first bar after rebalance
            if prev_w is not None and self.tc > 0.0:
                turnover = float(np.sum(np.abs(w - prev_w)))
                if turnover > 0:
                    first_idx = period_port.index[0]
                    period_port.loc[first_idx] = period_port.loc[first_idx] - self.tc * turnover

            port_series.loc[period_port.index] = period_port.values
            prev_w = w

        name = f"ExAnte_{mode}{'_cap'+str(int(sector_cap*100)) if sector_cap else ''}"
        return port_series.dropna().rename(name)


# ------------ Runner ------------
def make_qs_report(series: pd.Series, out_html: str):
    ensure_dir(REPORT_DIR)
    # Fetch benchmark over the same window
    bench = fetch_benchmark_returns(BENCHMARK_TICKER, series.index.min(), series.index.max())
    # Align
    idx = series.index.intersection(bench.index)
    s_aligned = series.loc[idx]
    b_aligned = bench.loc[idx]
    if len(s_aligned) < 50:
        print(f"[WARN] Not enough overlapping data points to render report for {series.name}")
        return
    qs.extend_pandas()
    qs.reports.html(
        s_aligned,
        benchmark=b_aligned,
        rf=RISK_FREE_ANNUAL,
        title=f"{series.name} vs {BENCHMARK_TICKER}",
        output=out_html
    )
    print(f"[Saved] {out_html}")


def main():
    ensure_dir(REPORT_DIR)
    ensure_dir(PIT_DIR)

    bt = ExAnteBacktester(
        trade_universe=UNIVERSE,
        freq=FREQ,
        lookback_days=LOOKBACK_DAYS,
        transaction_cost=TX_COST,
    )

    # Run a couple of ex-ante strategies
    s1 = bt.run_strategy(mode="max_sharpe", sector_cap=None,
                         snapshot_csv=os.path.join(PIT_DIR, "weights_exante_Max_Sharpe.csv"))
    s2 = bt.run_strategy(mode="max_sharpe", sector_cap=0.30,
                         snapshot_csv=os.path.join(PIT_DIR, "weights_exante_Max_Sharpe_cap30.csv"))
    s3 = bt.run_strategy(mode="min_variance", sector_cap=None,
                         snapshot_csv=os.path.join(PIT_DIR, "weights_exante_Min_Variance.csv"))

    # Reports
    make_qs_report(s1, os.path.join(REPORT_DIR, "ExAnte_Max_Sharpe_vs_SPY.html"))
    make_qs_report(s2, os.path.join(REPORT_DIR, "ExAnte_Max_Sharpe_cap30_vs_SPY.html"))
    make_qs_report(s3, os.path.join(REPORT_DIR, "ExAnte_Min_Variance_vs_SPY.html"))

    print("\nDone. Outputs:")
    print(f" - PIT weights: {PIT_DIR}/weights_exante_*.csv")
    print(f" - Reports:     {REPORT_DIR}/ExAnte_*.html")


if __name__ == "__main__":
    main()
