# benchmark.py
# Benchmark strategies from playground.py against S&P 500, Dow Jones, and NASDAQ using QuantStats.

import os
import pandas as pd
import numpy as np
import quantstats as qs
import yfinance as yf

# Import your optimizer & universe from playground.py
from playground import PortfolioOptimizer

# ---------- Config ----------
REPORT_DIR = "Sandbox/reports"
BENCHMARK_ETFS = {
    "S&P 500": "SPY",   # ETF proxy for S&P 500
    #"Dow Jones": "DIA", # ETF proxy for Dow Jones Industrial Average
    #"NASDAQ": "QQQ",    # ETF proxy for NASDAQ-100
}

TRADE_UNIVERSE = {
    "TECH": [
        "SNPS", "CDNS", "TER", "MCHP", "MPWR",
        "ANSS", "KEYS", "FTNT", "SMTC", "NTNX"
    ],
    "PHARMA": [
        "BMRN", "VTRS", "NBIX", "TECH", "INSM",
        "HRMY", "SGEN", "AMGN", "REGN", "VRTX"
    ],
    "ENERGY": [
        "FANG", "PXD", "HES", "OXY", "MUR",
        "APA", "DVN", "SM", "EQT", "AR"
    ],
    "DEFENSE": [
        "HII", "TDG", "CW", "HEI", "KTOS",
        "LDOS", "BWXT", "MRCY", "AXON", "AVAV"
    ],
    "INDUSTRIALS": [
        "EMR", "ROK", "XYL", "IEX", "DOV",
        "ALLE", "AME", "LECO", "PNR", "ITT"
    ],
    "CONSUMER": [
        "YETI", "CROX", "DKS", "RH", "BC",
        "DECK", "PVH", "SKX", "TPR", "COLM"
    ]
}



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def portfolio_returns_from_weights(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Given a returns matrix (daily %) and a weight vector, compute the daily portfolio returns series.
    """
    weights = np.asarray(weights).reshape(-1)
    # Align columns to weight order if needed (assumes returns_df columns are in the same order as weights)
    port_ret = returns_df.dot(weights)
    port_ret.name = "strategy"
    return port_ret


def compute_strategies():
    """
    Build the optimizer, load data, and compute weights & return series for:
      - Max Sharpe
      - Min Variance
      - Sector-Constrained Max Sharpe (30% per sector)
    Returns a dict of {strategy_name: returns_series}
    """
    optimizer = PortfolioOptimizer(TRADE_UNIVERSE)

    if not optimizer.load_asset_data():
        raise RuntimeError("Failed to load market data.")

    strategies = {}

    # Max Sharpe
    max_sharpe = optimizer.optimize_max_sharpe()
    if max_sharpe.get("success"):
        sr = portfolio_returns_from_weights(optimizer.returns_data, max_sharpe["weights"])
        strategies["Max_Sharpe"] = sr

    # Min Variance
    min_var = optimizer.optimize_min_variance()
    if min_var.get("success"):
        sr = portfolio_returns_from_weights(optimizer.returns_data, min_var["weights"])
        strategies["Min_Variance"] = sr

    # Sector-Constrained Max Sharpe (30% per sector)
    constraints = optimizer.add_sector_constraints(max_sector_weight=0.3)
    constrained = optimizer.optimize_max_sharpe(constraints=constraints)
    if constrained.get("success"):
        sr = portfolio_returns_from_weights(optimizer.returns_data, constrained["weights"])
        strategies["Sector_Constrained_Max_Sharpe_30pct"] = sr

    # Drop any empty/NaN series and align indexes
    for k, s in list(strategies.items()):
        strategies[k] = s.dropna()

    return strategies


def fetch_benchmark_returns(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Fetch benchmark ETF adjusted close prices and compute daily returns.
    """
    px = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if px.empty or "Close" not in px.columns:
        return pd.Series(dtype=float)
    ret = px["Close"].pct_change().dropna()
    ret.name = ticker
    return ret


def make_quantstats_reports(strategies: dict):
    """
    For each strategy and each benchmark ETF, generate a QuantStats HTML report.
    """
    ensure_dir(REPORT_DIR)
    qs.extend_pandas()

    # Determine global date range across strategies to fetch consistent benchmarks
    combined_index = None
    for s in strategies.values():
        combined_index = s.index if combined_index is None else combined_index.union(s.index)
    start = combined_index.min()
    end = combined_index.max()

    # Pre-fetch all benchmarks once
    benchmarks = {}
    for label, ticker in BENCHMARK_ETFS.items():
        b = fetch_benchmark_returns(ticker, start, end)
        benchmarks[label] = b

    # Generate reports
    outputs = []
    for strat_name, strat_rets in strategies.items():
        # Align per benchmark and render report
        for bench_label, bench_rets in benchmarks.items():
            if bench_rets.empty:
                continue

            # Align indexes
            idx = strat_rets.index.intersection(bench_rets.index)
            s_aligned = strat_rets.loc[idx]
            b_aligned = bench_rets.loc[idx]

            if len(s_aligned) < 50:  # require some data points for a meaningful report
                continue

            out_path = os.path.join(
                REPORT_DIR,
                f"QuantStats_{strat_name}_vs_{bench_label.replace(' ', '')}.html"
            )

            # Create HTML report
            qs.reports.html(
                s_aligned,
                benchmark=b_aligned,
                rf=0.02,     # Risk-free rate (annualized, e.g., 2% = 0.02)
                title=f"{strat_name} vs {bench_label}",
                output=out_path
            )
            outputs.append(out_path)

    return outputs


def main():
    strategies = compute_strategies()
    if not strategies:
        raise RuntimeError("No strategies computed.")

    reports = make_quantstats_reports(strategies)

    print("\nGenerated Reports:")
    for r in reports:
        print(f" - {r}")


if __name__ == "__main__":
    main()
