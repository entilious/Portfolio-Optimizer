# benchmark.py
# Benchmark strategies from playground.py against S&P 500 using QuantStats
# and SAVE the chosen stocks (non-trivial weights) for each strategy.

import os
import pandas as pd
import numpy as np
import quantstats as qs
import yfinance as yf

# Import your optimizer & universe from playground.py
from playground import PortfolioOptimizer

# To use Rebalanced strategies on top of playground.py
# from wRebalancing import compute_strategies

# ---------- Config ----------
REPORT_DIR = "Sandbox/reports"
BENCHMARK_ETFS = {
    "S&P 500": "SPY",   # ETF proxy for S&P 500
    # "Dow Jones": "DIA",
    # "NASDAQ": "QQQ",
}

# Save only meaningful positions (very tiny float weights are ignored)
WEIGHT_THRESHOLD = 1e-4  # change to 0.001 if you only want to keep >0.1% weights

TRADE_UNIVERSE = {
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def portfolio_returns_from_weights(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Given a returns matrix (daily %) and a weight vector, compute the daily portfolio returns series.
    """
    weights = np.asarray(weights).reshape(-1)
    port_ret = returns_df.dot(weights)
    port_ret.name = "strategy"
    return port_ret


def save_strategy_weights(strat_name: str, weights: np.ndarray, optimizer: PortfolioOptimizer) -> str:
    """
    Save the chosen stocks and weights for a strategy to CSV.
    Includes sector, annualized return (from historical sample), and volatility.
    """
    ensure_dir(REPORT_DIR)

    tickers = list(optimizer.returns_data.columns)
    w = np.asarray(weights).reshape(-1)

    rows = []
    for i, tkr in enumerate(tickers):
        wt = float(w[i])
        if wt <= WEIGHT_THRESHOLD:
            continue
        meta = optimizer.asset_data.get(tkr, {})
        rows.append({
            "ticker": tkr,
            "sector": meta.get("sector", ""),
            "weight": wt,
            "annualized_return_sample": meta.get("annualized_return", np.nan),
            "volatility_sample": meta.get("volatility", np.nan),
        })

    df = pd.DataFrame(rows).sort_values("weight", ascending=False)
    # sanity: re-normalize selected weights (optional, comment out if you want raw)
    if not df.empty:
        df["weight_norm"] = df["weight"] / df["weight"].sum()

    out_path = os.path.join(REPORT_DIR, f"Weights_{strat_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}  ({len(df)} positions kept > {WEIGHT_THRESHOLD})")
    return out_path


def compute_strategies():
    """
    Build the optimizer, load data, and compute weights & return series for:
      - Max Sharpe
      - Min Variance
      - Sector-Constrained Max Sharpe (30% per sector)
    Also saves per-strategy CSVs with the chosen stocks and weights.
    Returns a dict of {strategy_name: returns_series}
    """
    optimizer = PortfolioOptimizer(TRADE_UNIVERSE)

    if not optimizer.load_asset_data():
        raise RuntimeError("Failed to load market data.")

    strategies = {}

    # Max Sharpe
    max_sharpe = optimizer.optimize_max_sharpe()
    if max_sharpe.get("success"):
        save_strategy_weights("Max_Sharpe", max_sharpe["weights"], optimizer)
        sr = portfolio_returns_from_weights(optimizer.returns_data, max_sharpe["weights"])
        strategies["Max_Sharpe"] = sr

    # Min Variance
    min_var = optimizer.optimize_min_variance()
    if min_var.get("success"):
        save_strategy_weights("Min_Variance", min_var["weights"], optimizer)
        sr = portfolio_returns_from_weights(optimizer.returns_data, min_var["weights"])
        strategies["Min_Variance"] = sr

    # Sector-Constrained Max Sharpe (30% per sector)
    constraints = optimizer.add_sector_constraints(max_sector_weight=0.3)
    constrained = optimizer.optimize_max_sharpe(constraints=constraints)
    if constrained.get("success"):
        save_strategy_weights("Sector_Constrained_Max_Sharpe_30pct", constrained["weights"], optimizer)
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
        for bench_label, bench_rets in benchmarks.items():
            if bench_rets.empty:
                continue

            # Align indexes
            idx = strat_rets.index.intersection(bench_rets.index)
            s_aligned = strat_rets.loc[idx]
            b_aligned = bench_rets.loc[idx]

            if len(s_aligned) < 50:  # need enough data points for a meaningful report
                continue

            out_path = os.path.join(
                REPORT_DIR,
                f"QuantStats_{strat_name}_vs_{bench_label.replace(' ', '')}.html"
            )

            qs.reports.html(
                s_aligned,
                benchmark=b_aligned,
                rf=0.02,     # 2% annualized risk-free rate
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
