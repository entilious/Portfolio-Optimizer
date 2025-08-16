# backtest.py
# Visual backtest using ex-ante methods: compare $100k invested in strategy vs benchmark.

from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Import the ex-ante backtester + its default universe/config
from backtest import (
    ExAnteBacktester,
    UNIVERSE,
    FREQ,
    LOOKBACK_DAYS,
    TX_COST,
    BENCHMARK_TICKER,
    ensure_dir,
)

# -------- Config defaults (can be overridden via CLI) --------
REPORT_DIR_DEFAULT = "Sandbox/reports"
START_CAPITAL_DEFAULT = 100_000.0  # $100k


def fetch_benchmark_returns(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    px = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if px.empty or "Close" not in px.columns:
        return pd.Series(dtype=float, name=ticker)
    ret = px["Close"].pct_change().dropna()
    ret.name = ticker
    return ret


def ensure_series(x, name: str | None = None) -> pd.Series:
    """
    Coerce DataFrame/ndarray-like to a 1-D Series (fixes 'Data must be 1-dimensional').
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            raise ValueError(f"Expected 1 column, got {x.shape[1]}")
    elif not isinstance(x, pd.Series):
        # try to build a Series; need an index
        x = pd.Series(np.asarray(x).squeeze())
    else:
        x = x.squeeze()
    if name:
        x.name = name
    return x


def equity_curve(returns, start_capital: float) -> pd.Series:
    """
    Turn daily return series into a dollar equity curve starting at start_capital.
    Robust to receiving a single-column DataFrame.
    """
    returns = ensure_series(returns)
    eq = start_capital * (1.0 + returns).cumprod()
    eq.name = returns.name
    return eq


def plot_equity_curves(eq_strategy: pd.Series, eq_bench: pd.Series, out_path: str, title: str):
    """
    Plot and save the equity curves for strategy vs benchmark.
    """
    ensure_dir(os.path.dirname(out_path))

    # Align to common dates
    idx = eq_strategy.index.intersection(eq_bench.index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates between strategy and benchmark to plot.")
    s = ensure_series(eq_strategy.loc[idx], name=eq_strategy.name or "strategy")
    b = ensure_series(eq_bench.loc[idx], name=eq_bench.name or "benchmark")

    plt.figure(figsize=(12, 6))
    plt.plot(s.index, s.values, label=s.name, linewidth=2)
    plt.plot(b.index, b.values, label=b.name, linewidth=2, linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] {out_path}")


def run_backtest(
    strategy_mode: str,
    sector_cap: float | None,
    freq: str,
    lookback_days: int,
    tx_cost: float,
    start_capital: float,
    benchmark_ticker: str,
    report_dir: str,
):
    # 1) Build ex-ante strategy series
    bt = ExAnteBacktester(
        trade_universe=UNIVERSE,
        freq=freq,
        lookback_days=lookback_days,
        transaction_cost=tx_cost,
    )
    strat_name = f"ExAnte_{strategy_mode}" + (f"_cap{int(sector_cap*100)}" if sector_cap else "")
    strat_series = bt.run_strategy(
        mode=strategy_mode,
        sector_cap=sector_cap,
        snapshot_csv=os.path.join("Sandbox/pit", f"weights_{strat_name}.csv"),
    )
    strat_series = ensure_series(strat_series, name=strat_name)

    # 2) Fetch benchmark over same window
    bench = fetch_benchmark_returns(benchmark_ticker, strat_series.index.min(), strat_series.index.max())
    bench = ensure_series(bench, name=benchmark_ticker)

    # 3) Build equity curves from $100k
    eq_strat = equity_curve(strat_series, start_capital)
    eq_bench = equity_curve(bench, start_capital)

    # Align and also save CSV for inspection
    idx = eq_strat.index.intersection(eq_bench.index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates between equity curves.")
    eq_s = ensure_series(eq_strat.loc[idx], name="equity_strategy")
    eq_b = ensure_series(eq_bench.loc[idx], name="equity_benchmark")
    rs = ensure_series(strat_series.loc[idx], name="daily_ret_strategy")
    rb = ensure_series(bench.loc[idx], name="daily_ret_benchmark")

    df = pd.concat([eq_s, eq_b, rs, rb], axis=1)

    ensure_dir(report_dir)
    csv_path = os.path.join(report_dir, f"{strat_name}_vs_{benchmark_ticker}_equity.csv")
    df.to_csv(csv_path)
    print(f"[Saved] {csv_path}")

    # 4) Plot and save figure
    png_path = os.path.join(report_dir, f"{strat_name}_vs_{benchmark_ticker}_equity.png")
    plot_equity_curves(eq_strat, eq_bench, png_path, title=f"{strat_name} vs {benchmark_ticker} — $100k Growth")

    # 5) Print quick summary
    years = (df.index[-1] - df.index[0]).days / 365.25 if len(df) > 1 else np.nan

    def cagr(series: pd.Series) -> float:
        if years and years > 0:
            return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
        return np.nan

    def max_drawdown(series: pd.Series) -> float:
        peak = series.cummax()
        dd = (series / peak) - 1.0
        return float(dd.min())

    cagr_strat = cagr(eq_s)
    cagr_bench = cagr(eq_b)
    mdd_strat = max_drawdown(eq_s)
    mdd_bench = max_drawdown(eq_b)

    print("\nSummary")
    print("-------")
    print(f"Strategy:   {strat_name}")
    print(f"Benchmark:  {benchmark_ticker}")
    print(f"CAGR (S):   {cagr_strat:.2%} | MaxDD (S): {mdd_strat:.2%}")
    print(f"CAGR (B):   {cagr_bench:.2%} | MaxDD (B): {mdd_bench:.2%}")


def parse_args():
    p = argparse.ArgumentParser(description="Ex-ante backtest with $100k equity comparison plot.")
    p.add_argument("--strategy", choices=["max_sharpe", "min_variance"], default="max_sharpe",
                   help="Which optimizer objective to use at each decision date.")
    p.add_argument("--sector-cap", type=float, default=None,
                   help="Optional per-sector cap (e.g., 0.30 for 30%%).")
    p.add_argument("--freq", default=FREQ, help="Decision cadence: 'M', 'Q', 'W', etc.")
    p.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS,
                   help="Trailing days used to estimate μ and Σ (point-in-time).")
    p.add_argument("--tx-cost", type=float, default=TX_COST,
                   help="Turnover cost per rebalance (applied once at decision).")
    p.add_argument("--start-capital", type=float, default=START_CAPITAL_DEFAULT,
                   help="Starting portfolio value in dollars.")
    p.add_argument("--benchmark", default=BENCHMARK_TICKER,
                   help="Benchmark ticker (default from ex_ante_backtest).")
    p.add_argument("--report-dir", default=REPORT_DIR_DEFAULT,
                   help="Folder to save PNG and CSV outputs.")
    return p.parse_args()


def main():
    args = parse_args()
    run_backtest(
        strategy_mode=args.strategy,
        sector_cap=args.sector_cap,
        freq=args.freq,
        lookback_days=args.lookback_days,
        tx_cost=args.tx_cost,
        start_capital=args.start_capital,
        benchmark_ticker=args.benchmark,
        report_dir=args.report_dir,
    )


if __name__ == "__main__":
    main()
