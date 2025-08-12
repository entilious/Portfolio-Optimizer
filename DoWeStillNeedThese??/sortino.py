import os, glob
import numpy as np
import pandas as pd

ROOT_DIR       = "Data"          # top-level directory containing sub-folders
ANNUAL_RF_RATE = 0.02            # flat 2 % p.a. → /252 for daily
DATE_FMT       = "%Y-%m-%d"      # adjust if your dates use another layout

def load_price_series(path: str, date_fmt: str) -> pd.Series:
    """
    Return a float Series of prices with a clean datetime index.
    Tries 3-level header first, then simple header.
    """
    try:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            close_cols = [
                col for col in df.columns
                if str(col[1]).strip().lower() in ("close", "adj close")
            ]
            if close_cols:
                s = df[close_cols[0]]
                s.index = pd.to_datetime(s.index, format=date_fmt, errors="coerce")
                return pd.to_numeric(s, errors="coerce").dropna().sort_index()
    except Exception:
        pass  # fall through

    df = pd.read_csv(path, index_col=0)          # leave dates as object
    df.columns = [c.strip().lower() for c in df.columns]

    if "close" in df.columns:
        s = df["close"]
    elif "adj close" in df.columns:
        s = df["adj close"]
    else:
        raise KeyError(f"{os.path.basename(path)} – no 'Close' column found")

    s.index = pd.to_datetime(s.index, format=date_fmt, errors="coerce")
    return pd.to_numeric(s, errors="coerce").dropna().sort_index()


def compute_sortino(prices: pd.Series, rf_annual: float) -> float:
    """
    Annualised Sortino ratio from a price Series.
    """
    daily_ret      = prices.pct_change(fill_method=None).dropna()
    excess_ret     = daily_ret - rf_annual / 252.0
    downside_dev   = np.sqrt((np.minimum(excess_ret, 0.0) ** 2).mean()) * np.sqrt(252)
    ann_excess_ret = excess_ret.mean() * 252
    return ann_excess_ret / downside_dev


csv_paths = glob.glob(os.path.join(ROOT_DIR, "**", "*.csv"), recursive=True)
if not csv_paths:
    raise FileNotFoundError(f"No CSV files found under {ROOT_DIR}")

ratios = {}

for path in csv_paths:
    stem   = os.path.splitext(os.path.basename(path))[0]
    ticker = stem.split("_")[0].upper()          

    prices = load_price_series(path, DATE_FMT)
    if len(prices) < 2:
        print(f"{ticker}: skipped (not enough data)")
        continue

    ratios[ticker] = compute_sortino(prices, ANNUAL_RF_RATE)

print("\nAnnualised Sortino ratios found under", ROOT_DIR, "\n")
print(pd.Series(ratios).sort_index().round(4))