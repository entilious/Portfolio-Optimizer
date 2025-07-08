import os, glob, numpy as np, pandas as pd

ROOT_DIR   = "Data"             # top-level folder containing all CSV trees
RET_COL    = "DailyPctChange"   # column to create / refresh
OVERWRITE  = True               # False → write *_with_ret.csv copies
DATE_COL   = 0                  # first column holds the date
SIG_FIGS   = 3                  # number of significant figures to keep

def round_sig(x: float, sig: int = 3) -> float:
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

csv_paths = glob.glob(os.path.join(ROOT_DIR, "**", "*.csv"), recursive=True)
if not csv_paths:
    raise FileNotFoundError(f"No CSV files found under '{ROOT_DIR}'")

for path in csv_paths:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df.columns = [c.strip() for c in df.columns]

    if "Adj Close" not in df.columns:
        print(f"{os.path.basename(path)} – skipped (no 'Adj Close')")
        continue

    # numeric daily % change rounded to SIG_FIGS
    pct = (
        pd.to_numeric(df["Adj Close"], errors="coerce")
          .pct_change(fill_method=None)       # fractional return
          .mul(100)                           # → percent
          .apply(round_sig, sig=SIG_FIGS)     # keep 3 sig figs
    )

    # insert / refresh the column
    df.drop(columns=[RET_COL], errors="ignore", inplace=True)
    df[RET_COL] = pct

    # save
    out_path = path if OVERWRITE else path.replace(".csv", "_with_ret.csv")
    df.to_csv(out_path, index=False)
    print(f"{os.path.basename(out_path)} – updated '{RET_COL}' (numeric %)")

print("\n✅  All CSVs processed: daily % change added")