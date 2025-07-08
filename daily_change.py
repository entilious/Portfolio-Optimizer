import glob, pandas as pd, os

ROOT_DIR = "data"                           # where all the CSVs live
PRICE_COL = "Adj Close"                     # column to use for returns
RET_COL   = "DailyPctChange"                # new column name

for path in glob.glob(f"{ROOT_DIR}/**/*.csv", recursive=True):
    df = pd.read_csv(path)

    # one-liner: daily % change, ×100, round to 3 decimals
    df[RET_COL] = (
        pd.to_numeric(df[PRICE_COL], errors="coerce")
          .pct_change(fill_method=None)     # fractional return
          .mul(100)                         # → percent
          .round(3)                         # 3 decimal places (≈3 sig figs for small numbers)
    )

    df.to_csv(path, index=False)
    print(f"{os.path.basename(path)} – added '{RET_COL}'")