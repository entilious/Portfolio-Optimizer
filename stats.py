import numpy as np
import seaborn as sns
import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm

# compute daily precetnage change for each stock

def compute_daily_pct_change(df):
    """
    Computes the daily percentage change for each stock in the DataFrame.
    
    Parameters:
    df (DataFrame): DataFrame containing stock prices with a 'Close' column.
    
    Returns:
    DataFrame: Parent DataFrame with daily percentage changes.
    """
    pct_change_df = df.pct_change().dropna()
    return pct_change_df * 100  # Convert to percentage

# compute daily percentage change for each stock in the Data directory

for sec in os.listdir("Data/"):
    sec = os.path.join("Data", sec)  # Ensure the path is correct
    if not os.path.isdir(sec):
       continue
    print(f"Processing sector: {sec}")
    for ticker in tqdm(os.listdir(sec)):
        if ticker.endswith("_5Y_data.csv"):
            file_path = os.path.join(sec, ticker)
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            df['Daily_Pct_Change'] = compute_daily_pct_change(df['Close'])
            df.to_csv(file_path)  # Save the updated DataFrame with daily percentage change
            print(f"Processed {ticker} in sector {sec}")