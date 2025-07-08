import numpy as np
import seaborn as sns
import yfinance as yf

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
    return pct_change_df


