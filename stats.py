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
    return pct_change_df  # keep it as is for calculations later

# helper function to calculatte annualied return for a stock based on daily returns
def annualized_return(daily_returns: pd.Series, periods: int = 252):
    compounded = np.prod(1 + daily_returns)
    print(compounded, len(daily_returns))
    return np.power(compounded, (periods / len(daily_returns))) - 1


def calc_sortino(R: pd.Series, MAR: float, periods: int = 252) -> float:

    """    
    Calculate the Sortino ratio for a given series of returns.
    Parameters
    ----------  
    R : pd.Series
        Series of returns (daily, monthly, etc.).
    MAR : float
        Minimum acceptable return (MAR), typically the risk-free rate or zero.
    periods : int, default=252
        Number of periods in a year (e.g., 252 for daily returns).
    -------
    Returns : float
        The Sortino ratio, which is the ratio of the average excess return to the downside deviation.
    """

    # 1 calculate geometric mean of returns

    # anr : annualized return
    anr = annualized_return(R)

    # 2 calculate downside devaiation. Formula: sqrt([sum(min(R-Mar,0))^2] / n)
    neg_ret = np.minimum(R-MAR, 0) # isolating the negative returns
    down_dev = np.sqrt(np.mean(np.square(neg_ret))) # obtain downside devaition; this is daily downside deviation
    down_dev_annualized = down_dev * np.sqrt(periods)  # annualize the downside deviation
    
    # 3 calculate sortino ratio
    sortino = (anr - MAR) / down_dev_annualized if down_dev_annualized != 0 else np.nan # calc sortino ratio ; handle division by zero error
    
    return sortino

# compute daily percentage change and sortino ratio for each stock in the Data directory
# Calculating individual sortinos because we are building a portfolio from scratch and not using a pre-defined portfolio.

sortino_dict = {}

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

            # Proceed to calculate sortino ratio for the stock
            daily_returns = df['Daily_Pct_Change'].dropna()
            rf_annual = 0.02 # for now MAR is set to risk-free rate; to be changed to index return later
            sortino_ratio = calc_sortino(daily_returns, rf_annual)
            sortino_dict[ticker.split("_5Y_data.csv")[0]] = sortino_ratio
            print(f"Processed {ticker} in sector {sec}")


df_sortino = pd.DataFrame.from_dict(sortino_dict, orient='index', columns=['Sortino Ratio'])
print(df_sortino)