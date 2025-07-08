import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import os
from tqdm import tqdm


# helper function to get the sector-wiise data

def get_sector_data(sector, trade_universe):
    """
    Fetches the sector-wise data respective to the trade_universe dictionary.

    Returns: Nothing.
    """
    
    # gather 5-yr historical data and save csv

    if sector: # if the sector is provided, only download that sector's data

        if sector in os.listdir(os.getcwd()):
            print("Sector directory already exists. Skipping creation.") # scalability 100. if the sector exists skip. allows updatation of trade_universe dict to update stocks and their data
        else:
            print(f"Creating directory for sector: {sector}")
            os.mkdir(sector)

        # pull the sector-wise tickers

        tickers = trade_universe[sector]

        # iterate over the titcker and get the data - 5Y
        for ticker in tqdm(tickers):

            print(f"\nSector : {sector}\nDowloading stock info for {ticker}")

            data = yf.download(ticker, period="5Y")

            df = pd.DataFrame(data)

            df.to_csv(f"{sector}/{ticker}_5Y_data.csv")


    else: # if the sector is not provided, download all sectors' data

        for sec in list(trade_universe.keys()):

        # if the sector directory is not present, create it
            if sec in os.listdir(os.getcwd()):
                print("Sector directory already exists. Skipping creation.") # scalability 100. if the sector exists skip. allows updatation of trade_universe dict to update stocks and their data
            else:
                print(f"Creating directory for sector: {sec}")
                os.mkdir(sec)

            # pull the sector-wise tickers

            tickers = trade_universe[sec]

            # iterate over the titcker and get the data - 5Y
            for ticker in tqdm(tickers):

                print(f"\nSector : {sec}\nDowloading stock info for {ticker}")

                data = yf.download(ticker, period="5Y")

                df = pd.DataFrame(data)

                df.to_csv(f"{sec}/{ticker}_5Y_data.csv")

        return print(f"Sector data downloaded successfully.")

trade_universe = {
    'TECH' : ['CRM', 'PLTR', 'NVDA', 'GOOGL', 'NFLX'],
    'PHARMA' : ['PFE', 'CRSP', 'ABBV', 'GILD', 'JNJ'],
    'ENERGY' : ['VLO', 'AISI', 'NAT', 'NINE', 'EOG'],
    'DEFENSE' : ['KTOE', 'SARO', 'RKLB', 'SPCE', 'KITT']
}

get_sector_data('ENERGY', trade_universe)