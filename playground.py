# Portfolio Optimization Framework - v1

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Portfolio Optimization class. Easier to contain functions and stuff.
    """
    
    def __init__(self, trade_universe):
        """
        Initialize with trade universe dictionary
        
        Parameters:
        trade_universe (dict): Dictionary with sector names as keys and ticker lists as values
        """
        self.trade_universe = trade_universe
        self.asset_data = {}
        self.returns_data = None
        self.expected_returns = None
        self.cov_matrix = None
        self.correlation_matrix = None
        
    def load_asset_data(self, data_path="Data/"):
        """
        Load processed data
        """
        all_returns = []
        asset_names = []
        
        for sector in self.trade_universe.keys():
            sector_path = os.path.join(data_path, sector)
            if os.path.exists(sector_path):
                for file in os.listdir(sector_path):
                    if file.endswith('_5Y_data.csv'):
                        ticker = file.split('_5Y_data.csv')[0]
                        file_path = os.path.join(sector_path, file)
                        
                        # Load data
                        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                        if 'Daily_Pct_Change' in df.columns:
                            returns = df['Daily_Pct_Change'].dropna()
                            all_returns.append(returns)
                            asset_names.append(ticker)
                            
                            # Store individual asset data
                            self.asset_data[ticker] = {
                                'sector': sector,
                                'returns': returns,
                                'annualized_return': self.annualized_return(returns),
                                'volatility': returns.std() * np.sqrt(252)
                            }
        
        # Create returns matrix
        if all_returns: # PS: Need to perform sanity check to ensure all assets were loaded and their rreturns were calculated.
            # Align all return series to common dates
            returns_df = pd.concat(all_returns, axis=1, keys=asset_names)
            returns_df = returns_df.dropna()  # Remove dates where any asset has missing data
            
            self.returns_data = returns_df
            self.expected_returns = returns_df.mean() * 252  # Annualized expected returns
            self.cov_matrix = returns_df.cov() * 252  # Annualized covariance matrix
            self.correlation_matrix = returns_df.corr()
            
        print(f"Loaded data for {len(asset_names)} assets")
        return self.returns_data is not None
    
    def annualized_return(self, daily_returns, periods=252):
        """Calculate annualized return from daily returns"""
        compounded = np.prod(1 + daily_returns)
        return np.power(compounded, (periods / len(daily_returns))) - 1
    
    def calculate_portfolio_metrics(self, weights):
        """
        Calculate portfolio return, volatility, and Sharpe ratio
        
        Parameters:
        weights (array): Portfolio weights
        
        Returns:
        tuple: (return, volatility, sharpe_ratio)
        """
        weights = np.array(weights)
        
        # Portfolio return
        portfolio_return = np.sum(self.expected_returns * weights)
        
        # Portfolio volatility
        # Since > 2 assets, Port_vol = sqrt(transposed weights * [covariance matrix * weights])
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol != 0 else 0
        
        return portfolio_return, portfolio_vol, sharpe_ratio
    
    def negative_sharpe_ratio(self, weights):
        """Objective function for maximizing Sharpe ratio"""
        _, _, sharpe = self.calculate_portfolio_metrics(weights)
        return -sharpe
    
    def optimize_max_sharpe(self, constraints=None):
        """
        Find portfolio weights that maximize Sharpe ratio
        
        Parameters:
        constraints (list): Additional constraints for optimization
        
        Returns:
        dict: Optimization results including weights and metrics
        """
        n_assets = len(self.expected_returns)
        
        # Initial guess (equal weights)
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Bounds (all weights between 0 and 1)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Constraint: weights sum to 1
        constraint_sum = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Combine with additional constraints if provided
        all_constraints = [constraint_sum]
        if constraints:
            all_constraints.extend(constraints)
        
        # Optimize
        result = minimize(
            self.negative_sharpe_ratio,
            init_weights,
            method='SLSQP', # Sequential Least Squares Programming algorithm - I still don't understand this. GPT gods have used this, so I shall do the same.
            bounds=bounds,
            constraints=all_constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(optimal_weights)
            
            return {
                'weights': optimal_weights,
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def optimize_min_variance(self, target_return=None): # The optimization technique of choice = Min Variance Optimization
        """
        Find portfolio weights that minimize variance
        
        Parameters:
        target_return (float): Target portfolio return (optional)
        
        Returns:
        dict: Optimization results
        """
        n_assets = len(self.expected_returns)
        x0 = np.array([1/n_assets] * n_assets)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(self.expected_returns * x) - target_return
            })
        
        # Objective: minimize variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(optimal_weights)
            
            return {
                'weights': optimal_weights,
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def generate_efficient_frontier(self, num_portfolios=100):
        """
        Generate efficient frontier
        
        Parameters:
        num_portfolios (int): Number of portfolios to generate
        
        Returns:
        DataFrame: Efficient frontier data
        """
        # Find range of returns
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            result = self.optimize_min_variance(target_return=target_ret)
            if result['success']:
                efficient_portfolios.append({
                    'return': result['return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'weights': result['weights']
                })
        
        return pd.DataFrame(efficient_portfolios)
    
    def add_sector_constraints(self, max_sector_weight=0.4):
        """
        Create sector weight constraints
        
        Parameters:
        max_sector_weight (float): Maximum weight allowed per sector
        
        Returns:
        list: List of constraint functions
        """
        constraints = []
        
        # Group assets by sector
        sector_assets = {}
        asset_list = list(self.returns_data.columns)
        
        for asset in asset_list:
            if asset in self.asset_data:
                sector = self.asset_data[asset]['sector']
                if sector not in sector_assets:
                    sector_assets[sector] = []
                sector_assets[sector].append(asset_list.index(asset))
        
        # Create constraints for each sector
        for sector, asset_indices in sector_assets.items():
            constraint = {
                'type': 'ineq',
                'fun': lambda x, indices=asset_indices: max_sector_weight - sum(x[i] for i in indices)
            }
            constraints.append(constraint)
        
        return constraints
    
    def display_portfolio_summary(self, weights, title="Portfolio Summary"):
        """
        Display detailed portfolio summary
        
        Parameters:
        weights (array): Portfolio weights
        title (str): Title for the summary
        """
        print(f"\n{title}")
        print("=" * len(title))
        
        portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(weights)
        
        print(f"Expected Annual Return: {portfolio_return:.2%}")
        print(f"Annual Volatility: {portfolio_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        
        print(f"\nAsset Allocation:")
        print("-" * 40)
        
        asset_names = list(self.returns_data.columns)
        
        # Group by sector
        sector_weights = {}
        for i, weight in enumerate(weights):
            if weight > 0.001:  # Only show significant weights
                asset = asset_names[i]
                sector = self.asset_data[asset]['sector']
                
                if sector not in sector_weights:
                    sector_weights[sector] = []
                
                sector_weights[sector].append((asset, weight))
        
        for sector, assets in sector_weights.items():
            sector_total = sum(weight for _, weight in assets)
            print(f"\n{sector} ({sector_total:.1%}):")
            for asset, weight in sorted(assets, key=lambda x: x[1], reverse=True):
                print(f"  {asset}: {weight:.1%}")


def main():

    trade_universe = {
        'TECH': ['CRM', 'PLTR', 'NVDA', 'GOOGL', 'NFLX'],
        'PHARMA': ['PFE', 'CRSP', 'ABBV', 'GILD', 'JNJ'],
        'ENERGY': ['VLO', 'NAT', 'NINE', 'EOG'],
        'DEFENSE': ['SARO', 'RKLB', 'SPCE', 'KITT']
    }
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(trade_universe)
    
    # Load data (assumes data_collection.py has been run)
    if optimizer.load_asset_data():
        print("Data loaded successfully!")
        
        # Optimize for maximum Sharpe ratio
        max_sharpe_result = optimizer.optimize_max_sharpe()
        
        if max_sharpe_result['success']:
            print("\nMaximum Sharpe Ratio Portfolio:")
            optimizer.display_portfolio_summary(
                max_sharpe_result['weights'], 
                "Maximum Sharpe Ratio Portfolio"
            )
        
        # Optimize for min variance - returning the same result as Max Sharpe scenario. Need to tweak
        min_var_result = optimizer.optimize_min_variance()
        
        if min_var_result['success']:
            print("\nMinimum Variance Portfolio:")
            optimizer.display_portfolio_summary(
                max_sharpe_result['weights'], 
                "Minimum Variance Portfolio"
            )
        
        # Add sector constraints and re-optimize
        sector_constraints = optimizer.add_sector_constraints(max_sector_weight=0.3)
        constrained_result = optimizer.optimize_max_sharpe(constraints=sector_constraints)
        
        if constrained_result['success']:
            print("\n" + "="*60)
            optimizer.display_portfolio_summary(
                constrained_result['weights'], 
                "Sector-Constrained Portfolio (Max 30% per sector)"
            )
    
    else:
        print("Failed to load data. Make sure to run data_collection.py first!")

if __name__ == "__main__":
    main()