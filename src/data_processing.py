from datetime import datetime, timedelta
import yfinance as yf
import numpy as np 
import cvxpy as cp
from .optimization import optimize_portfolio
from .portfolio_stats import portfolio_return_and_risk, portfolio_sharpe_ratio

TRADING_DAYS_PER_YEAR = 252

def get_asset_data(tickers, days_back):
    """
    Calculates the historical individual returns and pairwise covariance 
    of a group of assets.

    Args:
        tickers (list): List of assets.
        days_back (int): The number of calendar days for historical data, 
        considering only trading days within the last days_back days. 
    
    Returns:
        tuple: (returns_vec, cov_matrix)
            returns_vec (np.array): Expected returns for each asset (1D array).
            cov_matrix (np.array): Covariance matrix of asset returns (2D array).
    """
    # Get historical date period
    now = datetime.now()
    start_time = (now - timedelta(days=days_back))

    # Get asset data
    closes = yf.download(tickers, start=start_time, end=now)["Close"]
    closes = closes[tickers] # Make sure the order is the same as the original ticker list
    
    # Get average geometric annualized return 
    daily_returns = closes.pct_change()
    avg_daily_returns = daily_returns.mean()
    annualized_returns = (1 + avg_daily_returns) ** TRADING_DAYS_PER_YEAR - 1
    returns_vec = annualized_returns.squeeze().to_numpy()

    # Get covariance matrix
    cov_df = daily_returns.cov()
    cov_matrix = cov_df.to_numpy() * TRADING_DAYS_PER_YEAR

    return returns_vec, cov_matrix
    
def get_efficient_frontier_data(returns, cov_matrix, risk_free_return, min_return, max_return, num_returns):
    """
    Calculates a series of returns and their risks and Sharpe ratios. Used to plot the efficient frontier
    and Sharpe ratio visuals.

    Args:
        returns (np.array): Expected returns for each asset (1D array).
        cov_matrix (np.array): Covariance matrix of asset returns (2D array).
        min_return (float): Smallest target return.
        max_return (float): Largest target return.
        num_returns (float): Amount of portfolios calculated.
    
    Returns:
        tuple: (actual_returns, risks, sharpe_ratios)
            - actual_return (list): The actual return of each portfolio, which can sometimes be
            higher than the target return.
            - risks (list): The risk of each portfolio.
            - sharpe_ratios (list): The Sharpe ratios of each portfolio.

    """
    target_returns = np.linspace(min_return, max_return, num_returns)

    actual_returns = []
    risks = []
    sharpe_ratios = []
    for target_return in target_returns:
        try:
            weights = optimize_portfolio(returns, cov_matrix, target_return)
            portfolio_return, portfolio_risk = portfolio_return_and_risk(returns, cov_matrix, weights)
            sharpe_ratio = portfolio_sharpe_ratio(portfolio_return, portfolio_risk, risk_free_return)
        except (ValueError, cp.error.SolverError) as e:
            # This will occur if there is not a feasible solution with the given
            # assets and target return
            print(f"Infeasible solution for target return: {target_return}")
            actual_returns.append(np.nan)
            risks.append(np.nan)
            sharpe_ratios.append(np.nan)
            continue 

        actual_returns.append(portfolio_return)
        risks.append(portfolio_risk)
        sharpe_ratios.append(sharpe_ratio)

    return actual_returns, risks, sharpe_ratios
    
def get_random_portfolios(returns, cov_matrix, amount):
    """
    Calculates the risks and returns for a series of randomly weighted porfolios.

    Args:
        returns (np.array): Expected returns for each asset (1D array).
        cov_matrix (np.array): Covariance matrix of asset returns (2D array).
        amount (int): Amount of portfolios generated.

    Returns:
        tuple: (actual_returns, risks)
            - actual_return (list): The return of each portfolio. Theres no optimization
            here so there is no target return.
            - risks (list): The risk of each portfolio.
    """
    num_assets = len(returns)
    risks = []
    actual_returns = []
    for _ in range(amount):
        weights = np.random.rand(num_assets)
        weights = weights / np.sum(weights)

        portfolio_return, portfolio_risk = portfolio_return_and_risk(returns, cov_matrix, weights)

        risks.append(portfolio_risk)
        actual_returns.append(portfolio_return)

    return actual_returns, risks

def find_similar_random_portfolio(returns, cov_matrix, target_return, tolerance=0.005):
    """
    Finds a randomly weight portfolio with a similar return to the target_return provided.

    Args:
        returns (np.array): Expected returns for each asset (1D array).
        cov_matrix (np.array): Covariance matrix of asset returns (2D array).
        target_return (float): The return the random portfolio's return has to be close to.
        tolerance (float): The maximum difference between the target return and the random 
        portfolio return.

    Returns:
        tuple: (example_returns[0], example_risks[0])
            - example_returns[0] (float): The return of the random portfolio.
            - example_risks[0] (float): The risk of the random portfolio.
    """
    while True:
        example_returns, example_risks = get_random_portfolios(returns, cov_matrix, 1)
        if abs(example_returns[0] - target_return) <= tolerance:
            return example_returns[0], example_risks[0]



