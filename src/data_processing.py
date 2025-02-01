from datetime import datetime, timedelta
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252

def get_asset_data(tickers, days_back):
    '''
    Calculates the historical individual returns and pairwise covariance 
    of a group of assets.

    Args:
        tickers (list): List of assets.
        days_back (int): The number of calendar days for historical data, 
        considering only trading days within the last days_back days. 
    
    Returns:
        returns_vec (np.array): Expected returns for each asset (1D array).
        cov_matrix (np.array): Covariance matrix of asset returns (2D array).
    '''
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
    cov_matrix = cov_df.to_numpy()

    return returns_vec, cov_matrix
    





