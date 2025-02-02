import numpy as np

def portfolio_return_and_risk(returns, cov_matrix, weights):
    """
    Calculates the expected return and risk (standard deviation) of a portfolio.

    Args:
        returns (np.array): Expected returns for each asset (1D array).
        cov_matrix (np.array): Covariance matrix of asset returns (2D array).
        weights (np.array): Portfolio weights for each asset (1D array).

    Returns:
        tuple: (portfolio_return, portfolio_risk)
            - portfolio_return (float): Expected portfolio return.
            - portfolio_risk (float): Expected portfolio risk (standard deviation of returns).
    """
    portfolio_return = weights @ returns

    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_risk = np.sqrt(portfolio_variance)

    return portfolio_return, portfolio_risk

def portfolio_sharpe_ratio(portfolio_return, portfolio_risk, risk_free_rate):
    """
    Calculate the Sharpe ratio for a portfolio.

    Args:
    portfolio_return (float): The portfolio's expected return.
    portfolio_risk (float): The portfolio's expected risk (standard deviation of returns).
    risk_free_rate (float): The risk-free return rate.

    Returns:
    float: The Sharpe ratio.
    
    Raises:
    ValueError: If portfolio_risk is zero, as division by zero is not allowed.
    """
    if portfolio_risk == 0:
        raise ValueError("Portfolio risk cannot be zero for a Sharpe ratio calculation. Results in dividing by zero.")
    
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    return sharpe_ratio
