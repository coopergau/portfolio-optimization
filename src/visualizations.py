import numpy as np 
import matplotlib.pyplot as plt
from optimization import optimize_portfolio, portfolio_metrics

MIN_RETURN = 0.05
MAX_RETURN = 0.25
NUM_RETURNS = 50

def get_efficient_frontier(returns, cov_matrix, min_return, max_return, num_returns):
    """
    Calculates a series returns and their correesponding minimum risks, which can be used 
    to plot the efficient frontier.

    Args:
        returns (np.array): Expected returns for each asset (1D array).
        cov_matrix (np.array): Covariance matrix of asset returns (2D array).
        min_return (float): Smallest target return.
        max_return (float): Largest target return.
        num_returns (float): Amount of portfolios calculated.
    
    Returns:
        tuple: (risks, actual_returns)
            - risks (list): The risk of each portfolio.
            - actual_return (list): The actual return of each portfolio, which can sometimes be
            higher than the target return.
    """
    target_returns = np.linspace(min_return, max_return, num_returns)

    risks = []
    actual_returns = []
    for target_return in target_returns:
        weights = optimize_portfolio(returns, cov_matrix, target_return)
        portfolio_return, portfolio_risk = portfolio_metrics(returns, cov_matrix, weights)

        risks.append(portfolio_risk)
        actual_returns.append(portfolio_return)

    return risks, actual_returns
     
def get_random_portfolios(returns, cov_matrix, amount):
    """
    Calculates the risks and returns for a series of randomly weighted porfolios.

    Args:
        returns (np.array): Expected returns for each asset (1D array).
        cov_matrix (np.array): Covariance matrix of asset returns (2D array).
        amount (int): Amount of portfolios generated.

    Returns:
        tuple: (risks, actual_returns)
            - risks (list): The risk of each portfolio.
            - actual_return (list): The return of each portfolio. Theres no optimization
            here so there is no target return.
    """
    num_assets = len(returns)
    risks = []
    actual_returns = []
    for _ in range(amount):
        weights = np.random.rand(num_assets)
        weights = weights / np.sum(weights)

        portfolio_return, portfolio_risk = portfolio_metrics(returns, cov_matrix, weights)

        risks.append(portfolio_risk)
        actual_returns.append(portfolio_return)

    return risks, actual_returns

def plot_efficient_frontier(risks, actual_returns):
    """
    Plots the efficient frontier line and points using risk and return values.
    """
    plt.plot(risks, actual_returns)
    plt.scatter(risks, actual_returns)
    plt.show()

def plot_random_portfolios(risks, actual_returns):
    """
    Plots portfolios as points on the return vs. risk graph, intended for the randomly weighted portfolios.
    """
    plt.scatter(risks, actual_returns)
    plt.show()


def plot_random_portfolios_with_EF(ef_risks, ef_returns, random_risks, random_returns):
    """
    Plots both the efficient frontier (line and points) and randomly generated portfolios.
    """
    plt.scatter(random_risks, random_returns, c="blue")
    plt.plot(ef_risks, ef_returns, "orange")
    plt.scatter(ef_risks, ef_returns, c="orange")
    plt.show()