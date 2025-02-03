import numpy as np 
import cvxpy as cp
import matplotlib.pyplot as plt
from optimization import optimize_portfolio
from portfolio_stats import portfolio_return_and_risk, portfolio_sharpe_ratio

def get_efficient_frontier(returns, cov_matrix, min_return, max_return, num_returns):
    """
    Calculates a series of returns and their correesponding minimum risks, which can be used 
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
        try:
            weights = optimize_portfolio(returns, cov_matrix, target_return)
            portfolio_return, portfolio_risk = portfolio_return_and_risk(returns, cov_matrix, weights)
        except (ValueError, cp.error.SolverError) as e:
            # This will occur if there is not a feasible solution with the given
            # portfolio and target return
            print(f"Infeasible solution for target return: {target_return}, Error: {str(e)}")
            risks.append(np.nan)
            actual_returns.append(np.nan)
            continue 

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

        portfolio_return, portfolio_risk = portfolio_return_and_risk(returns, cov_matrix, weights)

        risks.append(portfolio_risk)
        actual_returns.append(portfolio_return)

    return risks, actual_returns

def get_sharpe_ratios_and_returns(returns, cov_matrix, risk_free_return, min_return, max_return, num_returns):
    """
    Calculates a series of returns and their correesponding Sharpe ratios.

    Args:
        returns (np.array): Expected returns for each asset (1D array).
        cov_matrix (np.array): Covariance matrix of asset returns (2D array).
        min_return (float): Smallest target return.
        max_return (float): Largest target return.
        num_returns (float): Amount of portfolios calculated.
    
    Returns:
        tuple: (actual_returns, sharpe_ratios)
            - actual_returns (list): The actual return of each portfolio, which can sometimes be
            higher than the target return.
            - sharpe_ratios (list): The Sharpe ratios of each portfolio.
    """
    target_returns = np.linspace(min_return, max_return, num_returns)

    actual_returns = []
    sharpe_ratios = []
    for target_return in target_returns:
        try:
            weights = optimize_portfolio(returns, cov_matrix, target_return)
            portfolio_return, portfolio_risk = portfolio_return_and_risk(returns, cov_matrix, weights)
            sharpe_ratio = portfolio_sharpe_ratio(portfolio_return, portfolio_risk, risk_free_return)
        except (ValueError, cp.error.SolverError) as e:
            # This will occur if there is not a feasible solution with the given
            # portfolio and target return
            print(f"Infeasible solution for target return: {target_return}, Error: {str(e)}")
            actual_returns.append(np.nan)
            sharpe_ratios.append(np.nan)
            continue 

        actual_returns.append(portfolio_return)
        sharpe_ratios.append(sharpe_ratio)
    
    return actual_returns, sharpe_ratios

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

def plot_sharpe_ratio_vs_returns(actual_returns, sharpe_ratios):
    """
    Plots portfolios as points on the Sharpe ratio vs return graph.
    """
    plt.scatter(actual_returns, sharpe_ratios)
    plt.show()