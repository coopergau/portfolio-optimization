import numpy as np 
import matplotlib.pyplot as plt
from optimization import optimize_portfolio, portfolio_metrics

MIN_RETURN = 0.05
MAX_RETURN = 0.25
NUM_RETURNS = 50

def get_efficient_frontier(returns, cov_matrix, min_return, max_return, num_returns):
     # Generate evenly spaced target returns
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
    plt.plot(risks, actual_returns)
    plt.scatter(risks, actual_returns)
    plt.show()

def plot_random_portfolios(risks, actual_returns):
    plt.scatter(risks, actual_returns)
    plt.show()


def plot_random_portfolios_with_EF(ef_risks, ef_returns, random_risks, random_returns):
    plt.scatter(random_risks, random_returns, c="blue")
    plt.plot(ef_risks, ef_returns, "orange")
    plt.scatter(ef_risks, ef_returns, c="orange")
    plt.show()