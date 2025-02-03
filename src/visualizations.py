import numpy as np 
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from optimization import optimize_portfolio
from portfolio_stats import portfolio_return_and_risk, portfolio_sharpe_ratio

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
        tuple: (actual_returns, risks, shrpe_ratios)
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

#def get_

def plot_scatter(x, y, line=False, xlabel="X-axis", ylabel="Y-axis", title="Scatter Plot", color="blue"):
    """
    General scatter plot function for return-risk or Sharpe ratio-return plots. Plots dashed lines identifying
    the max y value point.
    
    Args:
    - x: Data for the x-axis.
    - y: Data for the y-axis.
    - line: (bool) Setting to True will plot a line connecting all points.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the graph.
    - color: Color of the scatter points.
    """
    # Find the max y value
    max_y = max(y)
    max_index = y.index(max_y)
    max_x = x[max_index]

    # Plot and label max y value info
    v_line = Line2D([max_x, max_x], [0, max_y], color='red', linestyle='--')
    h_line = Line2D([0, max_x], [max_y, max_y], color='red', linestyle='--')
    plt.gca().add_line(v_line)
    plt.gca().add_line(h_line)
    max_point_label = f'Max {ylabel} Point: ({round(max_x, 2)}, {round(max_y, 2)})'
    fake_handle = Line2D([], [], color='none', label=max_point_label)
    plt.legend(handles=[fake_handle], loc='best', handletextpad=0.1, borderpad=0.2, frameon=False)

    # Plot data
    plt.ylim(0, max_y * 1.2)
    plt.xlim(0, max(x) * 1.1)
    plt.scatter(x, y, c=color)
    if line:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_random_portfolios_with_EF_and_CML(ef_risks, ef_returns, random_risks, random_returns):
    """
    Plots both the efficient frontier (line and points) and randomly generated portfolios.
    """
    plt.scatter(random_risks, random_returns, c="blue")
    plt.plot(ef_risks, ef_returns, "orange")
    plt.scatter(ef_risks, ef_returns, c="orange")
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.title("Efficient Frontier with Capital Market Line")
    plt.show()
