import numpy as np
import cvxpy as cp

def optimize_portfolio(returns, cov_matrix, target_return):
    """
    Calculates the weights of a minimum risk portfolio that has meets 
    a specified target return and does not allow short selling.

    Args:
        returns (np.array): Expected returns for each asset (1D array).
        cov_matrix (np.array): Covariance matrix of asset returns (2D array).
        target_return (float): The minimum expected return of a feasible portfolio.

    Returns:
        np.array: The optimal portfolio weights for each asset (1D array).
    """
    weights = cp.Variable(len(cov_matrix))

    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))

    # Constraints: 
    # The weights are proportions so they must sum to 1.
    # No short selling so each weight must be non-negative.
    # The combined expected return must be at least the target return.
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights @ returns >= target_return
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return weights.value

def portfolio_metrics(returns, cov_matrix, weights):
    """
    Calculates the expected return and risk (standard deviation) of a portfolio.

    Args:
        returns (np.array): Expected returns for each asset (1D array).
        cov_matrix (np.array): Covariance matrix of asset returns (2D array).
        weights (np.array): Portfolio weights for each asset (1D array).

    Returns:
        tuple: (portfolio_return, portfolio_risk)
            - portfolio_return (float): Expected portfolio return.
            - portfolio_risk (float): Expected portfolio risk (standard deviation).
    """
    portfolio_return = weights @ returns

    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_risk = np.sqrt(portfolio_variance)

    return portfolio_return, portfolio_risk