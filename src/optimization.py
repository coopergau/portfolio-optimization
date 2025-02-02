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
