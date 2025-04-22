import numpy as np

def simulate_portfolio_returns(initial_value, expected_return, expected_risk, total_time, step_size, sims):
    """
    Function simulates a portfolios performance based on its expected return and risk.

    Args:
        initial_value (float): Starting value of the portfolio.
        portfolio_return (float): Expected portfolio return.
        portfolio_risk (float): Expected portfolio risk (standard deviation of returns).
        total_time (float): Total time period simulated, in years.
        step_size (float): Time difference of each incriment, in years.
        sims (int): Number of simulations computed.

    Returns:
        portfolio_paths (np.ndarray) of shape (sims, total_steps + 1): Each row is one path of the potfolio's
        value, there are sims amount of them. Each column is the porfolio value at that given timestep.
    """
    # Time incriments
    total_steps = int(total_time / step_size)
    
    # Initialize blank portfolio paths starting at initia_value
    portfolio_paths = np.empty((sims, total_steps + 1))
    portfolio_paths[:, 0] = initial_value
    
    # Random samples of the standard normal distribution, for the Brownian motion: Z ~ N(0, 1)
    Z = np.random.normal(0, 1, (sims, total_steps))

    # Compute log returns
    drift = (expected_return - 0.5 * expected_risk**2) * step_size
    random_diffusion = expected_risk * np.sqrt(step_size) * Z
    log_returns = drift + random_diffusion

    # Fill the portfolio paths
    portfolio_paths[:, 1:] = portfolio_paths[:, [0]] * np.exp(np.cumsum(log_returns, axis=1))

    return portfolio_paths