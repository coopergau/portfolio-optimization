import numpy as np
import pandas as pd
import random
from src.optimization import optimize_portfolio
from src.data_processing import get_asset_data
from src.portfolio_stats import portfolio_return_and_risk


def test_valid_weights():
    """
    Test the weights sum to exactly one and are all non-negative.
    """
    # Select 10 random tickers
    tickers_df = pd.read_csv("nasdaq100.csv")
    all_tickers = tickers_df["Ticker"].tolist()
    num_assets = 10
    tickers = random.sample(all_tickers, num_assets)

    # Get optimal portfolio weights
    days_back = 30
    target_return = 0.10
    returns, cov_matrix = get_asset_data(tickers, days_back)
    optimal_weights = optimize_portfolio(returns, cov_matrix, target_return)

    # Round small negative that are essentially zero
    optimal_weights[abs(0 - optimal_weights) < 1e-5] = 0

    assert np.all(optimal_weights >= 0), "All weights must be non-negative"
    assert np.isclose(np.sum(optimal_weights), 1), "Sum of weights must be 1"


def test_basic_optimal_weights():
    """
    Test the correct weights are returned for a basic case. The two assets have the same expected return,
    no covariance, and a 2:1 ratio of risk so the optimal solution is investing in a 1:2 ratio (opposite 
    to that of the risk).
    """
    returns = np.array([0.15, 0.15])
    cov_matrix = np.array([[0.2, 0.0],
                           [0.0, 0.1]])
    target_return = 0.15
    
    # Pre calculated values based on the given covariance matrix
    expected_weights = np.array([1/3, 2/3])
    actual_weights = optimize_portfolio(returns, cov_matrix, target_return)

    assert np.allclose(expected_weights, actual_weights, atol=1e-8), "Weights do not match expected weights"

def test_fuzz_optimization_is_optimal():
    """
    Uses randomly weighted portfolios to test that none of them provide a lower risk while also
    providing an expected return that is at least the target return.
    """
    # Select 10 random tickers
    tickers_df = pd.read_csv("nasdaq100.csv")
    all_tickers = tickers_df["Ticker"].tolist()
    num_assets = 10
    tickers = random.sample(all_tickers, num_assets)

    # Get optimal portfolio return and risk
    days_back = 30
    target_return = 0.10
    returns, cov_matrix = get_asset_data(tickers, days_back)
    optimal_weights = optimize_portfolio(returns, cov_matrix, target_return)
    _, min_risk = portfolio_return_and_risk(returns, cov_matrix, optimal_weights)

    # Compare to portfolios with randomly generated weights
    num_trials = 10000
    for _ in range(num_trials):
        random_weights = np.random.rand(num_assets)
        random_weights /= np.sum(random_weights)

        rand_return, rand_risk = portfolio_return_and_risk(returns, cov_matrix, random_weights)

        # Assert no portfolio that gives at least the target return has a lower risk
        assert not (
            (rand_risk < min_risk) and (rand_return >= target_return)
        ), f"Random portfolio found with risk {rand_risk:.4f} < optimal {min_risk:.4f} and return {rand_return:.4f} >= target {target_return:.4f}"

