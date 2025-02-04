import numpy as np
import pytest
from src.portfolio_stats import portfolio_return_and_risk, portfolio_sharpe_ratio

def test_portfolio_metrics_are_accurate():
    """
    Test that the expected return and risk for a portfolio are accurate.
    """
    returns = np.array([0.1, 0.2])
    cov_matrix = np.array([[0.2, 0.05],
                           [0.05, 0.1]])
    weights = np.array([0.25, 0.75])

    # Pre calculated values based on the vars above
    intended_return = 0.175
    intended_risk = np.sqrt(0.0875)

    actual_return, actual_risk = portfolio_return_and_risk(returns, cov_matrix, weights)

    assert np.allclose(intended_return, actual_return, atol=1e-8), "Portfolio return calculation gives incorrect return"
    assert np.allclose(intended_risk, actual_risk, atol=1e-8), "Portfolio risk calculation gives incorrect risk"

def test_sharpe_ratio_with_zero_risk_errors():
    """
    Calling portfolio_sharpe_ratio and providing and portfolio_risk that is zero should return an error.
    """
    valid_return = 0.10
    invalid_risk = 0
    valid_risk_free_rate = 0.02
    with pytest.raises(ValueError):
        portfolio_sharpe_ratio(valid_return, invalid_risk, valid_risk_free_rate)

def test_basic_sharpe_ratio_is_correct():
    portfolio_return = 0.13
    portfolio_risk = 0.20
    risk_free_rate = 0.03

    expected_sharpe_ratio = 0.5
    actual_sharpe_ratio = portfolio_sharpe_ratio(portfolio_return, portfolio_risk, risk_free_rate)

    assert expected_sharpe_ratio == actual_sharpe_ratio, "Resulting Sharpe ratio is not as expected"