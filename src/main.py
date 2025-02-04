# Standard Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Custom Modules
from optimization import optimize_portfolio
from data_processing import (
    get_asset_data,
    get_efficient_frontier_data,
    get_random_portfolios,
    find_similar_random_portfolio
)
from visualizations import (
    plot_random_portfolios_with_EF,
    plot_scatter,
    display_portfolio_bar_chart,
    plot_monte_carlo_all,
    plot_monte_carlo_avg
)
from portfolio_stats import portfolio_return_and_risk
from monte_carlo import brownian_simulation

# Historical data settings
TOTAL_YEARS_BACK = 3
TOTAL_DAYS_BACK = 365 * TOTAL_YEARS_BACK
ASSETS = 10

# Optimization settings
RISK_FREE_RETURN = 0.03
TARGET_RETURN = 0.15

# Efficient frontier and Sharpe ratio settings
# From 0.05 to 0.8 jumping by 0.01
MIN_RETURN = 0.05
MAX_RETURN = 0.8
NUM_RETURNS = 76

# Random portfolios settings
RANDOM_PORTFOLIOS = 1_000

# Monte Carlo simulations settings
INITIAL_VALUE = 10_000 # Dollars
TOTAL_TIME = 1 # Years
STEP_SIZE = 1 / 252 # 252 trading days in a year
SIMS = 1_000

def main():
    # Load the list of tickers on the Nasdaq 100
    # Some tickers were removed because of a lack of data on yahoo finance:
    # SPLK, FISV, SGEN
    tickers_df = pd.read_csv("../nasdaq100.csv")
    tickers = tickers_df["Ticker"].tolist()
    tickers = random.sample(tickers, ASSETS)

    # Get expected returns vector and covariance matrix
    returns, cov_matrix = get_asset_data(tickers, TOTAL_DAYS_BACK)

    # Get single minimum risk portfolio
    weights = optimize_portfolio(returns, cov_matrix, TARGET_RETURN)
    rounded_weights = np.round(weights, 3)
    title =f"Minimum Risk Portfolio with {TARGET_RETURN*100} % Expected Return"
    display_portfolio_bar_chart(rounded_weights, tickers, title)

    # Get expected portfolio return and risk
    ex_return, ex_risk = portfolio_return_and_risk(returns, cov_matrix, weights)
    # Get a random portfolio with a return that is within half a percent
    example_return, example_risk = find_similar_random_portfolio(returns, cov_matrix, ex_return)

    # Calculate Monte Carlo smiulations
    potimal_portfolio_paths = brownian_simulation(INITIAL_VALUE, ex_return, ex_risk, TOTAL_TIME, STEP_SIZE, SIMS)
    example_portfolio_paths = brownian_simulation(INITIAL_VALUE, example_return, example_risk, TOTAL_TIME, STEP_SIZE, SIMS)
    
    # Plot simulation visuals
    plot_monte_carlo_all(potimal_portfolio_paths)
    plot_monte_carlo_all(example_portfolio_paths)

    plot_monte_carlo_avg(potimal_portfolio_paths)
    plot_monte_carlo_avg(example_portfolio_paths)

    # Get the efficient fronteir data points
    ef_returns, ef_risks, ef_sharp_ratios = get_efficient_frontier_data(returns, cov_matrix, RISK_FREE_RETURN, MIN_RETURN, MAX_RETURN, NUM_RETURNS)
    random_returns, random_risks = get_random_portfolios(returns, cov_matrix, RANDOM_PORTFOLIOS)
    max_sharpe_ratio = max(ef_sharp_ratios)
    max_sharpe_ratio_index = ef_sharp_ratios.index(max_sharpe_ratio)

    plot_random_portfolios_with_EF(ef_risks, ef_returns, max_sharpe_ratio, max_sharpe_ratio_index, random_risks, random_returns)
    plot_scatter(ef_returns, ef_sharp_ratios, line=True, xlabel="Return", ylabel="Sharpe Ratio", title="Sharpe Ratio vs Return")
    plot_scatter(ef_risks, ef_sharp_ratios, line=True, xlabel="Risk", ylabel="Sharpe Ratio", title="Sharpe Ratio vs Risk")

if __name__ == "__main__":
    main()