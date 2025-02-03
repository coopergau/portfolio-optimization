import pandas as pd
from data_processing import get_asset_data
from optimization import optimize_portfolio
from portfolio_stats import portfolio_return_and_risk
from visualizations import get_efficient_frontier_data, get_random_portfolios, plot_random_portfolios_with_EF, plot_scatter 

import random

# Historical data settings
TOTAL_YEARS_BACK = 3
TOTAL_DAYS_BACK = 365 * TOTAL_YEARS_BACK

# Optimization settings
RISK_FREE_RETURN = 0.03
TARGET_RETURN = 0.15

# Efficient frontier and Sharpe ratio settings
# From 0.05 to 0.8 jumping by 0.015
MIN_RETURN = 0.05
MAX_RETURN = 0.8
NUM_RETURNS = 51

# Visualizing portfolios settings
RANDOM_PORTFOLIOS = 1_000_000

def main():
    # Load the list of tickers on the Nasdaq 100
    # Some tickers were removed because of a lack of data on yahoo finance:
    # SPLK, FISV, SGEN
    tickers_df = pd.read_csv("../nasdaq100.csv")
    tickers = tickers_df["Ticker"].tolist()
    tickers = random.sample(tickers, 20)

    # Get expected returns vector and covariance matrix
    returns, cov_matrix = get_asset_data(tickers, TOTAL_DAYS_BACK)

    # Solve using cvxpy
    weights = optimize_portfolio(returns, cov_matrix, TARGET_RETURN)
    portfolio_return, portfolio_risk = portfolio_return_and_risk(returns, cov_matrix, weights)

    # Get the efficient fronteir data points
    ef_returns, ef_risks, ef_sharp_ratios = get_efficient_frontier_data(returns, cov_matrix, RISK_FREE_RETURN, MIN_RETURN, MAX_RETURN, NUM_RETURNS)
    random_returns, random_risks = get_random_portfolios(returns, cov_matrix, RANDOM_PORTFOLIOS)

    plot_random_portfolios_with_EF(ef_risks, ef_returns, random_risks, random_returns)
    plot_scatter(ef_returns, ef_sharp_ratios, xlabel="Return", ylabel="Sharpe Ratio")
    plot_scatter(ef_risks, ef_sharp_ratios, xlabel="Risk", ylabel="Sharpe Ratio")

if __name__ == "__main__":
    main()