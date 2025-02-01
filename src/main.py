import numpy as np
import pandas as pd
from data_processing import get_asset_data
from optimization import optimize_portfolio, portfolio_metrics
from visualizations import get_efficient_frontier, get_random_portfolios, plot_random_portfolios_with_EF

import random

TOTAL_DAYS_BACK = 365 * 3
RISK_FREE_RETURN = 0.03
TARGET_RETURN = 0.15

# Load the list of tickers on the Nasdaq 100
# Some tickers were removed because of a lack of data on yahoo finance:
# SPLK, FISV, SGEN
tickers_df = pd.read_csv("../nasdaq100.csv")
nasdaq_100_tickers = tickers_df["Ticker"].tolist()

ASSETS_PER_SAMPLE = 10

def main():
    tickers = random.sample(nasdaq_100_tickers, ASSETS_PER_SAMPLE)

    # Get expected returns vector and covariance matrix
    returns, cov_matrix = get_asset_data(tickers, TOTAL_DAYS_BACK)

    # Solve using cvxpy
    weights = optimize_portfolio(returns, cov_matrix, TARGET_RETURN)
    portfolio_return, portfolio_risk = portfolio_metrics(returns, cov_matrix, weights)
    print(portfolio_return)
    print(portfolio_risk)

    # Get the efficient fronteir data points
    MIN_RETURN = 0.05
    MAX_RETURN = 0.50
    NUM_RETURNS = 50

    random_portfolios = 1000000

    ef_risks, ef_returns = get_efficient_frontier(returns, cov_matrix, MIN_RETURN, MAX_RETURN, NUM_RETURNS)
    random_risks, random_returns = get_random_portfolios(returns, cov_matrix, random_portfolios)

    plot_random_portfolios_with_EF(ef_risks, ef_returns, random_risks, random_returns)

if __name__ == "__main__":
    main()