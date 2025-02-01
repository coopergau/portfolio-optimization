import pandas as pd
import random
from data_processing import get_asset_data
from optimization import optimize_portfolio, portfolio_metrics
from visualizations import get_efficient_frontier, get_random_portfolios, plot_random_portfolios_with_EF

import numpy as np

ASSETS_PER_SAMPLE = 10
TOTAL_DAYS_BACK = 365 * 3

RISK_FREE_RETURN = 0.03
TARGET_RETURN = 0.15

MIN_RETURN = 0.05
MAX_RETURN = 0.5
NUM_RETURNS = 50

RANDOM_PORTFOLIOS = 1_000_000

def main():
    # Load the list of tickers on the Nasdaq 100
    # Some tickers were removed because of a lack of data on yahoo finance:
    # SPLK, FISV, SGEN
    tickers_df = pd.read_csv("../nasdaq100.csv")
    nasdaq_100_tickers = tickers_df["Ticker"].tolist() 
    tickers = random.sample(nasdaq_100_tickers, ASSETS_PER_SAMPLE)
    tickers = nasdaq_100_tickers

    # Get expected returns vector and covariance matrix
    returns, cov_matrix = get_asset_data(tickers, TOTAL_DAYS_BACK)
    np.set_printoptions(precision=2)
    print(tickers)
    print(returns)
    print(cov_matrix)

    # Solve using cvxpy
    weights = optimize_portfolio(returns, cov_matrix, TARGET_RETURN)
    portfolio_return, portfolio_risk = portfolio_metrics(returns, cov_matrix, weights)
    print(weights)
    rounded_matrix = np.round(weights, 2)

    print(rounded_matrix)
    print(portfolio_return)
    print(portfolio_risk)

    # Get the efficient fronteir data points
    ef_risks, ef_returns = get_efficient_frontier(returns, cov_matrix, MIN_RETURN, MAX_RETURN, NUM_RETURNS)
    random_risks, random_returns = get_random_portfolios(returns, cov_matrix, RANDOM_PORTFOLIOS)

    plot_random_portfolios_with_EF(ef_risks, ef_returns, random_risks, random_returns)

if __name__ == "__main__":
    main()