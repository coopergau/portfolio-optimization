# Standard Libraries
import pandas as pd
import numpy as np
import random

# Custom Modules
from optimization import optimize_portfolio
from data_processing import (
    get_asset_data,
    get_efficient_frontier_data,
    get_random_portfolios,
)
from visualizations import (
    plot_random_portfolios_with_EF,
    plot_scatter,
    display_portfolio_bar_chart,
    plot_monte_carlo_all,
    plot_monte_carlo_avg
)
from portfolio_stats import portfolio_return_and_risk
from monte_carlo import simulate_portfolio_returns

# Historical data settings
TOTAL_YEARS_BACK = 3
TOTAL_DAYS_BACK = 365 * TOTAL_YEARS_BACK
ASSETS = 5

# Optimization settings
RISK_FREE_RETURN = 0.03
TARGET_RETURN = 0.15 # Can be used in place of even_return if not comparing to evenly weighted portfolio

# Efficient frontier and Sharpe ratio settings
# From 0.05 to 0.8 jumping by 0.01
MIN_RETURN = 0.05
MAX_RETURN = 0.8
NUM_RETURNS = 76

# Random portfolios settings for the efficient frontier plot
RANDOM_PORTFOLIOS = 1_000

# Monte Carlo simulations settings
INITIAL_VALUE = 10_000 # Dollars
TOTAL_TIME = 1 # Years
STEP_SIZE = 1 / 252 # 252 trading days in a year
SIMS = 1_000

def main():
    # --------------- Load Data ---------------
    # Load the list of tickers on the Nasdaq 100 and randomly choose a group of them
    # Some tickers were removed because of a lack of data on yahoo finance:
    # SPLK, FISV, SGEN
    tickers_df = pd.read_csv("../nasdaq100.csv")
    tickers = tickers_df["Ticker"].tolist()
    tickers = random.sample(tickers, ASSETS)
    #tickers = ["MSFT", "AAPL", "NFLX", "NVDA"] # This line can be used to select specific tickers
    tickers = ['PDD', 'ORLY', 'TMUS', 'DLTR', 'ON'] # Good example
    print(tickers)

    # Get expected returns vector and covariance matrix
    returns, cov_matrix = get_asset_data(tickers, TOTAL_DAYS_BACK)

    # --------------- Evenly Weighted Portfolio Calculations ---------------
    # Get risk and return of evenly weighted portfolio
    even_weights = np.full(len(tickers), 1 / len(tickers))    
    even_return, even_risk = portfolio_return_and_risk(returns, cov_matrix, even_weights)
    even_weight_title = f"Evenly Weighted Portfolio with {np.round(even_return*100, 1)} % Expected Return and {np.round(even_risk*100, 1)} % Risk"
    display_portfolio_bar_chart(even_weights, tickers, even_weight_title)

    # --------------- Optimized Minimum Risk Portfolio Calculations ---------------
    # Calculate and display weights of the minimum risk portfolio
    weights = optimize_portfolio(returns, cov_matrix, even_return)
    optimized_portfolio_return, optimized_portfolio_risk = portfolio_return_and_risk(returns, cov_matrix, weights)
    rounded_weights = np.round(weights, 3)
    min_risk_title =f"Minimum Risk Portfolio with {np.round(optimized_portfolio_return*100, 1)} % Expected Return and {np.round(optimized_portfolio_risk*100, 1)} % Risk"
    display_portfolio_bar_chart(rounded_weights, tickers, min_risk_title)

    # --------------- Monte Carlo Simulations ---------------
    # Calculate Monte Carlo simulations
    optimized_portfolio_paths = simulate_portfolio_returns(INITIAL_VALUE, optimized_portfolio_return, optimized_portfolio_risk, TOTAL_TIME, STEP_SIZE, SIMS)
    even_portfolio_paths = simulate_portfolio_returns(INITIAL_VALUE, even_return, even_risk, TOTAL_TIME, STEP_SIZE, SIMS)
    
    # Plot simulation visuals
    even_portfolio_sims_title = f"Portfolio Simulations Using Even Weights"
    optimized_portfolio_sims_title = f"Portfolio Simulations Using Optimal Weights"
    plot_monte_carlo_all(even_portfolio_paths, even_portfolio_sims_title)
    plot_monte_carlo_all(optimized_portfolio_paths, optimized_portfolio_sims_title)
    
    even_portfolio_avg_title = f"Average Portfolio Performance Using Even Weights"
    optimized_portfolio_avg_title = f"Average Portfolio Performance Using Optimal Weights"
    plot_monte_carlo_avg(even_portfolio_paths, even_portfolio_avg_title)
    plot_monte_carlo_avg(optimized_portfolio_paths, optimized_portfolio_avg_title)

    # --------------- Efficient Frontier ---------------
    # Get the efficient frontier data points
    ef_returns, ef_risks, ef_sharp_ratios = get_efficient_frontier_data(returns, cov_matrix, RISK_FREE_RETURN, MIN_RETURN, MAX_RETURN, NUM_RETURNS)
    max_sharpe_ratio = max(ef_sharp_ratios)
    max_sharpe_ratio_index = ef_sharp_ratios.index(max_sharpe_ratio)
    # Get random portfolios
    random_returns, random_risks = get_random_portfolios(returns, cov_matrix, RANDOM_PORTFOLIOS)

    plot_random_portfolios_with_EF(ef_risks, ef_returns, max_sharpe_ratio, max_sharpe_ratio_index, random_risks, random_returns)
    plot_scatter(ef_returns, ef_sharp_ratios, line=True, xlabel="Return", ylabel="Sharpe Ratio", title="Sharpe Ratio vs Return")
    plot_scatter(ef_risks, ef_sharp_ratios, line=True, xlabel="Risk", ylabel="Sharpe Ratio", title="Sharpe Ratio vs Risk")

if __name__ == "__main__":
    main()