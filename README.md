# Portfolio Optimization Using Markowitz Theory
## Overview
This project uses Markowitz Portfolio Theory and convex optimization to determine the ideal asset allocations based on risk and return. The optimizer finds the minimum-risk portfolio that meets a target return while visualizing key insights, such as the Efficient Frontier, Monte Carlo simulations, and Sharpe Ratio comparisons.

## Features
For a selected set of assets, the project provides:

## 1. Portfolio Weight Comparisons
   * First, calculates an evenly weighted portfolio as a baseline.  
   * Then, optimizes for the lowest-risk portfolio that achieves the same expected return.  
   * Visualizes the weight distributions in a stacked bar chart and compares risk metrics.

## 2. Efficeint Frontier Approximation
   * Computes minimum risk portfolios for a series of different rates of return and plots them on a return vs. risk graph.
   * Uses the minimum risk portfolios to interpolate and draw the efficient frontier.
   * Identifies the portfolio with maximum Sharpe ratio on the graph.
   * Computes and plots a large number of randomly weighted portfolios to help visulaize the meaning behind the efficient frontier.

## 3. Sharpe Ratio Graphs
   * Computes the Sharpe ratios of the minimum risk portfolios and graphs them vs. risk and vs. return.

## 4. Monte Carlo Simulations
   * Simulates thousands of possible portfolio value paths to compare potential performance of different portfolios.
   * Used to identify realistic worst case scenarios to assess portfolio risk exposure.


What do we have
    - For a selected group of tickers:
        - Optimizer that finds weights for portfolio with min risk that gives at least a target return (bar display of weights)
        - Efficeint fronteir graph: EF, random portfolios, and Max Sharpe Ratio point highlighted
        - Monte Carlo sims of portfolios
            - Simulate the optimal porfolio paths and other ones with similar return to compare worst case paths
            - Can compare avg paths but they end up being very similar 
        - Graph of Sharpe Ratio vs return 
        - Graph of Sharpe Ratio vs. risk

    - Do some comparisons of 10 random assets: optimal vs evenly weights, and/or optimal vs randomly weighted
    - Compare picking from the whole nasdaq (or like top 20 biggest companies on nasdaq) vs the qqq index
    - mention that yfinance needs to be up to date for it to work