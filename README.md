# Portfolio Optimization Using Markowitz Theory
## Overview
This project uses Modern Portfolio Theory and convex optimization to find the ideal asset allocations based on risk and return. The optimizer calculates the minimum-risk portfolio that meets a target expected return while visualizing key insights, such as the Efficient Frontier, portfolio performance simulations, and Sharpe Ratio behaviour.

## Features
For a selected set of assets, the project provides:

### 1. Portfolio Weight Comparisons
   * First, calculates an evenly weighted portfolio as a baseline.  
   * Then, optimizes for the lowest-risk portfolio that achieves the same (or better) expected return.  
   * Visualizes the weight distributions in a stacked bar chart and compares risk metrics.

### 2. Monte Carlo Simulations
   * Simulates thousands of possible portfolio value paths to compare the potential performance of the baseline and the optimized portfolios.
   * Used to identify realistic worst case scenarios to assess portfolio risk exposure.
   * Also calculates the average performance of the two portfolios.

### 3. Efficeint Frontier Approximation
   * Computes minimum risk portfolios for a series of different rates of return and plots them on a return vs. risk graph.
   * Uses these minimum risk portfolios to interpolate and draw the efficient frontier.
   * Computes and plots a large number of randomly weighted portfolios to help visulaize the meaning behind the efficient frontier.
   * Identifies the portfolio with maximum Sharpe ratio on the graph.

### 4. Sharpe Ratio Visuals
   * Computes the Sharpe ratios of the minimum risk portfolios and plots them vs. risk and vs. return.
   * Used to analyze the behaviour of the Sharpe ratio as risk and return vary.

## Requirements  
- Python 3.x  
- Dependencies:  
  - `yfinance` (Ensure package is up to date for proper api usage)  
  - `numpy`, `pandas`, `cvxpy`, `matplotlib`

## How to Run  

1. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the optimization script:**
   ```bash
   python3 main.py
   ```

3. **Adjust parameters as necessary:**
- To choose specific tickers, in main.py replace
   ```python
   tickers = random.sample(tickers, ASSETS)
   ```
   with

   ```python
   tickers = ["MSFT", "AAPL", "NFLX", "NVDA"] # Or whatever combination of assets you want
   ```

## Example

For this example we will use the tickers ['PDD', 'ORLY', 'TMUS', 'DLTR', 'ON'].

### 1. Portfolio Weight Comparisons

<img src="images/even_weights.PNG", width="300" />
<img src="images/min_risk_weights.PNG", width="300" />

### 2. Monte Carlo Simulations

<img src="images/even_sim.PNG", width="300" />
<img src="images/min_risk_sim.PNG", width="300" />

<img src="images/even_avg.PNG", width="300" />
<img src="images/min_risk_avg.PNG", width="300" />
   
### 3. Efficeint Frontier Approximation

<img src="images/ef.PNG", width="300" />
   
### 4. Sharpe Ratio Visuals
<img src="images/sharpe_vs_return.PNG", width="300" />
<img src="images/sharpe_vs_risk.PNG", width="300" />

## Theory


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

# Things to note
The solver aim is to minimze risk while having expected return be at least the target given return, so it often gives a portfolio that has a return that is actually higher than the evenly weighted portfolio