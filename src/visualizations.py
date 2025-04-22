import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def plot_scatter(x, y, line=False, xlabel="X-axis", ylabel="Y-axis", title="Scatter Plot", color="blue"):
    """
    General scatter plot function for return-risk or Sharpe ratio-return plots. Plots dashed lines identifying
    the max y value point.
    
    Args:
    - x (list): Data for the x-axis.
    - y (list): Data for the y-axis.
    - line (bool): Setting to True will plot a line connecting all points.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - title (str): Title of the graph.
    - color (str): Color of the scatter points.
    """
    # Find the max y value
    max_y = max(y)
    max_index = y.index(max_y)
    max_x = x[max_index]

    # Plot and label max y value info
    v_line = Line2D([max_x, max_x], [0, max_y], color='red', linestyle='--')
    h_line = Line2D([0, max_x], [max_y, max_y], color='red', linestyle='--')
    plt.gca().add_line(v_line)
    plt.gca().add_line(h_line)
    max_point_label = f'Max {ylabel} Point: ({round(max_x, 2)}, {round(max_y, 2)})'
    max_point_handle = Line2D([], [], color='none', label=max_point_label)
    plt.legend(handles=[max_point_handle], loc='best', handletextpad=0.1, borderpad=0.2, frameon=False)

    # Plot data
    plt.ylim(0, max_y * 1.2)
    plt.xlim(0, max(x) * 1.1)
    plt.scatter(x, y, c=color)
    if line:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_random_portfolios_with_EF(ef_risks, ef_returns, max_sharpe_ratio, max_sharpe_ratio_index, random_risks, random_returns):
    """
    Plots the efficient frontier line and points, and randomly generated portfolio points. Colours
    the portfolio on the EF with max Sharpe ratio as black.
    """
    # Random portfolios
    plt.scatter(random_risks, random_returns, c="blue", label="Random Portfolios")
    
    # Efficient frontier
    plt.scatter(ef_risks, ef_returns, c="orange")
    plt.plot(ef_risks, ef_returns, "orange", label="Efficient Frontier", zorder=1)

    # Max Sharpe ratio point
    plt.scatter(ef_risks[max_sharpe_ratio_index], ef_returns[max_sharpe_ratio_index], c="black", label=f"Max Sharpe Ratio {round(max_sharpe_ratio, 2)}", zorder=1)

    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.show()

def display_portfolio_bar_chart(weights, assets, title):
    """
    Displays a stacked bar chart showing the proportions of each asset in a ortfolio
    """
    # Get random distinct colours
    cmap = plt.get_cmap('tab20')  # Good for up to 20 distinct colors
    colours = [cmap(i) for i in np.linspace(0, 1, len(assets))]

    # Create placeholder bar
    height = 1
    plt.bar([""], [height], color="white")

    # Add each asset with a label of the ticker and portfolio proportion
    plt.bar([""], weights, bottom=np.cumsum([0] + list(weights[:-1])), color=colours,
            label=[f"{asset} ({abs(weight)*100:.1f}%)" for asset, weight in zip(assets, weights)])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title="Asset Weights", bbox_to_anchor=(1, 1), loc='best')
    plt.title(title)
    plt.show()

def plot_monte_carlo_all(paths, title):
    """
    Plot the lines of all simulated portfolio paths.
    """
    final_values = paths[:, -1]
    lowest_final_value = np.min(final_values)
    worst_performance_label = f"Worst Performing Portfolio Value: ${np.round(lowest_final_value, 0)} "
    plt.hlines(y=lowest_final_value, xmin=0, xmax=252, color='red', linestyle='--', label=worst_performance_label)
    plt.legend(loc="lower left")

    plt.plot(paths.T)
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value (Dollars)")
    plt.title(title)
    plt.show()

def plot_monte_carlo_avg(paths, title):
    """
    Plot the avrage of all the portfolio paths.
    """
    average_path = np.mean(paths, axis=0)
    plt.plot(average_path)
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value")
    plt.title(title)
    plt.show()