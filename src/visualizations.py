import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    fake_handle = Line2D([], [], color='none', label=max_point_label)
    plt.legend(handles=[fake_handle], loc='best', handletextpad=0.1, borderpad=0.2, frameon=False)

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
