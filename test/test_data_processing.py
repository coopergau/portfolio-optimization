import pandas as pd
import numpy as np
import random
from src.data_processing import get_asset_data

def test_data_returned_for_all_tickers():
    """
    Test that get_asset_data returns expected returns and covariance for all
    assets given.
    """
    # Select 10 random tickers
    tickers_df = pd.read_csv("nasdaq100.csv")
    all_tickers = tickers_df["Ticker"].tolist()
    num_assets = 10
    tickers = random.sample(all_tickers, num_assets)

    days_back = 30
    returns, cov_matrix = get_asset_data(tickers, days_back)

    assert returns.shape == (num_assets,), "Expected returns vector doesn't have the right number of returns"
    assert cov_matrix.shape == (num_assets, num_assets), "Covariance matrix doesn't have the right dimensions"

def test_covariance_matrix_is_symmetric():
    """
    Test by testing if the covariance matrix is equal to the transpose of itself.
    """
    # Select 10 random tickers
    tickers_df = pd.read_csv("nasdaq100.csv")
    all_tickers = tickers_df["Ticker"].tolist()
    num_assets = 10
    tickers = random.sample(all_tickers, num_assets)

    days_back = 30
    _, cov_matrix = get_asset_data(tickers, days_back)

    assert np.allclose(cov_matrix, cov_matrix.T), "Covariance matrix is not symmetric"
