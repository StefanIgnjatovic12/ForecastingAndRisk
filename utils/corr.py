import pandas as pd
import numpy as np


def get_portfolio(dataframe, columns_to_evaluate):
    """
    Calculate portfolio statistics based on correlations.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing historical data for assets.
        columns_to_evaluate (list): A list of column names (assets) to evaluate.

    Returns:
        pd.DataFrame: A DataFrame containing the optimized portfolio weights for each asset.
        float: The total correlation percent.

    This function calculates portfolio statistics based on correlations between assets in the provided DataFrame.
    It calculates the optimized portfolio weights for each asset and the total correlation percent.

    Example:
        To calculate portfolio statistics:
        df = pd.DataFrame(...)  # Your historical data DataFrame
        assets = ['Asset1', 'Asset2', 'Asset3']  # List of asset names
        portfolio_weights, total_corr_percent = get_portfolio(df, assets)
    """
    hist_dict = {}
    for ticker in columns_to_evaluate:
        hist = dataframe[columns_to_evaluate]
        hist_dict.update({ticker: hist})
    portfolio = dataframe.ffill().copy()
    correlations = portfolio.corr(method='pearson')
    percents = {}
    percents_sum = 0

    total_correlation_percent = (
        np.round(correlations.sum().sum()
                 / np.power(portfolio.shape[0], 2) * 100, 2)
    )

    for ticker in correlations.columns:
        correlation_sum = correlations[ticker].sum()
        value = np.power((1 / correlation_sum) * 100, 2)
        percents_sum += value
        percents.update({ticker: value})

    for key in percents.keys():
        percents[key] *= 100 / percents_sum
        percents[key] = np.round(percents[key], decimals=2)

    print("\ntotal_correlation_percent: ", total_correlation_percent, "%\n")
    print("optimized_portfolio: ", pd.DataFrame(percents, index=[0]))

    return pd.DataFrame(percents, index=[0]), total_correlation_percent

