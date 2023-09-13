import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def get_arima_model(dataframe, column_to_predict, depth: int = 50):
    """
    Fit an ARIMA model to a time series and make predictions.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing time series data.
        column_to_predict (str): The name of the column in the DataFrame to predict.
        depth (int, optional): The number of time steps to forecast into the future. Default is 50.

    Returns:
        statsmodels.tsa.arima_model.ARIMAResultsWrapper: The trained ARIMA model.

    This function fits an ARIMA (AutoRegressive Integrated Moving Average) model to a given time series data.
    It then makes predictions for the specified number of time steps into the future and visualizes the results.

    Example:
        To fit an ARIMA model and make predictions:
        df = pd.DataFrame(...)  # Your time series data DataFrame
        column_name = 'Price'  # The column to predict
        depth = 50  # Number of time steps to forecast
        arima_model = get_arima_model(df, column_name, depth)
    """

    data_len = dataframe.shape[0]
    if depth > data_len:
        depth = data_len

    dataset = dataframe.tail(data_len)
    data = dataset

    dataset = dataset[column_to_predict].tolist()
    data = data[column_to_predict].tolist()

    model = ARIMA(dataset, order=(40, 0, 0)).fit()
    print("MAE: ", model.mae)
    print("MSE: ", model.mse)

    start = len(dataset) - 1
    finish = len(dataset) + depth - 1

    prediction = model.predict(start, finish, dynamic=True).tolist()

    print("Current price: ", dataset[-1])
    print("Prediction: ", prediction[-1])
    print("Prediction high: ", np.max(prediction))
    print("Prediction low: ", np.min(prediction))

    plt.plot(np.arange(start, finish + 1).tolist(), prediction, "g--")
    plt.plot(data)
    plt.show()
    return model
