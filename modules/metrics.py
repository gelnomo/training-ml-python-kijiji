import numpy as np
import pandas as pd
from sklearn import metrics
from typing import List, Tuple


def get_metrics(
    target: str, data: pd.DataFrame, predict_value: str, exclude_col_metric: List
) -> Tuple[float, float, float]:
    """
    Generate metrics MAE, MSE, RMSE
    :param target:
    :param data:
    :param predict_value:
    :param exclude_col_metric:
    :return:
    """
    y_orig = data.to_frame().transpose()
    y_orig.drop(columns=exclude_col_metric, inplace=True)
    y_predict = y_orig.copy()
    y_predict[target] = predict_value
    y_orig = y_orig.to_numpy()[0].tolist()
    y_predict = y_predict.to_numpy()[0].tolist()

    # Result of MAE
    mae = metrics.mean_absolute_error(y_orig, y_predict)
    # Result of MSE
    mse = metrics.mean_squared_error(y_orig, y_predict)
    # Result of RMSE
    r2s = np.sqrt(metrics.mean_squared_error(y_orig, y_predict))

    return mae, mse, r2s


def metric_variance(data: pd.DataFrame, column_name: str) -> float:
    """
    Get variance value by column
    :param data: initial dataframe
    :param column_name: column name to get the variance
    :return: float value
    """
    return data[column_name].var()


def metric_standard_deviation(data: pd.DataFrame, column_name: str) -> float:
    """
    Get std value by column
    :param data: initial dataframe
    :param column_name: column name to get the standard deviation
    :return: float value
    """
    return data[column_name].std()
