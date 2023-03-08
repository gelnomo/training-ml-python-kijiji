import numpy as np
from sklearn import metrics


def get_metrics(data, predict_value, columns_metrics, columns_score):
    y_orig = data.to_frame().transpose()
    y_orig.drop(columns=["make"] + columns_metrics + columns_score, inplace=True)
    y_predict = y_orig.copy()
    y_predict["price"] = predict_value
    y_predict["price"] = int(round(y_predict["price"]))
    y_orig = y_orig.to_numpy()[0].tolist()
    y_predict = y_predict.to_numpy()[0].tolist()

    # Result of MAE
    mae = metrics.mean_absolute_error(y_orig, y_predict)
    # Result of MSE
    mse = metrics.mean_squared_error(y_orig, y_predict)
    # Result of RMSE
    r2s = np.sqrt(metrics.mean_squared_error(y_orig, y_predict))
    # Accuracy
    accuracy = metrics.accuracy_score(y_orig, y_predict)

    return mae, mse, r2s, accuracy
