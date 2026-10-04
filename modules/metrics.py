from typing import Dict

import numpy as np
from sklearn import metrics


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Standard regression metrics on the original scale of the target
    :param y_true: observed values
    :param y_pred: predicted values
    :return: dict with MAE, MedAE, RMSE, R2 and R2 on log1p scale
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MedAE": metrics.median_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
        "R2": metrics.r2_score(y_true, y_pred),
        # Visits are heavy-tailed: R2 on log scale tells how well the order of
        # magnitude is captured, which the raw R2 (dominated by a few viral ads) hides.
        "R2_log": metrics.r2_score(np.log1p(y_true), np.log1p(y_pred)),
    }
