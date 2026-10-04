"""Candidate regressors. Every model predicts log1p(visits) and is scored on visits."""

from typing import Dict

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from modules.config import RANDOM_STATE
from modules.features import build_preprocessor


def _log_target(regressor) -> TransformedTargetRegressor:
    # Visits are heavily right-skewed (median 75, max 54k). Fitting on log1p keeps a
    # few viral ads from dominating the loss; predictions are mapped back with expm1.
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def get_models(fast: bool = False) -> Dict[str, Pipeline]:
    """
    Build the candidate pipelines (preprocessing + regressor)
    :param fast: smaller ensembles, for quick runs and tests
    :return: dict name -> unfitted pipeline
    """
    n_trees = 50 if fast else 300
    return {
        # Baseline: always predicts the median. Any useful model must beat it.
        "Baseline (median)": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=True)),
                ("model", _log_target(Ridge(alpha=1.0))),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    _log_target(
                        RandomForestRegressor(
                            n_estimators=n_trees // 2,
                            min_samples_leaf=3,
                            max_features=0.3,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        )
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    _log_target(
                        HistGradientBoostingRegressor(
                            max_iter=n_trees,
                            learning_rate=0.08,
                            max_leaf_nodes=31,
                            l2_regularization=1.0,
                            random_state=RANDOM_STATE,
                        )
                    ),
                ),
            ]
        ),
    }
