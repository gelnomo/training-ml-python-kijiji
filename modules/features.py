"""Feature lists and preprocessing pipelines shared by every model."""

from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from modules.cleansing import FLAG_COLUMNS

NUMERIC_FEATURES: List[str] = [
    "price",
    "price_missing",
    "Kilometers",
    "car_age",
    "km_per_year",
    "ad_age_days",
    "No of door",
    "No of seats",
    "title_length",
    *FLAG_COLUMNS,
]

CATEGORICAL_FEATURES: List[str] = [
    "city",
    "saleby",
    "make",
    "model",
    "condition",
    "body",
    "Drivetrain",
    "fueltype",
    "color",
    "Transmission",
    "publish_weekday",
]

FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """
    Impute and encode the raw features. Fitted inside each model pipeline, so the
    statistics (medians, scaler, categories) are learnt from training folds only.
    :param scale_numeric: log1p + standardize numeric columns (needed by linear models only;
        price, km and ad age are heavily skewed and all non-negative)
    :return: unfitted ColumnTransformer
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps += [("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]

    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encode",
                # Rare categories (e.g. most of the ~740 car models) are pooled
                # into one "infrequent" column to avoid overfitting.
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=30,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
