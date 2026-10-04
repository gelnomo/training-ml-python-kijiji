"""
Univariate outlier detectors. Each returns a boolean mask aligned with the input
(True = outlier), so results can be combined, counted or used for filtering.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from modules.config import RANDOM_STATE
from modules.enums import OutlierTypeEnum


def z_score_outliers(values: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Classic Z score: distance to the mean in standard deviations. Assumes a roughly
    normal distribution, and the mean/std are themselves inflated by the outliers.
    """
    std = values.std(ddof=0)
    if not std:
        return pd.Series(False, index=values.index)
    return ((values - values.mean()) / std).abs() > threshold


def robust_z_score_outliers(values: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    Modified Z score: uses the median and the median absolute deviation (MAD) instead of
    mean/std, so it is not distorted by the outliers it is looking for.
    0.6745 makes the MAD comparable to a standard deviation under normality.
    """
    mad = stats.median_abs_deviation(values)
    if not mad:
        return pd.Series(False, index=values.index)
    return (0.6745 * (values - values.median()) / mad).abs() > threshold


def iqr_outliers(values: pd.Series, k: float = 1.5) -> pd.Series:
    """Tukey fences: below Q1 - k*IQR or above Q3 + k*IQR (box-plot whiskers)."""
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (values < q1 - k * iqr) | (values > q3 + k * iqr)


def percentile_outliers(values: pd.Series, lower: float = 1, upper: float = 99) -> pd.Series:
    """Values outside the [lower, upper] percentiles (the values winsorization would cap)."""
    low, high = np.percentile(values, [lower, upper])
    return (values < low) | (values > high)


def dbscan_outliers(values: pd.Series, eps: float = 0.5, min_samples: int = 5) -> pd.Series:
    """
    Density based: points DBSCAN cannot attach to any dense cluster get label -1.
    Run on standardized values so ``eps`` is expressed in standard deviations.
    """
    scaled = (values - values.mean()) / (values.std(ddof=0) or 1)
    # Cluster the distinct values weighted by their frequency: same labels as clustering
    # every row, without the O(n^2) memory of 50k points sharing a few hundred values.
    counts = scaled.value_counts()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
        counts.index.to_numpy().reshape(-1, 1), sample_weight=counts.to_numpy()
    )
    is_noise = pd.Series(labels == -1, index=counts.index)
    return pd.Series(scaled.map(is_noise).to_numpy(), index=values.index)


def isolation_forest_outliers(values: pd.Series, contamination="auto") -> pd.Series:
    """
    Isolation Forest: points isolated by few random splits are anomalous
    (``fit_predict`` returns -1 for them).
    """
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    labels = model.fit_predict(values.to_numpy().reshape(-1, 1))
    return pd.Series(labels == -1, index=values.index)


DETECTORS = {
    OutlierTypeEnum.Z_SCORE: z_score_outliers,
    OutlierTypeEnum.ROBUST_Z_SCORE: robust_z_score_outliers,
    OutlierTypeEnum.IQR: iqr_outliers,
    OutlierTypeEnum.PERCENTILE: percentile_outliers,
    OutlierTypeEnum.DBSCAN: dbscan_outliers,
    OutlierTypeEnum.ISOLATION_FOREST: isolation_forest_outliers,
}


def get_outliers(values: pd.Series, outlier_type: OutlierTypeEnum) -> pd.Series:
    """
    Detect outliers with the chosen method, ignoring missing values
    :param values: numeric series
    :param outlier_type: detection method
    :return: boolean mask aligned with ``values`` (NaN rows are never outliers)
    """
    mask = pd.Series(False, index=values.index)
    clean = values.dropna()
    if len(clean) > 1:
        mask.loc[clean.index] = DETECTORS[outlier_type](clean)
    return mask


def outlier_summary(values: pd.Series) -> pd.DataFrame:
    """
    Compare how many points every method flags
    :param values: numeric series
    :return: dataframe with count, share and the value range each method keeps
    """
    rows = []
    for outlier_type in OutlierTypeEnum:
        mask = get_outliers(values, outlier_type)
        kept = values[~mask & values.notna()]
        rows.append(
            {
                "method": outlier_type.name,
                "outliers": int(mask.sum()),
                "%outliers": round(100 * mask.mean(), 2),
                "kept_min": kept.min(),
                "kept_max": kept.max(),
            }
        )
    return pd.DataFrame(rows)
