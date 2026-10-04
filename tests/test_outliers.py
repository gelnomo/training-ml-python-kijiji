import numpy as np
import pandas as pd
import pytest

from modules.enums import OutlierTypeEnum
from modules.outliers import get_outliers, outlier_summary


@pytest.fixture
def values() -> pd.Series:
    rng = np.random.default_rng(1)
    data = pd.Series(rng.normal(100, 10, 1000))
    data.iloc[[10, 20]] = [1_000, -800]
    data.iloc[30] = np.nan
    return data


@pytest.mark.parametrize("method", list(OutlierTypeEnum))
def test_every_method_flags_extremes(values, method):
    mask = get_outliers(values, method)

    assert mask.dtype == bool
    assert mask.index.equals(values.index)
    assert mask.iloc[10] and mask.iloc[20]
    assert not mask.iloc[30]  # missing values are never outliers


def test_constant_series_has_no_outliers():
    mask = get_outliers(pd.Series([5.0] * 50), OutlierTypeEnum.Z_SCORE)
    assert not mask.any()


def test_outlier_summary_has_one_row_per_method(values):
    summary = outlier_summary(values)
    assert list(summary["method"]) == [m.name for m in OutlierTypeEnum]
    assert (summary["outliers"] >= 2).all()
