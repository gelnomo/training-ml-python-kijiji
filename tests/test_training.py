import numpy as np

from modules.cleansing import clean_data
from modules.enums import OutputTypeEnum
from modules.metrics import regression_metrics
from modules.training import TrainingData


def test_regression_metrics_perfect_prediction():
    result = regression_metrics([0, 10, 100], [0, 10, 100])
    assert result["MAE"] == 0 and result["RMSE"] == 0
    assert result["R2"] == 1 and result["R2_log"] == 1


def test_regression_metrics_clips_negative_predictions():
    assert regression_metrics([0, 0], [-5, -5])["MAE"] == 0


def test_training_beats_baseline(raw_ads, tmp_path, monkeypatch):
    monkeypatch.setattr("modules.training.storage_sqlite", lambda *args, **kwargs: None)
    df, _ = clean_data(raw_ads)

    result = TrainingData(df, OutputTypeEnum.CONSOLE, plots=False, fast=True).process()

    assert result["best_name"] != "Baseline (median)"
    metrics = result["test_metrics"]
    assert metrics.loc[result["best_name"], "MAE"] < metrics.loc["Baseline (median)", "MAE"]
    assert result["importance"].index[0] in {"ad_age_days", "car_age"}
    assert np.all(result["model"].predict(df.head()) >= -1)
