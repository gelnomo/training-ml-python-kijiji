"""Model selection with cross-validation, held-out evaluation and model persistence."""

import json
import time
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from modules.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE, TARGET, TEST_SIZE
from modules.enums import OutputTypeEnum
from modules.features import FEATURES
from modules.metrics import regression_metrics
from modules.models import get_models
from modules.output_data import OutputResult
from modules.utils import storage_sqlite
from modules.visualization import Visualization


def _r2_log(y_true, y_pred):
    return r2_score(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))


SCORING = {
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
    "R2_log": make_scorer(_r2_log),
}


class TrainingData:
    """Compare candidate models, pick the best by CV and evaluate it once on the test set."""

    def __init__(self, data: pd.DataFrame, output: OutputTypeEnum, save_model: bool = False,
                 plots: bool = True, fast: bool = False):
        self.data = data
        self.output_result = OutputResult(output)
        self.save_model = save_model
        self.graphs = Visualization() if plots else None
        self.fast = fast

    def _report(self, title: str, content) -> None:
        self.output_result.title = title
        self.output_result.process(content)

    def _cross_validate(self, models: Dict, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        folds = KFold(n_splits=3 if self.fast else CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        rows = []
        for name, model in models.items():
            print(f"Cross-validating: {name}")
            start = time.perf_counter()
            scores = cross_validate(model, x_train, y_train, cv=folds, scoring=SCORING)
            rows.append(
                {
                    "model": name,
                    "MAE": -scores["test_MAE"].mean(),
                    "MAE_std": scores["test_MAE"].std(),
                    "R2": scores["test_R2"].mean(),
                    "R2_log": scores["test_R2_log"].mean(),
                    "seconds": round(time.perf_counter() - start, 1),
                }
            )
        return pd.DataFrame(rows).set_index("model").sort_values("MAE")

    def _importance(self, model, x_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
        print("Computing: permutation importance")
        sample = x_test.sample(n=min(3000, len(x_test)), random_state=RANDOM_STATE)
        result = permutation_importance(model, sample, y_test.loc[sample.index], scoring="neg_mean_absolute_error",
                                        n_repeats=3 if self.fast else 5, random_state=RANDOM_STATE)
        return pd.Series(result.importances_mean, index=sample.columns).sort_values(ascending=False)

    def process(self) -> Dict:
        """
        Train, select and evaluate
        :return: dict with CV table, test metrics, chosen model name and fitted pipeline
        """
        x, y = self.data[FEATURES], self.data[TARGET]
        # Stratify on visit deciles so train and test share the same skewed distribution
        strata = pd.qcut(np.log1p(y), q=10, labels=False, duplicates="drop")
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata
        )
        self._report("Train / test sizes", {"train": len(x_train), "test": len(x_test)})

        models = get_models(fast=self.fast)
        cv_results = self._cross_validate(models, x_train, y_train)
        self._report("Cross-validation on the training set (mean of folds)", cv_results.round(3))

        # Selection uses CV only; the test set is touched once, for the final estimate
        best_name = cv_results.drop(index="Baseline (median)").index[0]
        print(f"Fitting best model on the full training set: {best_name}")
        best_model = models[best_name].fit(x_train, y_train)
        baseline = models["Baseline (median)"].fit(x_train, y_train)

        y_pred = best_model.predict(x_test)
        test_metrics = pd.DataFrame(
            {
                best_name: regression_metrics(y_test, y_pred),
                "Baseline (median)": regression_metrics(y_test, baseline.predict(x_test)),
            }
        ).T
        self._report("Held-out test metrics", test_metrics.round(3))

        predictions = x_test[["city", "make", "model", "price", "ad_age_days"]].copy()
        predictions[TARGET] = y_test
        predictions["visit_predicted"] = np.clip(y_pred, 0, None).round()
        predictions["abs_error"] = (predictions[TARGET] - predictions["visit_predicted"]).abs()
        by_city = predictions.groupby("city").agg(
            ads=(TARGET, "size"), median_visits=(TARGET, "median"), MAE=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
        ).sort_values("ads", ascending=False)
        self._report("Test error by city", by_city.round(1))

        importance = self._importance(best_model, x_test, y_test)
        self._report("Permutation importance (MAE increase when a feature is shuffled)", importance.round(2))

        if self.graphs:
            self.graphs.bar(cv_results["MAE"], "Model comparison, cross-validated MAE (lower is better)",
                            "Mean absolute error (visits)", "", horizontal=True, xerr=cv_results["MAE_std"])
            self.graphs.predicted_vs_actual(y_test, y_pred, f"{best_name}: predicted vs actual visits (test set)")
            self.graphs.residuals(y_test, y_pred, f"{best_name}: residuals (test set)")
            self.graphs.bar(importance.head(12), "Most important features (permutation importance)",
                            "Increase in MAE when shuffled (visits)", "", horizontal=True, label_fmt="{:,.1f}")

        storage_sqlite(predictions.reset_index(names="row_id"), "PREDICTED")
        if self.save_model:
            self._save(best_model, best_name, test_metrics.loc[best_name].to_dict())

        return {
            "cv_results": cv_results,
            "test_metrics": test_metrics,
            "best_name": best_name,
            "model": best_model,
            "importance": importance,
        }

    def _save(self, model, name: str, metrics: Dict) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / "visit_model.joblib")
        metadata = {
            "model": name,
            "target": TARGET,
            "features": FEATURES,
            "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "scikit_learn_version": sklearn.__version__,
            "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        (MODELS_DIR / "visit_model.json").write_text(json.dumps(metadata, indent=2))
        print(f"Model saved to {MODELS_DIR / 'visit_model.joblib'}")
