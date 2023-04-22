import joblib
import numpy as np
import pandas as pd
from typing import List

from sklearn.preprocessing import StandardScaler

from modules.enums import OutputTypeEnum, OptionEnum
from modules.utils import dummify, undummify, storage_sqlite
from tqdm import tqdm
from modules.output_data import OutputResult
from models.gradient_boostring_regressor.train_model import (
    train_gradient_boosting_regressor,
)
from models.linear_regression.train_model import train_linear_regression
from modules.metrics import get_metrics
from models.random_forest_regressor.train_model import train_random_forest_regressor
from modules.visualization import Visualization


class TrainingData:
    train_size = 0.6
    random_state = 42

    # Score train/test Columns
    columns_score = [
        "lr score train",
        "lr score test",
        "gbr score train",
        "gbr score test",
        "rfr score train",
        "rfr score test",
    ]

    # Metric Columns
    columns_metrics = [
        "predicted train size",
        "predicted random state",
        "predicted lr",
        "predicted lr MAE",
        "predicted lr MSE",
        "predicted lr R2S",
        "predicted gbr",
        "predicted gbr MAE",
        "predicted gbr MSE",
        "predicted gbr R2S",
        "predicted rfr",
        "predicted rfr MAE",
        "predicted rfr MSE",
        "predicted rfr R2S",
    ]

    exclude_columns = ["visit", "city"]

    def __init__(
            self,
            data: pd.DataFrame,
            unique_cities: List,
            output: OutputTypeEnum,
            storage_model: OptionEnum,
    ):
        self.output = output
        self.data = data.copy()
        self.unique_cities = unique_cities
        self.general_graphs = Visualization()
        self.storage_model = storage_model

    def __del__(self):
        del self.general_graphs
        del self.data

    def __predict(self, model: object, x_find: str) -> str:
        """
        Create a prediction
        :param model: model to check the prediction
        :param x_find: value to predict
        :return: value predicted
        """
        if model is None:
            return None
        x_find = x_find.to_frame().transpose()
        return model.predict(x_find)

    def __get_percentage_use(self, data: pd.DataFrame) -> None:
        """
        Get percentage model used
        :param data: initial dataframe
        """
        output_result = OutputResult(self.output)
        output_result.title = "Get Percentage use"

        df_result_model = data.groupby(by=["model_predicted"], as_index=False)[
            "city"
        ].count()
        df_result_model["model_predicted%"] = (df_result_model["city"] * 100) / len(
            data
        )
        output_result.process(df_result_model)

    def __get_mean_visit_predicted_by_model(
            self, data: pd.DataFrame, model: str
    ) -> None:
        """
        Get mean visit predicted by model
        :param data: initial dataframe
        :param model: model GBR, LR, RFR
        """
        print(f"Generating Graph: {model} Mean visit vs Mean visit predicted")

        result = (
            data[data["model_predicted"] == model]
            .groupby(by="cyear", as_index=False)["visit", "visit_predicted"]
            .mean()
        )
        self.general_graphs.title = f"{model} Mean visit vs Mean visit predicted"
        self.general_graphs.comparison_plot(
            result, "cyear", "visit", "visit_predicted"
        )

    def __get_mean_visit_predicted(self, data: pd.DataFrame) -> None:
        """
        Generate graph mean visit vs mean visit predicted
        :param data: initial dataframe
        """
        print("Generating Graph: Mean visit vs Mean visit predicted")

        result = data.groupby(by="cyear", as_index=False)[
            "visit", "visit_predicted"
        ].mean()
        self.general_graphs.title = "Mean visit vs Mean visit predicted"
        self.general_graphs.comparison_plot(
            result, "cyear", "visit", "visit_predicted"
        )

    def process(self, standard_normalization: StandardScaler):
        """
        Predict visits
        :param standard_normalization: visit standard scaler
        """
        dict_df_transform = {}
        exclude_col = self.exclude_columns + self.columns_score
        exclude_col_metric = ["city"] + self.columns_metrics + self.columns_score

        for item in self.unique_cities:
            dict_df_transform[item] = self.data[self.data["city"] == item].copy()
            dict_df_transform[item].drop_duplicates(inplace=True)

        for item in tqdm(self.unique_cities, desc="City"):
            columns_names = (
                dict_df_transform[item]
                .drop(columns=self.exclude_columns)
                .columns.values
            )
            dict_df_transform[item][[self.columns_score]] = np.nan

            lr, train_score, test_score = train_linear_regression(
                dict_df_transform[item],
                "visit",
                exclude_col,
                self.train_size,
                self.random_state,
                self.storage_model,
            )
            if train_score:
                dict_df_transform[item]["lr score train"] = train_score
            if test_score:
                dict_df_transform[item]["lr score test"] = test_score

            gbr, train_score, test_score = train_gradient_boosting_regressor(
                dict_df_transform[item],
                "visit",
                exclude_col,
                self.train_size,
                self.random_state,
                self.storage_model,
            )
            if train_score:
                dict_df_transform[item]["gbr score train"] = train_score
            if test_score:
                dict_df_transform[item]["gbr score test"] = test_score

            rfr, train_score, test_score = train_random_forest_regressor(
                dict_df_transform[item],
                "visit",
                exclude_col,
                self.train_size,
                self.random_state,
                self.storage_model,
            )
            if train_score:
                dict_df_transform[item]["rfr score train"] = train_score
            if test_score:
                dict_df_transform[item]["rfr score test"] = test_score

            dict_df_transform[item][self.columns_metrics] = np.nan

            for index, row in tqdm(
                    dict_df_transform[item].iterrows(),
                    desc=f"{item}: {len(dict_df_transform[item])}",
                    leave=False,
            ):
                predicted_lr = self.__predict(
                    lr, dict_df_transform[item][columns_names].loc[index]
                )
                predicted_gbr = self.__predict(
                    gbr, dict_df_transform[item][columns_names].loc[index]
                )
                predicted_rfr = self.__predict(
                    rfr, dict_df_transform[item][columns_names].loc[index]
                )

                dict_df_transform[item]["predicted train size"] = self.train_size
                dict_df_transform[item]["predicted random state"] = self.random_state

                if predicted_lr is not None:
                    dict_df_transform[item]["predicted lr"].loc[[index]] = predicted_lr
                    mae, mse, r2s = get_metrics(
                        "visit",
                        dict_df_transform[item].loc[index],
                        predicted_lr,
                        exclude_col_metric,
                    )
                    dict_df_transform[item]["predicted lr MAE"].loc[[index]] = mae
                    dict_df_transform[item]["predicted lr MSE"].loc[[index]] = mse
                    dict_df_transform[item]["predicted lr R2S"].loc[[index]] = r2s

                if predicted_gbr is not None:
                    dict_df_transform[item]["predicted gbr"].loc[
                        [index]
                    ] = predicted_gbr
                    mae, mse, r2s = get_metrics(
                        "visit",
                        dict_df_transform[item].loc[index],
                        predicted_gbr,
                        exclude_col_metric,
                    )
                    dict_df_transform[item]["predicted gbr MAE"].loc[[index]] = mae
                    dict_df_transform[item]["predicted gbr MSE"].loc[[index]] = mse
                    dict_df_transform[item]["predicted gbr R2S"].loc[[index]] = r2s

                if predicted_rfr is not None:
                    dict_df_transform[item]["predicted rfr"].loc[
                        [index]
                    ] = predicted_rfr
                    mae, mse, r2s = get_metrics(
                        "visit",
                        dict_df_transform[item].loc[index],
                        predicted_rfr,
                        exclude_col_metric,
                    )
                    dict_df_transform[item]["predicted rfr MAE"].loc[[index]] = mae
                    dict_df_transform[item]["predicted rfr MSE"].loc[[index]] = mse
                    dict_df_transform[item]["predicted rfr R2S"].loc[[index]] = r2s

            del lr
            del gbr
            del rfr

        for item in self.unique_cities:
            dict_df_transform[item] = undummify(dict_df_transform[item])

        df_result = pd.concat(dict_df_transform.values())
        del dict_df_transform

        df_result.drop_duplicates(inplace=True)
        df_result.reset_index(drop=True, inplace=True)

        df_result["visit_predicted"] = np.where(
            (df_result["predicted rfr R2S"] < df_result["predicted lr R2S"])
            & (df_result["predicted rfr R2S"] < df_result["predicted gbr R2S"]),
            df_result["predicted rfr"],
            np.where(
                (df_result["predicted lr R2S"] < df_result["predicted rfr R2S"])
                & (df_result["predicted lr R2S"] < df_result["predicted gbr R2S"]),
                df_result["predicted lr"],
                df_result["predicted gbr"],
            ),
        )
        df_result["model_predicted"] = np.where(
            (df_result["predicted rfr R2S"] < df_result["predicted lr R2S"])
            & (df_result["predicted rfr R2S"] < df_result["predicted gbr R2S"]),
            "RFR",
            np.where(
                (df_result["predicted lr R2S"] < df_result["predicted rfr R2S"])
                & (df_result["predicted lr R2S"] < df_result["predicted gbr R2S"]),
                "LR",
                "GBR",
            ),
        )

        self.__get_percentage_use(df_result)
        self.__get_mean_visit_predicted_by_model(df_result, "LR")
        self.__get_mean_visit_predicted_by_model(df_result, "RFR")
        self.__get_mean_visit_predicted_by_model(df_result, "GBR")

        if standard_normalization:
            df_result["visit"] = standard_normalization.inverse_transform(
                df_result[["visit"]]
            )
            df_result["visit_predicted"] = standard_normalization.inverse_transform(
                df_result[["visit_predicted"]]
            )

        for item in self.unique_cities:
            for j in range(2022, 2024):
                if j == 2022:
                    n = 9
                else:
                    n = 1

                for i in range(n, 12):
                    data_set_visits = pd.DataFrame({
                        "cmonth": i,
                        "cyear": j,
                    }, index=[0]
                    )
                    predict_value = self.__predict_new_visits(data_set_visits, "GBR", item)
                    temp_df = pd.DataFrame({
                        "city": item,
                        "cmonth": i,
                        "cyear": j,
                        "visit": predict_value,
                    }, index=[df_result.index.max() + 1]
                    )

                df_result = df_result.append(temp_df)

        self.__get_mean_visit_predicted(df_result)

        for item in self.unique_cities:
            self.__get_visit_predicted_by_city(item, df_result)

        storage_sqlite(df_result, "PREDICTED")

    def __get_visit_predicted_by_city(self, city, data: pd.DataFrame) -> None:
        """
        Generate graph mean visit vs mean visit predicted
        :param data: initial dataframe
        """
        print(f"Generating Graph: Mean visit vs Mean visit predicted [{city}]")

        result = data[data["city"] == city].groupby(by="cyear", as_index=False)[
            "visit", "visit_predicted"
        ].mean()
        self.general_graphs.title = f"Mean visit vs Mean visit predicted - {city}"
        self.general_graphs.comparison_plot(
            result, "cyear", "visit", "visit_predicted"
        )

    def __predict_new_visits(self, X_test, model_name, city):

        model = joblib.load(f"reports/job_lib/{model_name}_{city.replace('/', '-')}.joblib")

        # Make predictions
        y_pred = model.predict(X_test)

        return y_pred
