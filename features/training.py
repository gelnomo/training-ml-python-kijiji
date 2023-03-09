import numpy as np
import pandas as pd

from features.enums import OutputTypeEnum
from features.utils import dummify, undummify, storage_result
from tqdm import tqdm
from features.output_result import OutputResult
from models.gradient_boostring_regressor.train_model import (
    train_gradient_boosting_regressor,
)
from models.linear_regression.train_model import train_linear_regression
from models.metrics.all_metrics import get_metrics
from models.random_forest_regressor.train_model import train_random_forest_regressor
from visualizations.visualize import GeneralGraphs


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
        "predicted lr accuracy",
        "predicted lr MAE",
        "predicted lr MSE",
        "predicted lr R2S",
        "predicted gbr",
        "predicted gbr accuracy",
        "predicted gbr MAE",
        "predicted gbr MSE",
        "predicted gbr R2S",
        "predicted rfr",
        "predicted rfr accuracy",
        "predicted rfr MAE",
        "predicted rfr MSE",
        "predicted rfr R2S",
    ]

    exclude_columns = ["price", "make"]

    def __init__(self, df, unique_make, output: OutputTypeEnum):
        self.output = output
        self.df_transform = df.copy()
        self.unique_make = unique_make
        self.general_graphs = GeneralGraphs()

    def __del__(self):
        del self.general_graphs
        del self.df_transform

    # Predict Value
    def __predict(self, model, x_find):
        if model is None:
            return None
        x_find = x_find.to_frame().transpose()
        return model.predict(x_find)

    def __get_percentage_use(self, df_result):
        output_result = OutputResult(self.output)
        output_result.title = "Get Percentage use"

        df_result_model = df_result.groupby(by=["model_predicted"], as_index=False)[
            "make"
        ].count()
        df_result_model["model_predicted%"] = (df_result_model["make"] * 100) / len(
            df_result
        )
        output_result.process(df_result_model)

    def __get_rfr_mean_price_predicted(self, df_result):
        print("Generating Graph: RFR Mean price vs Mean price predicted")

        result = (
            df_result[df_result["model_predicted"] == "RFR"]
            .groupby(by="year", as_index=False)["price", "price_predicted"]
            .mean()
        )
        self.general_graphs.title = "RFR Mean price vs Mean price predicted"
        self.general_graphs.comparison_plot(result, "year", "price", "price_predicted")

    def __get_lr_mean_price_predicted(self, df_result):
        print("Generating Graph: LR Mean price vs Mean price predicted")

        result = (
            df_result[df_result["model_predicted"] == "LR"]
            .groupby(by="year", as_index=False)["price", "price_predicted"]
            .mean()
        )
        self.general_graphs.title = "LR Mean price vs Mean price predicted"
        self.general_graphs.comparison_plot(result, "year", "price", "price_predicted")

    def __get_gbr_mean_price_predicted(self, df_result):
        print("Generating Graph: GBR Mean price vs Mean price predicted")

        result = (
            df_result[df_result["model_predicted"] == "GBR"]
            .groupby(by="year", as_index=False)["price", "price_predicted"]
            .mean()
        )
        self.general_graphs.title = "GBR Mean price vs Mean price predicted"
        self.general_graphs.comparison_plot(result, "year", "price", "price_predicted")

    def __get_mean_price_predicted(self, df_result):
        print("Generating Graph: Mean price vs Mean price predicted")

        result = df_result.groupby(by="year", as_index=False)[
            "price", "price_predicted"
        ].mean()
        self.general_graphs.title = "Mean price vs Mean price predicted"
        self.general_graphs.comparison_plot(result, "year", "price", "price_predicted")

    def process(self):
        dict_df_transform = {}
        exclude_col = self.exclude_columns + self.columns_score
        exclude_col_metric = ["make"] + self.columns_metrics + self.columns_score

        for item in self.unique_make:
            dict_df_transform[item] = self.df_transform[
                self.df_transform["make"] == item
            ].copy()
            dict_df_transform[item] = dummify(dict_df_transform[item], "fueltype")
            dict_df_transform[item] = dummify(dict_df_transform[item], "condition")
            dict_df_transform[item] = dummify(dict_df_transform[item], "model")
            dict_df_transform[item] = dummify(dict_df_transform[item], "city")
            dict_df_transform[item] = dummify(dict_df_transform[item], "Drivetrain")
            dict_df_transform[item].drop(
                columns=["visit", "Kilometers", "c_year", "c_month", "c_month_name"],
                inplace=True,
            )
            dict_df_transform[item].drop_duplicates(inplace=True)

        for item in tqdm(self.unique_make, desc="Make"):
            columns_names = (
                dict_df_transform[item]
                .drop(columns=self.exclude_columns)
                .columns.values
            )
            dict_df_transform[item][[self.columns_score]] = np.nan
            lr = train_linear_regression(
                dict_df_transform[item],
                "price",
                exclude_col,
                self.train_size,
                self.random_state,
            )
            gbr = train_gradient_boosting_regressor(
                dict_df_transform[item],
                "price",
                exclude_col,
                self.train_size,
                self.random_state,
            )
            rfr = train_random_forest_regressor(
                dict_df_transform[item],
                "price",
                exclude_col,
                self.train_size,
                self.random_state,
            )

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
                    mae, mse, r2s, accuracy = get_metrics(
                        lr,
                        dict_df_transform[item].loc[index],
                        predicted_lr,
                        exclude_col_metric,
                    )
                    dict_df_transform[item]["predicted lr MAE"].loc[[index]] = mae
                    dict_df_transform[item]["predicted lr MSE"].loc[[index]] = mse
                    dict_df_transform[item]["predicted lr R2S"].loc[[index]] = r2s
                    dict_df_transform[item]["predicted lr accuracy"].loc[
                        [index]
                    ] = accuracy

                if predicted_gbr is not None:
                    dict_df_transform[item]["predicted gbr"].loc[
                        [index]
                    ] = predicted_gbr
                    mae, mse, r2s, accuracy = get_metrics(
                        gbr,
                        dict_df_transform[item].loc[index],
                        predicted_gbr,
                        exclude_col_metric,
                    )
                    dict_df_transform[item]["predicted gbr MAE"].loc[[index]] = mae
                    dict_df_transform[item]["predicted gbr MSE"].loc[[index]] = mse
                    dict_df_transform[item]["predicted gbr R2S"].loc[[index]] = r2s
                    dict_df_transform[item]["predicted gbr accuracy"].loc[
                        [index]
                    ] = accuracy

                if predicted_rfr is not None:
                    dict_df_transform[item]["predicted rfr"].loc[
                        [index]
                    ] = predicted_rfr
                    mae, mse, r2s, accuracy = get_metrics(
                        rfr,
                        dict_df_transform[item].loc[index],
                        predicted_rfr,
                        exclude_col_metric,
                    )
                    dict_df_transform[item]["predicted rfr MAE"].loc[[index]] = mae
                    dict_df_transform[item]["predicted rfr MSE"].loc[[index]] = mse
                    dict_df_transform[item]["predicted rfr R2S"].loc[[index]] = r2s
                    dict_df_transform[item]["predicted rfr accuracy"].loc[
                        [index]
                    ] = accuracy

            del lr
            del gbr
            del rfr

        for item in self.unique_make:
            dict_df_transform[item] = undummify(dict_df_transform[item])

        df_result = pd.concat(dict_df_transform.values())
        del dict_df_transform

        df_result.drop_duplicates(inplace=True)
        df_result.reset_index(drop=True, inplace=True)

        df_result["price_predicted"] = np.where(
            (df_result["predicted rfr MSE"] < df_result["predicted lr MSE"])
            & (df_result["predicted rfr MSE"] < df_result["predicted gbr MSE"]),
            df_result["predicted rfr"],
            np.where(
                (df_result["predicted lr MSE"] < df_result["predicted rfr MSE"])
                & (df_result["predicted lr MSE"] < df_result["predicted gbr MSE"]),
                df_result["predicted lr"],
                df_result["predicted gbr"],
            ),
        )
        df_result["model_predicted"] = np.where(
            (df_result["predicted rfr MSE"] < df_result["predicted lr MSE"])
            & (df_result["predicted rfr MSE"] < df_result["predicted gbr MSE"]),
            "RFR",
            np.where(
                (df_result["predicted lr MSE"] < df_result["predicted rfr MSE"])
                & (df_result["predicted lr MSE"] < df_result["predicted gbr MSE"]),
                "LR",
                "GBR",
            ),
        )

        self.__get_percentage_use(df_result)
        self.__get_lr_mean_price_predicted(df_result)
        self.__get_rfr_mean_price_predicted(df_result)
        self.__get_gbr_mean_price_predicted(df_result)
        self.__get_mean_price_predicted(df_result)
        storage_result(df_result)

        del df_result
