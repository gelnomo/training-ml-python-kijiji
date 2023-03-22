import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from typing import List, Tuple
from modules.outliers import OutliersData
from modules.output_data import OutputResult
from modules.enums import OutputTypeEnum, OutlierTypeEnum
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from modules.utils import get_outliers


class CleansingData:
    def __init__(
            self, output: OutputTypeEnum, data: pd.DataFrame, summary: pd.DataFrame
    ):
        self.output_result = OutputResult(output)
        self.data = data.copy()
        self.summary = summary
        self.unique_make = None

    def __del__(self):
        del self.output_result

    def process(self):
        self.output_result.title = "Identify the duplicated values"
        self.output_result.process(self.data[self.data.duplicated()])

        # Duplicated values after identify is an isolated issue
        self.data.drop_duplicates(inplace=True)

        # More than **60%** of nulls on columns are going to be deleted
        columns_to_drop = self.summary[self.summary["%Null"] > 60].index.values
        static_columns_to_drop = [
            "title",
            "saleby",
            "Trim",
            "body",
            "color",
            "No of door",
            "No of seats",
            "model",
            "fueltype",
            "Kilometers",
            "Drivetrain",
            "Transmission",
            "year",
            "price",
        ]
        self.data.drop(columns=columns_to_drop, inplace=True)
        self.data.drop(columns=static_columns_to_drop, inplace=True)

        self.output_result.title = "Columns to be deleted with more than 60% of nulls"
        self.output_result.process(columns_to_drop)
        self.output_result.title = "Explicit Columns to be deleted"
        self.output_result.process(static_columns_to_drop)

        # Standardize values with title and removes empty values at the final and start of data.
        columns_to_capitalize = ["city", "make", "condition"]
        self.data[columns_to_capitalize] = self.data[columns_to_capitalize].apply(
            lambda x: x.str.title().str.strip().str.replace("_", " ")
        )

        # Rename Alfarome as a Alfa Romeo
        self.data.loc[self.data["make"] == "Alfaromeo", "make"] = "Alfa Romeo"

        # Identify unique make cars values
        self.unique_city = self.data["city"].unique()

        self.output_result.title = "Identify unique cities values"
        self.output_result.process(self.unique_city)
        self.__delete_primary_key("AdID")
        return self.data, self.unique_city

    def __delete_primary_key(self, column_name: str) -> None:
        """
        Delete primary key value, because is not a relevant field to the prediction
        :param column_name: column name
        """
        self.data.drop(columns=[column_name], inplace=True)

    def normalize_data(self, column_name: str) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Normalize data
        :return: return tuple dataframe and standard scaler
        """
        print("Generating: Normalization")

        standard_normalization = StandardScaler()
        self.data[[column_name]] = standard_normalization.fit_transform(
            self.data[[column_name]]
        )

        return self.data, standard_normalization

    def identify_outliers(self, unique_city: List) -> pd.DataFrame:
        """
        Identify outliers
        :param unique_city: city list
        :return: dataframe cleaned of outliers
        """
        for item in unique_city:
            data_city_item = self.data[self.data["city"] == item][["visit"]]

            outliers = get_outliers(data_city_item, "visit", OutlierTypeEnum.ISOLATION_FOREST)
            outliers += get_outliers(data_city_item, "visit", OutlierTypeEnum.Z_R_SCORE)
            outliers += get_outliers(data_city_item, "visit", OutlierTypeEnum.WINSORIZATION)
            outliers += get_outliers(data_city_item, "visit", OutlierTypeEnum.Z_SCORE)
            outliers += get_outliers(data_city_item, "visit", OutlierTypeEnum.DBSCAN)
            outliers += get_outliers(data_city_item, "visit", OutlierTypeEnum.IQR)

            self.data.drop(
                self.data[(self.data["city"] == item) & (self.data["visit"].isin(set(outliers)))].index,
                inplace=True,
            )

            del outliers
        return self.data
