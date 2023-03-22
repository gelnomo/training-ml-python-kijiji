import pandas as pd
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from modules.utils import undummify, dummify
from typing import List


class TransformationData:
    def __init__(self, data: pd.DataFrame, unique_city: List):
        self.data = data.copy()
        self.unique_city = unique_city

    def __impute_data_by_iterative(
            self, data: pd.DataFrame, columns_to_filter: List, column_to_impute: str
    ) -> pd.DataFrame:
        """
        Impute data by Iterative imputer function
        :param data: initial dataframe
        :param columns_to_filter: columns to use as a filter to find the value
        :param column_to_impute: column to impute
        :return: dataframe imputed
        """

        data_df = data.copy()
        columns_to_impute = [column_to_impute] + columns_to_filter
        impute_it = IterativeImputer()
        df_transform_fit = impute_it.fit_transform(data_df[columns_to_impute])
        df_temp = pd.DataFrame(df_transform_fit[:, 0], columns=[column_to_impute])
        df_temp[column_to_impute] = df_temp[column_to_impute]
        data_df.update(df_temp)

        return data_df

    def __impute_data(self) -> None:
        """
        Impute visit data
        """
        self.data.loc[self.data["condition"] == "used", "condition"] = 0
        self.data.loc[self.data["condition"] == "damaged", "condition"] = 1
        self.data.loc[self.data["condition"] == "new", "condition"] = 2
        self.data.loc[self.data["condition"] == "salvage", "condition"] = 3
        self.data.loc[self.data["condition"] == "lease_takeover", "condition"] = 4

        for item in tqdm(self.unique_city, desc="City"):
            city_df = self.data[(self.data["city"] == item)]
            city_df = dummify(city_df, "make")
            city_df = dummify(city_df, "condition")
            columns_to_impute = city_df.loc[
                                :, ~city_df.columns.isin(["city", "cmonth", "cmonthname"])
                                ].columns.values.tolist()

            city_df = self.__impute_data_by_iterative(
                city_df, columns_to_impute, "visit"
            )
            city_df = undummify(city_df)
            self.data[(self.data["city"] == item)] = city_df[self.data[(self.data["city"] == item)].columns.tolist()]

        self.data.loc[self.data["condition"] == 0, "condition"] = "used"
        self.data.loc[self.data["condition"] == 1, "condition"] = "damaged"
        self.data.loc[self.data["condition"] == 2, "condition"] = "new"
        self.data.loc[self.data["condition"] == 3, "condition"] = "salvage"
        self.data.loc[self.data["condition"] == 4, "condition"] = "lease_takeover"

        self.data.loc[self.data['visit'].isna(), 'visit'] = 0
        self.data = self.data[self.data["visit"] != 0]

    def __create_date_fields(self) -> None:
        """
        Create year, month, month_name fields
        """
        self.data["cyear"] = self.data["Date"].dt.year
        self.data["cmonth"] = self.data["Date"].dt.month
        self.data["cmonthname"] = self.data["Date"].dt.month_name()
        self.data.drop(columns="Date", inplace=True)

    def process(self) -> pd.DataFrame:
        """
        Process tranformation data
        :return: dataframe
        """
        self.__create_date_fields()
        self.__impute_data()

        return self.data
