import pandas as pd
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from features.utils import undummify, dummify


class TransformationData:
    def __init__(self, df, unique_make):
        self.df = df
        self.unique_make = unique_make

    def __impute_data_itself(self, column):
        # Impute data to Drivetrain, fueltype, transmission based on already existing data (make, model, year, visit).

        df_imputed = pd.merge(
            self.df[self.df[column].isna()][["make", "model", "year", "AdID"]],
            self.df[~self.df[column].isna()][
                ["make", "model", "year", column, "visit"]
            ].drop_duplicates(),
            on=["make", "model", "year"],
        ).drop_duplicates()
        df_imputed["item"] = df_imputed.groupby(["AdID"])["visit"].rank(
            method="first", ascending=False, na_option="top"
        )
        df_imputed = df_imputed[df_imputed["item"] == 1][["AdID", column]]
        self.df.set_index("AdID", inplace=True)
        df_imputed.set_index("AdID", inplace=True)
        self.df.update(df_imputed)
        self.df["AdID"] = self.df.index
        self.df.reset_index(drop=True, inplace=True)

    def __impute_data_by_iterative(self, columns_impute, column):
        # Function to impute data using IterativeImputer to fill the nan to Drievetrain and fueltype

        columns_to_impute = [column] + columns_impute
        impute_it = IterativeImputer()
        df_transform_fit = impute_it.fit_transform(self.df[columns_to_impute])
        df_temp = pd.DataFrame(df_transform_fit[:, 0], columns=[column])
        df_temp[column] = abs(round(df_temp[column]))
        self.df.update(df_temp)

    def __impute_data(self):
        # Becoming categorical fields (Drivetrain, fueltype) values into numerical values.

        # Drivetrain
        self.df.loc[self.df["Drivetrain"] == "4x4", "Drivetrain"] = 0
        self.df.loc[self.df["Drivetrain"] == "rwd", "Drivetrain"] = 1
        self.df.loc[self.df["Drivetrain"] == "fwd", "Drivetrain"] = 2
        self.df.loc[self.df["Drivetrain"] == "awd", "Drivetrain"] = 3
        self.df.loc[self.df["Drivetrain"] == "other", "Drivetrain"] = 4

        # Fuel Type
        self.df.loc[self.df["fueltype"] == "gas", "fueltype"] = 0
        self.df.loc[self.df["fueltype"] == "diesel", "fueltype"] = 1
        self.df.loc[self.df["fueltype"] == "electric", "fueltype"] = 2
        self.df.loc[self.df["fueltype"] == "hybrid", "fueltype"] = 3
        self.df.loc[self.df["fueltype"] == "other", "fueltype"] = 4

        for item in tqdm(self.unique_make, desc="Make"):
            model_columns = self.df[
                (~self.df["model"].isna()) & (self.df["make"] == item)
            ]["model"].unique()
            model_columns = [f"model_{x}" for x in model_columns]

            self.df = dummify(self.df, "model")

            columns_to_impute = ["year"] + model_columns

            self.__impute_data_by_iterative(columns_to_impute, "Transmission")
            self.__impute_data_by_iterative(columns_to_impute, "Drivetrain")
            self.__impute_data_by_iterative(columns_to_impute, "fueltype")
            self.df = undummify(self.df)

        # Drivetrain
        self.df.loc[self.df["Drivetrain"] == 0, "Drivetrain"] = "4x4"
        self.df.loc[self.df["Drivetrain"] == 1, "Drivetrain"] = "rwd"
        self.df.loc[self.df["Drivetrain"] == 2, "Drivetrain"] = "fwd"
        self.df.loc[self.df["Drivetrain"] == 3, "Drivetrain"] = "awd"
        self.df.loc[self.df["Drivetrain"] == 4, "Drivetrain"] = "other"

        # Fuel Type
        self.df.loc[self.df["fueltype"] == 0, "fueltype"] = "gas"
        self.df.loc[self.df["fueltype"] == 1, "fueltype"] = "diesel"
        self.df.loc[self.df["fueltype"] == 2, "fueltype"] = "electric"
        self.df.loc[self.df["fueltype"] == 3, "fueltype"] = "hybrid"
        self.df.loc[self.df["fueltype"] == 4, "fueltype"] = "other"

    def __create_date_fields(self):
        self.df["c_year"] = self.df["Date"].dt.year
        self.df["c_month"] = self.df["Date"].dt.month
        self.df["c_month_name"] = self.df["Date"].dt.month_name()
        self.df.drop(columns="Date", inplace=True)

    def process(self):
        self.__impute_data_itself("Drivetrain")
        self.__impute_data_itself("fueltype")
        self.__impute_data_itself("Transmission")
        self.__impute_data()
        self.__create_date_fields()

        return self.df
