import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from features.output_result import OutputResult
from features.enums import OutputTypeEnum
from sklearn.ensemble import IsolationForest


class CleansingData:
    def __init__(self, output: OutputTypeEnum, df, overall):
        self.output_result = OutputResult(output)
        self.df = df.copy()
        self.overall = overall
        self.unique_make = None

    def __del__(self):
        del self.output_result

    def process(self):
        self.output_result.title = "Identify the duplicated values"
        self.output_result.process(
            self.df[self.df["AdID"].isin(self.df[self.df.duplicated()]["AdID"].values)]
        )

        # Duplicated values after identify is an isolated issue
        self.df.drop_duplicates(inplace=True)

        # More than **60%** of nulls on columns are going to be deleted
        columns_to_drop = self.overall[self.overall["%Null"] > 60].index.values
        static_columns_to_drop = [
            "title",
            "saleby",
            "Trim",
            "body",
            "color",
            "No of door",
            "No of seats",
        ]
        self.df.drop(columns=columns_to_drop, inplace=True)
        self.df.drop(columns=static_columns_to_drop, inplace=True)

        self.output_result.title = "Columns to be deleted"
        self.output_result.process(columns_to_drop)
        self.output_result.process(static_columns_to_drop)

        # Standardize values with capitalization and removes empty values at the final and start of data.
        columns_to_capitalize = ["make", "model", "condition"]
        self.df[columns_to_capitalize] = self.df[columns_to_capitalize].apply(
            lambda x: x.str.title().str.strip().str.replace("_", " ")
        )

        # Rename Alfarome as a Alfa Romeo
        self.df.loc[self.df["make"] == "Alfaromeo", "make"] = "Alfa Romeo"

        # Identify unique make cars values
        self.unique_make = self.df["make"].unique()

        self.output_result.title = "Identify unique make cars values"
        self.output_result.process(self.unique_make)

        # Delete ads without price because that will affect the prediction
        self.df.drop(index=self.df[self.df["price"] == 0].index, axis=0, inplace=True)

        # Delete 2019 row, it contains just one value that will affect the prediction
        self.df.drop(
            index=self.df[self.df["Date"].dt.year == 2019].index, axis=0, inplace=True
        )

        return self.df, self.unique_make

    def delete_primary_key(self):
        # Delete primary key value, because is not a relevant field to the prediction

        self.df.drop(columns=["AdID"], inplace=True)

        return self.df

    def identify_outliers(self, unique_make):
        for item in unique_make:
            df_make_item = self.df[self.df["make"] == item]["price"]
            outlier_data = OutliersData(df_make_item, True)
            outliers = outlier_data.zr_score_outlier()

            # outliers = outlier_data.z_score_outlier()
            # outliers = outlier_data.iqr_outliers()
            # outliers = outlier_data.winsorization_outliers()
            # outliers = outlier_data.db_outliers()
            # outliers = outlier_data.iso_outliers()

            self.df.drop(self.df[self.df["price"].isin(outliers)].index, inplace=True)

            del outlier_data
        return self.df


class OutliersData:
    def __init__(self, df, inplace=False):
        self.df = df if inplace else df.copy()

    def z_score_outlier(self):
        out = []
        if not any(self.df):
            return out
        m = np.mean(self.df)
        sd = np.std(self.df)
        for i in self.df:
            z = (i - m) / sd
            if np.abs(z) > 3:
                out.append(i)
        return out

    def zr_score_outlier(self):
        out = []
        if not any(self.df):
            return out
        med = np.median(self.df)
        ma = stats.median_abs_deviation(self.df)
        for i in self.df:
            z = (0.6745 * (i - med)) / (np.median(ma))
            if np.abs(z) > 3:
                out.append(i)
        return out

    def iqr_outliers(self):
        out = []
        if not any(self.df):
            return out
        q1 = self.df.quantile(0.25)
        q3 = self.df.quantile(0.75)
        iqr = q3 - q1
        lower_tail = q1 - 1.5 * iqr
        upper_tail = q3 + 1.5 * iqr
        for i in self.df:
            if i > upper_tail or i < lower_tail:
                out.append(i)
        return out

    def winsorization_outliers(self):
        out = []
        if not any(self.df):
            return out
        q1 = np.percentile(self.df, 1)
        q3 = np.percentile(self.df, 99)
        for i in self.df:
            if i > q3 or i < q1:
                out.append(i)
        return out

    def db_outliers(self):
        if not any(self.df):
            return []
        outlier_detection = DBSCAN(eps=2, metric="euclidean", min_samples=5)
        clusters = outlier_detection.fit_predict(self.df.values.reshape(-1, 1))
        data = pd.DataFrame()
        data["cluster"] = clusters
        outliers = data["cluster"].value_counts().sort_values(ascending=False)
        outliers = outliers.to_frame()["cluster"]
        return list(set([x for x in outliers]))

    def iso_outliers(self):
        if not any(self.df):
            return []
        iso = IsolationForest(random_state=1, contamination="auto")
        preds = iso.fit_predict(self.df.values.reshape(-1, 1))
        data = pd.DataFrame()
        data["cluster"] = preds
        result = data["cluster"].value_counts().sort_values(ascending=False)
        outliers = result.to_frame()["cluster"]
        return list(set([x for x in outliers]))
