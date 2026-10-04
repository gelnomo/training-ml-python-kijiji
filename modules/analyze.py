"""Exploratory data analysis: data quality, distributions and drivers of visits."""

from typing import Tuple

import pandas as pd

from modules.cleansing import clean_data
from modules.config import SCRAPE_DATE, TARGET
from modules.enums import OutputTypeEnum
from modules.features import NUMERIC_FEATURES
from modules.outliers import outlier_summary
from modules.output_data import OutputResult
from modules.visualization import Visualization


def get_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Data quality overview per column
    :param data: dataframe
    :return: count, nulls, uniques and dtype per column
    """
    return pd.DataFrame(
        {
            "Count": data.count(),
            "Null": data.isna().sum(),
            "%Null": (data.isna().mean() * 100).round(2),
            "Unique": data.nunique(),
            "%Unique": (data.nunique() * 100 / len(data)).round(2),
            "type": data.dtypes,
        }
    )


class AnalyzeData:
    """Run the EDA, write tables to the chosen output and charts to reports/figures."""

    def __init__(self, output: OutputTypeEnum, plots: bool = True):
        self.output_result = OutputResult(output)
        self.graphs = Visualization() if plots else None

    def _report(self, title: str, content) -> None:
        self.output_result.title = title
        self.output_result.process(content)

    def _data_quality(self, raw: pd.DataFrame) -> None:
        print("Generating: data quality")
        self._report("Shape of the raw dataset", raw.shape)
        self._report("Column summary (raw)", get_summary(raw))
        self._report("Description (raw)", raw.describe(include="all").transpose())
        if self.graphs:
            nulls = (raw.isna().mean() * 100).sort_values(ascending=False)
            self.graphs.bar(nulls[nulls > 0], "Percentage of missing values per column",
                            "% missing", "", horizontal=True, label_fmt="{:.1f}%")

    def _snapshot_effect(self, df: pd.DataFrame) -> None:
        """
        Visits are cumulative and the data is one snapshot: older ads have had more time
        to collect visits, but most older ads were already sold and removed.
        """
        print("Generating: snapshot effect (ad age)")
        by_month = df["Date"].dt.to_period("M").astype(str).value_counts().sort_index()
        self._report("Ads still online by publish month", by_month)
        if self.graphs:
            recent = by_month.tail(12)
            self.graphs.bar(recent, f"Ads still online on {SCRAPE_DATE:%Y-%m-%d}, by publish month",
                            "Publish month", "Number of ads")
            self.graphs.median_line(df["ad_age_days"], df[TARGET], "Visits grow with the age of the ad",
                                    "Days online at scrape date", "Visits")

    def _target(self, df: pd.DataFrame) -> None:
        print("Generating: target distribution and outliers")
        self._report("Visits description", df[TARGET].describe(percentiles=[0.5, 0.9, 0.99, 0.999]))
        # Outliers are reported, not deleted: they are real popular ads, and the models
        # handle the skew with a log transform instead.
        self._report("Visit outliers flagged by each method", outlier_summary(df[TARGET]))
        if self.graphs:
            self.graphs.distribution(df[TARGET], "Distribution of visits per ad", "visits")

    def _drivers(self, df: pd.DataFrame) -> None:
        print("Generating: drivers of visits")
        by_city = df.groupby("city")[TARGET].agg(ads="count", median="median", mean="mean")
        self._report("Visits by city", by_city.sort_values("ads", ascending=False).round(1))

        by_seller = df.groupby(["saleby", "condition"])[TARGET].agg(ads="count", median="median")
        self._report("Visits by seller type and condition", by_seller)

        top_makes = df["make"].value_counts().head(15).index
        by_make = df[df["make"].isin(top_makes)].groupby("make")[TARGET].median().sort_values(ascending=False)

        corr = df[NUMERIC_FEATURES + [TARGET]].corr(method="spearman")
        self._report("Spearman correlation with visits", corr[TARGET].drop(TARGET).sort_values())

        if self.graphs:
            self.graphs.bar(df["city"].value_counts(), "Number of ads by city", "Number of ads", "",
                            horizontal=True)
            self.graphs.bar(by_city["median"].sort_values(ascending=False), "Median visits per ad by city",
                            "Median visits", "", horizontal=True)
            self.graphs.bar(by_make, "Median visits per ad, 15 most listed makes", "Median visits", "",
                            horizontal=True)
            self.graphs.median_line(df["price"].dropna(), df.loc[df["price"].notna(), TARGET],
                                    "Visits by asking price", "Price (CAD)", "Visits")
            self.graphs.heatmap_corr(corr.round(2), "Spearman correlation of numeric features")

    def process(self, raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Run the full analysis
        :param raw: raw dataframe
        :return: cleaned dataframe and cleaning report
        """
        self._data_quality(raw)
        df, report = clean_data(raw)
        self._report("Cleaning report", pd.Series(report))
        self._report("Column summary (clean)", get_summary(df))
        self._snapshot_effect(df)
        self._target(df)
        self._drivers(df)
        return df, report
