import random
import pandas as pd
from typing import List, Tuple
from modules.extraction import ExtractionData
from modules.output_data import OutputResult
from modules.enums import OutputTypeEnum, OptionEnum
from modules.visualization import Visualization
from modules.cleansing import CleansingData
from modules.transformations import TransformationData
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AnalyzeData:
    """A class for analyzing and visualizing data"""

    MONTHS: List[str] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    def __init__(self, output: OutputTypeEnum, normalization: OptionEnum):
        self.output = output
        self.VISITS = None
        self.standard_normalization = None
        self.general_graphs = Visualization()
        self.extraction_data = ExtractionData()
        self.output_result = OutputResult(self.output)
        self.normalization = normalization

    def __del__(self):
        del self.VISITS
        del self.general_graphs

    def __get_basic_info(self, data: pd.DataFrame) -> None:
        """Get basic information about the data.
        :param data: initial dataframe
        """
        print("Generating basic info")

        self.output_result.title = "Get Initial data"
        self.output_result.process(data.head())

        self.output_result.title = "Get shape"
        self.output_result.process(f"The shape of the dataset is: {data.shape}")

        self.output_result.title = "Get description"
        self.output_result.process(data.describe(include="all").transpose())

        self.output_result.title = "Get info"
        self.output_result.process(data.info())

    def __get_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall dataset statistics
        :param data: initial dataframe
        :return: summary dataframe
        """

        summary = pd.DataFrame(
            {
                "Count": data.count(),
                "Difference": len(data) - data.count(),
                "%Difference": ((len(data) - data.count()) * 100) / len(data),
                "Unique": data.nunique(),
                "%Unique": data.nunique() * 100 / len(data),
                "Duplicated": data.duplicated().sum(),
                "%Duplicated": data.duplicated().sum() * 100 / len(data),
                "Null": data.isna().sum(),
                "%Null": data.isnull().sum() * 100 / len(data),
                "type": data.dtypes,
            }
        )
        return summary

    def __get_nulls(self, data: pd.DataFrame) -> None:
        """Get nulls columns from the data.
        :param data: initial dataframe
        """

        print("Generating graph nulls")
        self.general_graphs.title = "Quantity Nulls per Column"
        self.general_graphs.heatmap_nulls(data)

    def __get_percentages_nulls(self, data: pd.DataFrame) -> None:
        """
        Get percentages nulls
        :param data: initial dataframe
        """
        print("Generating graph nulls percentages")
        df_graph = round((data.isnull().sum() * 100 / len(data)), 2)
        self.general_graphs.title = "Percentage Nulls per Columns"
        self.general_graphs.x_rotation = 90
        self.general_graphs.bar_fontsize = 15
        self.general_graphs.bar_plot(
            df_graph.index, df_graph.values, "Columns", "Percentage"
        )

    def __get_number_ads_by_city(self) -> None:
        """
        Number of ads by city
        """
        print("Generating Graph: Number of Ads by City")
        self.general_graphs.title = "Number of Ads by City"
        self.general_graphs.x_rotation = 90
        self.general_graphs.count_bar_plot(self.VISITS["city"], "City", "Count")

    def __get_number_ads_by_year(self) -> None:
        """
        Number of ads by year
        """
        print("Generating Graph: Number of Ads by Year")
        self.general_graphs.title = "Number of Ads by Year"
        self.general_graphs.count_bar_plot(self.VISITS["cyear"], "Year", "Count")

    def __get_number_ads_by_month(self) -> None:
        """
        Number of ads by month
        """
        print("Generating Graph: Number of Ads by Month")
        self.general_graphs.title = "Number of Ads by Month"
        self.general_graphs.count_bar_plot(
            self.VISITS["cmonthname"], "Month", "Count", self.MONTHS
        )

    def __get_number_of_visits_by_month(self) -> None:
        """
        Number of visits by month
        """
        print("Generating Graph: Number of Visits by Month")

        df_graph = self.VISITS.groupby(by=["cmonthname"], as_index=False)[
            "visit"
        ].sum()

        self.general_graphs.title = "Number of Visits by Month"
        self.general_graphs.bar_fontsize = 10
        self.general_graphs.bar_plot(
            df_graph["cmonthname"],
            df_graph["visit"],
            "Month",
            "Visits",
            None,
            self.MONTHS,
        )

        del df_graph

    def __get_number_of_visits_by_month_year(self) -> None:
        print("Generating Graph: Number of Visits by Month and Year")

        # Create a data frame to summarize the visits to year and month
        df_graph = self.VISITS[["visit", "cyear", "cmonthname", "cmonth"]]
        df_graph = df_graph.groupby(
            ["cyear", "cmonthname", "cmonth"], as_index=False
        )["visit"].sum()

        self.general_graphs.title = "Monthly Visits by year"
        self.general_graphs.column_huge_name = "Year"
        self.general_graphs.line_plot(
            df_graph["cmonth"], df_graph["visit"], df_graph["cyear"], "Month", "Visit"
        )

        del df_graph

    def __get_number_of_visits_by_year_make_car(self) -> None:
        """
        Number of visits by year (make car)
        """
        print("Generating Graph: Number of Visits by Year (Make Car)")

        # Create a data frame to summarize the visits to year and month
        df_graph = self.VISITS[["visit", "cyear", "make"]]
        df_graph = df_graph.groupby(["cyear", "make"], as_index=False)["visit"].sum()

        self.general_graphs.title = "Make (visit) by year"
        self.general_graphs.bar_fontsize = 5
        self.general_graphs.x_rotation = 90
        self.general_graphs.bar_rotation = 90
        self.general_graphs.column_huge_name = "Year"
        self.general_graphs.legend_loc = "upper center"
        self.general_graphs.bar_plot(
            df_graph["make"], df_graph["visit"], "Make", "visit", df_graph["cyear"]
        )

        del df_graph

    def __get_number_of_visits_by_city(self) -> None:
        """
        Number of visits by city
        """
        print("Generating Graph: Number of Visits by city")

        # Create a data frame to summarize the visits to year and month
        df_graph = self.VISITS[["visit", "cyear", "city"]]
        df_graph = df_graph.groupby(["cyear", "city"], as_index=False)["visit"].sum()

        self.general_graphs.title = "Visits by year by city"
        self.general_graphs.bar_fontsize = 5
        self.general_graphs.x_rotation = 90
        self.general_graphs.bar_rotation = 90
        self.general_graphs.column_huge_name = "Year"
        self.general_graphs.bar_plot(
            df_graph["city"], df_graph["visit"], "city", "visit", df_graph["cyear"]
        )

        del df_graph

    def __get_shared_of_visit_by_year(self) -> None:
        """
        Get shared of visits by year
        """
        print("Generating Graph: Shared-of-Visit by Year")

        df_graph = self.VISITS[["visit", "cyear"]].groupby("cyear").sum()
        n_explote = len(self.VISITS["cyear"].unique())

        explode = []
        list_colors = []

        random_color = lambda: random.randint(0, 255)
        [explode.append(0.1 if n == 0 else 0) for n in range(n_explote)]
        [
            list_colors.append(
                "#%02X%02X%02X" % (random_color(), random_color(), random_color())
            )
            for i in range(n_explote)
        ]

        total_visits = df_graph.sum()

        self.general_graphs.title = "Share-of-Visit by year"
        self.general_graphs.count_pie_plot(
            df_graph,
            "visit",
            tuple(explode),
            list_colors,
            f'TOTAL:\n {"{:,}".format(total_visits[0])}',
        )

        del df_graph

    def __get_density(self, data: pd.DataFrame, column_name: str, title: str) -> None:
        """
        Get density column
        :param data: initial dataframe
        :param column_name: column name
        :param title: title graph
        """
        print(f"Generating Graph: {title}")

        self.general_graphs.title = f"{title}"
        self.general_graphs.distribution_kdeplot(
            data[column_name], column_name, "density"
        )

    def __get_covariance(self) -> None:
        """
        Get covariance
        """
        print("Generating: Covariance")

        self.output_result.title = "Get covariance"
        self.output_result.process(self.VISITS["cmonth"].cov(self.VISITS["visit"]))

        self.general_graphs.title = "Correlation Month"
        self.general_graphs.count_scatter(
            self.VISITS["cmonth"], self.VISITS["visit"], "month", "visit"
        )
        self.general_graphs.title = "Correlation Price per Year"
        self.general_graphs.column_huge_name = "cyear"
        self.general_graphs.count_lmplot(
            self.VISITS, "cmonth", "visit", "Month", "Visits"
        )

    def __get_correlation(self, title) -> None:
        """
        Get correlation heat map
        :param title: Title graph
        """
        print(f"Generating Graph: {title}")

        le_drivetrain = LabelEncoder()
        le_fuel_type = LabelEncoder()
        le_model = LabelEncoder()
        le_make = LabelEncoder()

        self.VISITS["condition"] = le_model.fit_transform(self.VISITS["condition"])
        self.VISITS["make"] = le_make.fit_transform(self.VISITS["make"])

        self.general_graphs.title = title
        self.general_graphs.heat_map_corr(self.VISITS.corr())

        self.VISITS["condition"] = le_model.inverse_transform(self.VISITS["condition"])
        self.VISITS["make"] = le_make.inverse_transform(self.VISITS["make"])

    def __get_facet_grid_make(self) -> None:
        """
        Get facet grid to make cars
        """
        print("Generating Graph: Facet Grid")

        self.general_graphs.title = "Density city vs visit"
        self.general_graphs.facet_grid(self.VISITS[["city", "visit"]], "city", "visit")

    def process(self) -> Tuple[pd.DataFrame, List, StandardScaler]:
        """
        Process Analysis
        :return: Tuple dataframe, list cities, standardscaler visits
        """
        self.VISITS = self.extraction_data.read(
            "VISITS.csv", ["Date"], "%m/%d/%Y, %H:%M:%S"
        )

        summary_visits = self.__get_summary(self.VISITS)
        self.__get_basic_info(self.VISITS)
        self.output_result.title = "Get overall"
        self.output_result.process(summary_visits)
        self.__get_nulls(self.VISITS)
        self.__get_percentages_nulls(self.VISITS)
        self.__get_correlation("Correlation before cleaning")

        self.VISITS, unique_city = CleansingData(
            self.output, self.VISITS, summary_visits
        ).process()
        self.VISITS = TransformationData(self.VISITS, unique_city).process()

        self.__get_number_ads_by_city()
        self.__get_number_ads_by_year()
        self.__get_number_ads_by_month()
        self.__get_number_of_visits_by_month()
        self.__get_number_of_visits_by_month_year()
        self.__get_number_of_visits_by_year_make_car()
        self.__get_number_of_visits_by_city()
        self.__get_shared_of_visit_by_year()

        if self.normalization == OptionEnum.YES:
            self.VISITS, self.standard_normalization = CleansingData(
                self.output, self.VISITS, summary_visits
            ).normalize_data("visit")

        self.__get_density(self.VISITS, "visit", "Before - Visits Density")
        self.VISITS = CleansingData(
            self.output, self.VISITS, summary_visits
        ).identify_outliers(unique_city)
        self.__get_density(self.VISITS, "visit", "After - Visits Density")
        self.__get_covariance()
        self.__get_correlation("Correlation after cleaning")
        self.__get_facet_grid_make()

        del self.extraction_data
        self.__group_values()
        return self.VISITS, unique_city, self.standard_normalization

    def __group_values(self):
        self.VISITS = self.VISITS.groupby(["city", "cmonth", "cyear"], as_index=False)["visit"].sum()