import pandas as pd
from features.extraction import ExtractionData
from features.output_result import OutputResult
from features.enums import OutputTypeEnum, OptionEnum
from visualizations.visualize import GeneralGraphs
from features.cleansing import CleansingData
from features.transformations import TransformationData
from sklearn.preprocessing import LabelEncoder


class AnalyzeData:
    def __init__(self, output: OutputTypeEnum, normalization: OptionEnum):
        self.output = output
        self.df = None
        self.overall = None
        self.unique_make = None
        self.general_graphs = GeneralGraphs()
        self.extraction_data = ExtractionData()
        self.normalization = normalization
        self.MONTHS = [
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
        pd.set_option("display.max_columns", None)

    def __del__(self):
        del self.df
        del self.unique_make
        del self.general_graphs

    def __get_overall(self):
        self.overall = pd.DataFrame(
            {
                "Count": self.df.count(),
                "Difference": len(self.df) - self.df.count(),
                "%Difference": ((len(self.df) - self.df.count()) * 100) / len(self.df),
                "Unique": self.df.nunique(),
                "%Unique": self.df.nunique() * 100 / len(self.df),
                "Duplicated": self.df.duplicated().sum(),
                "%Duplicated": self.df.duplicated().sum() * 100 / len(self.df),
                "Null": self.df.isna().sum(),
                "%Null": self.df.isnull().sum() * 100 / len(self.df),
                "type": self.df.dtypes,
            }
        )

    def __get_basic_info(self):
        print("Generating basic info")
        self.df = self.extraction_data.process()
        output_result = OutputResult(self.output)

        output_result.title = "Get Initial data"
        output_result.process(self.df.head())
        output_result.title = "Get shape"
        output_result.process(f"The shape of the dataset is: {self.df.shape}")
        output_result.title = "Get description"
        output_result.process(self.df.describe(include="all").T)
        output_result.title = "Get overall"
        output_result.process(self.overall)

    def __get_nulls(self):
        print("Generating graph nulls")
        self.general_graphs.title = "Quantity Nulls per Column"
        self.general_graphs.heatmap_nulls(self.df)

    def __get_nulls_percentages(self):
        print("Generating graph nulls percentages")
        df_graph = round((self.df.isnull().sum() * 100 / len(self.df)), 2)
        self.general_graphs.title = "Percentage Nulls per Columns"
        self.general_graphs.x_rotation = 90
        self.general_graphs.bar_fontsize = 15
        self.general_graphs.bar_plot(
            df_graph.index, df_graph.values, "Columns", "Percentage"
        )

    def __get_correlation(self):
        print("Generating correlation graph")
        self.general_graphs.title = "Correlation of Features"
        self.general_graphs.heat_map_corr(self.df.corr())

    def __get_number_ads_by_city(self):
        print("Generating Graph: Number of Ads by City")
        self.general_graphs.title = "Number of Ads by City"
        self.general_graphs.x_rotation = 90
        self.general_graphs.count_bar_plot(self.df["city"], "City", "Count")

    def __get_number_ads_by_year(self):
        print("Generating Graph: Number of Ads by Year")
        self.general_graphs.title = "Number of Ads by Year"
        self.general_graphs.count_bar_plot(self.df["c_year"], "Year", "Count")

    def __get_number_ads_by_month(self):
        print("Generating Graph: Number of Ads by Month")
        self.general_graphs.title = "Number of Ads by Month"
        self.general_graphs.count_bar_plot(
            self.df["c_month_name"], "Month", "Count", self.MONTHS
        )

    def __get_number_of_visits_by_month(self):
        print("Generating Graph: Number of Visits by Month")

        df_graph = self.df.groupby(by=["c_month_name"], as_index=False)["visit"].sum()

        self.general_graphs.title = "Number of Visits by Month"
        self.general_graphs.bar_fontsize = 10
        self.general_graphs.bar_plot(
            df_graph["c_month_name"],
            df_graph["visit"],
            "Month",
            "Visits",
            None,
            self.MONTHS,
        )

        del df_graph

    def __get_number_of_visits_by_month_year(self):
        print("Generating Graph: Number of Visits by Month and Year")

        # Create a data frame to summarize the visits to year and month
        df_graph = self.df[["visit", "c_year", "c_month_name", "c_month"]]
        df_graph = df_graph.groupby(
            ["c_year", "c_month_name", "c_month"], as_index=False
        )["visit"].sum()

        self.general_graphs.title = "Monthly Visits by year"
        self.general_graphs.column_huge_name = "Year"
        self.general_graphs.line_plot(
            df_graph["c_month"], df_graph["visit"], df_graph["c_year"], "Month", "Visit"
        )

        del df_graph

    def __get_number_of_visits_by_year_make_car(self):
        print("Generating Graph: Number of Visits by Year (Make Car)")

        # Create a data frame to summarize the visits to year and month
        df_graph = self.df[["visit", "c_year", "make"]]
        df_graph = df_graph.groupby(["c_year", "make"], as_index=False)["visit"].sum()

        self.general_graphs.title = "Make (visit) by year"
        self.general_graphs.bar_fontsize = 5
        self.general_graphs.x_rotation = 90
        self.general_graphs.bar_rotation = 90
        self.general_graphs.column_huge_name = "Year"
        self.general_graphs.bar_plot(
            df_graph["make"], df_graph["visit"], "Make", "visit", df_graph["c_year"]
        )

        del df_graph

    def __get_number_of_kilometers_by_year_make_car(self):
        print("Generating Graph: Number of Kilometers by Year (Make Car)")

        df_graph = self.df[["Kilometers", "c_year", "make"]]
        df_graph = df_graph.groupby(["c_year", "make"], as_index=False)[
            "Kilometers"
        ].mean()

        self.general_graphs.title = "Make (Kilometers) by year"
        self.general_graphs.bar_fontsize = 5
        self.general_graphs.x_rotation = 90
        self.general_graphs.bar_rotation = 90
        self.general_graphs.column_huge_name = "Year"
        self.legend_loc = "upper center"
        self.general_graphs.bar_plot(
            df_graph["make"],
            df_graph["Kilometers"],
            "Make",
            "Kilometers",
            df_graph["c_year"],
        )

        del df_graph

    def __get_media_price_by_year_make_car(self):
        print("Generating Graph: Media Price by Year (Make Car)")

        df_graph = self.df[["price", "c_year", "make"]]
        df_graph = df_graph.groupby(["c_year", "make"], as_index=False)["price"].mean()

        self.general_graphs.title = "Make (Mean Price) by year"
        self.general_graphs.bar_fontsize = 6
        self.general_graphs.x_rotation = 90
        self.general_graphs.bar_rotation = 90
        self.general_graphs.column_huge_name = "Year"
        self.legend_loc = "upper center"
        self.general_graphs.bar_plot(
            df_graph["make"], df_graph["price"], "Make", "Price", df_graph["c_year"]
        )

        del df_graph

    def __get_media_price_by_year(self):
        print("Generating Graph: Media Price by Year")

        df_graph = self.df[["price", "c_year"]]
        df_graph = df_graph.groupby(["c_year"], as_index=False)["price"].mean()

        self.general_graphs.title = "Price per year"
        self.general_graphs.bar_plot(
            df_graph["c_year"], df_graph["price"], "Year", "Price"
        )

        del df_graph

    def __get_shared_of_visit_by_year(self):
        print("Generating Graph: Shared-of-Visit by Year")

        df_graph = self.df[["visit", "c_year"]].groupby("c_year").sum()
        total_visits = df_graph.sum()

        self.general_graphs.title = "Share-of-Visit by year"
        self.general_graphs.count_pie_plot(
            df_graph,
            "visit",
            (0, 0, 0.1),
            ["#ff9999", "#66b3ff", "#ffcc99"],
            f'TOTAL:\n {"{:,}".format(total_visits[0])}',
        )

        del df_graph

    def __get_model_car_by_year(self):
        print("Generating Graph: Model Car by year Car")

        self.general_graphs.title = "Model Car by year"
        self.general_graphs.x_rotation = 90
        self.general_graphs.x_fontsize = 14
        self.general_graphs.bar_rotation = 90
        self.general_graphs.count_bar_plot(self.df["year"], "Year", "Count")

    def __get_mean_price_model_car_by_year(self):
        print("Generating Graph: Mean Price per model year")

        self.general_graphs.title = "Mean Price per model year"
        self.general_graphs.x_rotation = 90
        self.general_graphs.x_fontsize = 14
        self.general_graphs.bar_rotation = 90
        self.general_graphs.bar_plot(self.df["year"], self.df["price"], "Year", "Count")

    def __get_price_density(self, subtitle):
        print(f"Generating Graph: {subtitle} Price density")

        self.general_graphs.title = f"{subtitle} Price density"
        self.general_graphs.distribution_kdeplot(self.df["price"], "price", "density")

    def __get_covariance(self):
        output_result = OutputResult(self.output)
        output_result.title = "Get covariance"
        output_result.process(self.df["price"].cov(self.df["visit"]))

        self.general_graphs.title = "Correlation Price"
        self.general_graphs.count_scatter(
            self.df["price"], self.df["visit"], "price", "visit"
        )
        self.general_graphs.title = "Correlation Price per Year"
        self.general_graphs.column_huge_name = "c_year"
        self.general_graphs.count_lmplot(self.df, "price", "visit", "Price", "Visits")

    def __get_correlation(self):
        self.general_graphs.title = "Correlation of Features"
        self.general_graphs.heat_map_corr(
            self.df.loc[
                :, ~self.df.columns.isin(["c_month", "c_year", "visit", "c_month_name"])
            ].corr()
        )

    def __get_correlation_cleaned(self):
        le_drivetrain = LabelEncoder()
        le_fuel_type = LabelEncoder()
        le_model = LabelEncoder()
        le_make = LabelEncoder()

        self.df["Drivetrain"] = le_drivetrain.fit_transform(self.df["Drivetrain"])
        self.df["fueltype"] = le_fuel_type.fit_transform(self.df["fueltype"])
        self.df["model"] = le_model.fit_transform(self.df["model"])
        self.df["make"] = le_make.fit_transform(self.df["make"])

        self.general_graphs.title = "Correlation of Features Cleaned"
        self.general_graphs.heat_map_corr(
            self.df.loc[
                :, ~self.df.columns.isin(["c_month", "c_year", "visit", "c_month_name"])
            ].corr()
        )

        self.df["Drivetrain"] = le_drivetrain.inverse_transform(self.df["Drivetrain"])
        self.df["fueltype"] = le_fuel_type.inverse_transform(self.df["fueltype"])
        self.df["model"] = le_model.inverse_transform(self.df["model"])
        self.df["make"] = le_make.inverse_transform(self.df["make"])

    def __get_facet_grid_models(self):
        self.general_graphs.title = "Density make vs price"
        self.general_graphs.facet_grid(self.df[["make", "price"]], "make", "price")

    def process(self):
        self.__get_basic_info()
        self.__get_nulls()
        self.__get_nulls_percentages()
        self.__get_correlation()
        self.__get_overall()

        self.df, self.unique_make = CleansingData(
            self.output, self.df, self.overall
        ).process()

        self.df = TransformationData(self.df, self.unique_make).process()
        self.df = CleansingData(self.output, self.df, self.overall).delete_primary_key()

        self.__get_number_ads_by_city()
        self.__get_number_ads_by_year()
        self.__get_number_ads_by_month()
        self.__get_number_of_visits_by_month()
        self.__get_number_of_visits_by_month_year()
        self.__get_number_of_visits_by_year_make_car()
        self.__get_number_of_kilometers_by_year_make_car()
        self.__get_media_price_by_year_make_car()
        self.__get_media_price_by_year()
        self.__get_shared_of_visit_by_year()
        self.__get_model_car_by_year()
        self.__get_mean_price_model_car_by_year()

        self.__get_price_density("Before")
        self.df = CleansingData(self.output, self.df, self.overall).identify_outliers(
            self.unique_make
        )
        self.__get_price_density("After")
        self.__get_covariance()
        self.__get_correlation_cleaned()
        self.__get_facet_grid_models()
        del self.extraction_data
        del self.overall

        return self.df, self.unique_make
