import pandas as pd
from modules.utils import storage_sqlite
from typing import List


class ExtractionData:
    def __init__(self):
        self.path = "data"

        # Watch in console more columns in one line
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.width", 1000)

    def read(
        self, file_name: str, columns_datetime: List = None, format_date: str = None
    ) -> pd.DataFrame:
        """
        Read csv file and become the info to dataframe
        :param file_name: file name
        :param columns_datetime: columns to become as a datetime type
        :param format_date: format of dates
        :return: DataFrame object
        """

        # Convert datetime data type to a specific format
        custom_date_parser = lambda c: pd.to_datetime(
            c, format=format_date, errors="coerce"
        )

        # Read the CSV file using pandas, and apply the custom date parser to the "Date" column
        df = pd.read_csv(
            f"{self.path}/{file_name}",
            parse_dates=columns_datetime,
            date_parser=custom_date_parser,
        )

        # Store the original dataframe using a utility function
        storage_sqlite(df, file_name.split(".")[0].replace(" ", "_"))

        return df
