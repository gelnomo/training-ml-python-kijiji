import pandas as pd
from datetime import datetime

from features.utils import storage_result


class ExtractionData:
    def __init__(self):
        # Define the path to the input file and the date format used in the file
        self.file = "data/input/data.csv"
        self.format_date = "%m/%d/%Y, %H:%M:%S"

    def process(self):
        # Define a custom date parser using the format specified in the constructor
        custom_date_parser = lambda x: datetime.strptime(x, self.format_date)

        # Read the CSV file using pandas, and apply the custom date parser to the "Date" column
        df = pd.read_csv(
            self.file, parse_dates=["Date"], date_parser=custom_date_parser
        )

        # Store the original dataframe using a utility function
        storage_result(df, "original")

        # df = df[df["make"] == "Suzuki"]

        # Return the resulting dataframe
        return df
