import pandas as pd
from datetime import datetime

from features.utils import storage_result


class ExtractionData:
    def __init__(self):
        self.file = "data/input/data.csv"
        self.format_date = "%m/%d/%Y, %H:%M:%S"

    def process(self):
        custom_date_parser = lambda x: datetime.strptime(x, self.format_date)
        df = pd.read_csv(
            self.file, parse_dates=["Date"], date_parser=custom_date_parser
        )
        #df = df[df["make"] == "Suzuki"]
        storage_result(df, "original")
        return df
