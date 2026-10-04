from pathlib import Path

import pandas as pd

from modules.config import DATA_FILE, DATE_FORMAT


def read_visits(path: Path = DATA_FILE) -> pd.DataFrame:
    """
    Read the raw Kijiji ads file
    :param path: CSV path
    :return: raw dataframe with ``Date`` parsed as datetime
    """
    df = pd.read_csv(path, dtype={"AdID": str})
    df["Date"] = pd.to_datetime(df["Date"], format=DATE_FORMAT, errors="coerce")
    return df
