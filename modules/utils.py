import re
import sqlite3
from pathlib import Path

import pandas as pd

from modules.config import DATABASE_FILE


def slugify(text: str) -> str:
    """
    Turn a title into a safe file name, e.g. "Visits by City (median)" -> "visits_by_city_median"
    :param text: free text
    :return: lowercase snake_case string
    """
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def storage_sqlite(data: pd.DataFrame, table_name: str, database: Path = DATABASE_FILE) -> None:
    """
    Store a dataframe in the SQLite database, replacing the table if it exists
    :param data: dataframe to store
    :param table_name: table name
    :param database: SQLite file path
    """
    with sqlite3.connect(database) as conn:
        data.to_sql(table_name, conn, if_exists="replace", index=False)
