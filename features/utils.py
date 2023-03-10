import pandas as pd
import sqlite3


class PrintColor:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def dummify(data, column):
    dummies = pd.get_dummies(data[column], prefix=column, dummy_na=True)
    dummified_df = pd.concat([data, dummies], axis=1)
    dummified_df.drop([column], axis=1, inplace=True)

    return dummified_df


def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)

    return undummified_df


def storage_result(df_result, table_name):
    database = "result.sqlite"
    conn = sqlite3.connect(database)
    df_result.to_sql(table_name, conn, if_exists="replace")
    conn.close()
