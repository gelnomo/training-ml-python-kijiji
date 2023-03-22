import pandas as pd
import sqlite3
import pandas as pd
from typing import List

from modules.enums import OutlierTypeEnum
from modules.outliers import OutliersData


def find_duplicates(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Finds the duplicate entries of a data frame given a column name
    :param data: the data frame to check the duplicate values
    :param column_name: the column to check for duplicates
    :return: either False (no duplicates) or True (duplicates)
    """
    response = data[column_name].duplicated().any()
    return response


def get_pivot(
    data: pd.DataFrame, list_of_index: List, list_of_columns: List, value_column: str
) -> pd.DataFrame:
    """
    Generate pivot dataframe base on conditions
    :param data: initial dataframe
    :param list_of_index: list indexes (unique value)
    :param list_of_columns: list columns to pivot
    :param value_column: value to pivot
    :return: pivot dataframe
    """
    pivot_df = data.copy()
    pivot_df["unique_index"] = data[list_of_index].astype(str).agg("-".join, axis=1)

    pivot_df = pd.pivot_table(
        data=pivot_df,
        index=["unique_index"],
        columns=list_of_columns,
        values=value_column,
    ).reset_index()

    return pivot_df


def get_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall dataset statistics
    :param data: initial dataframe
    :return: summary about duplicated values
    """

    summary_df = pd.DataFrame(
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

    return summary_df


def get_outliers(
    data: pd.DataFrame, column_name: str, outlier_type: OutlierTypeEnum
) -> List:
    """
    Get outliers based on dataframe
    :param data: initial dataframe
    :param column_name: column name to identify outliers
    :param outlier_type: type of model to identify outliers
    :return: list of outliers
    """

    data_filtered = data[column_name]
    outlier_data = OutliersData(data_filtered)

    if outlier_type == OutlierTypeEnum.Z_SCORE:
        return outlier_data.z_score_outlier()
    if outlier_type == OutlierTypeEnum.Z_R_SCORE:
        return outlier_data.zr_score_outlier()
    if outlier_type == OutlierTypeEnum.DBSCAN:
        return outlier_data.db_outliers()
    if outlier_type == OutlierTypeEnum.WINSORIZATION:
        return outlier_data.winsorization_outliers()
    if outlier_type == OutlierTypeEnum.ISOLATION_FOREST:
        return outlier_data.iso_outliers()
    if outlier_type == OutlierTypeEnum.IQR:
        return outlier_data.iqr_outliers()

    return []


def dummify(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Convert values into columns with get_dummies function
    :param data: initial dataframe
    :param column_name: column name
    :return: dummified dataframe
    """
    dummies = pd.get_dummies(data[column_name], prefix=column_name, dummy_na=True)
    dummified_df = pd.concat([data, dummies], axis=1)
    dummified_df.drop([column_name], axis=1, inplace=True)

    return dummified_df


def undummify(data: pd.DataFrame, prefix_sep: str = "_") -> pd.DataFrame:
    """
    Revert dummies to rows
    :param data: initial dataframe
    :param prefix_sep: prefix to identify values in columns names
    :return: undummify dataframe
    """
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in data.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                data.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(data[col])
    undummified_df = pd.concat(series_list, axis=1)

    return undummified_df


def storage_sqlite(data: pd.DataFrame, table_name: str) -> None:
    """
    Storage result on sqlite database
    :param data: initial dataframe
    :param table_name: name table to save or rewrite
    """
    database = "result.sqlite"
    conn = sqlite3.connect(database)
    data.to_sql(table_name, conn, if_exists="replace")
    conn.close()
