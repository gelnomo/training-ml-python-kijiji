"""Data cleaning and feature derivation for the Kijiji ads snapshot."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from modules.config import SCRAPE_DATE, TARGET

# Equipment flags only contain 1 or NaN in practice: the listing either advertises the
# option or says nothing. NaN is therefore treated as "not advertised" (0).
FLAG_COLUMNS = ["push button start", "parking assist", "sunroof", "alloy wheels"]

LOWERCASE_COLUMNS = ["saleby", "model", "condition", "body", "Drivetrain", "fueltype", "color"]

# Domain sanity limits. Values outside them are placeholders ("$1, call us",
# 9,999,999 km, year 1900) and are set to NaN instead of dropping the whole ad.
MIN_VALID_PRICE = 100
MAX_VALID_PRICE = 500_000
MAX_VALID_KM = 1_000_000
MIN_VALID_YEAR = 1940
MAX_VALID_SEATS = 9

TRANSMISSION_LABELS = {0: "none", 1: "manual", 2: "automatic", 3: "other"}


def clean_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean the raw ads and derive model features
    :param data: raw dataframe (see ``extraction.read_visits``)
    :return: cleaned dataframe and a report with the number of rows/values affected per step
    """
    df = data.copy()
    report: Dict[str, int] = {"rows_raw": len(df)}

    # 1. Duplicates: exact copies and re-posted AdIDs
    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="AdID", keep="first")
    report["duplicates_removed"] = before - len(df)

    # 2. Columns without a single value (latitude, longitude, mapAddress, VIN)
    empty_columns = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=empty_columns)
    report["empty_columns_removed"] = len(empty_columns)

    # 3. Never impute the target: ads without a visit counter are unusable for training
    before = len(df)
    df = df[df[TARGET].notna()]
    report["rows_without_target_removed"] = before - len(df)

    # 4. Consistent text categories
    df["city"] = df["city"].str.strip()
    df["make"] = df["make"].str.strip().str.replace("_", " ").str.title()
    df.loc[df["make"] == "Alfaromeo", "make"] = "Alfa Romeo"
    for column in LOWERCASE_COLUMNS:
        df[column] = df[column].str.strip().str.lower()

    # 5. Numeric fields stored as text ("others" -> NaN) and impossible values
    for column in ["No of door", "No of seats"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df.loc[df["No of seats"] > MAX_VALID_SEATS, "No of seats"] = np.nan
    df[FLAG_COLUMNS] = df[FLAG_COLUMNS].fillna(0).astype(int)
    df["Transmission"] = df["Transmission"].map(TRANSMISSION_LABELS)

    invalid_price = (df["price"] < MIN_VALID_PRICE) | (df["price"] > MAX_VALID_PRICE)
    report["invalid_price_set_nan"] = int(invalid_price.sum())
    df["price_missing"] = invalid_price.astype(int)
    df["price"] = df["price"].where(~invalid_price).astype(float)

    invalid_km = df["Kilometers"] > MAX_VALID_KM
    report["invalid_km_set_nan"] = int(invalid_km.sum())
    df["Kilometers"] = df["Kilometers"].where(~invalid_km).astype(float)

    invalid_year = (df["year"] < MIN_VALID_YEAR) | (df["year"] > SCRAPE_DATE.year + 1)
    report["invalid_year_set_nan"] = int(invalid_year.sum())
    df["year"] = df["year"].where(~invalid_year).astype(float)

    # 6. Derived features
    df["ad_age_days"] = (SCRAPE_DATE - df["Date"]).dt.days.clip(lower=0)
    df["publish_month"] = df["Date"].dt.month
    df["publish_weekday"] = df["Date"].dt.day_name()
    df["car_age"] = (df["Date"].dt.year - df["year"]).clip(lower=0)
    df["km_per_year"] = df["Kilometers"] / (df["car_age"] + 1)
    df["title_length"] = df["title"].fillna("").str.len()

    report["rows_clean"] = len(df)
    return df.reset_index(drop=True), report
