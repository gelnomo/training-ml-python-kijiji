import numpy as np

from modules.cleansing import FLAG_COLUMNS, clean_data


def test_clean_data_report_and_rows(raw_ads):
    df, report = clean_data(raw_ads)

    assert report["duplicates_removed"] == 1
    assert report["empty_columns_removed"] == 4
    assert report["rows_without_target_removed"] == 6
    assert report["rows_clean"] == len(df) == len(raw_ads) - 1 - 6
    assert df["visit"].notna().all()
    assert not {"latitude", "longitude", "mapAddress", "VIN"} & set(df.columns)


def test_placeholders_become_missing(raw_ads):
    df, report = clean_data(raw_ads)
    row = df.set_index("AdID")

    assert np.isnan(row.loc["1", "price"]) and row.loc["1", "price_missing"] == 1
    assert np.isnan(row.loc["2", "Kilometers"])
    assert np.isnan(row.loc["3", "year"])
    assert df["No of seats"].max() <= 9
    assert report["invalid_price_set_nan"] == 1


def test_text_normalisation_and_flags(raw_ads):
    df, _ = clean_data(raw_ads)

    assert set(df["make"]) == {"Ford", "Toyota", "Alfa Romeo"}
    assert "City of Toronto" in set(df["city"])
    assert set(df["Transmission"].dropna()) <= {"none", "manual", "automatic", "other"}
    for column in FLAG_COLUMNS:
        assert set(df[column]) <= {0, 1}


def test_derived_features(raw_ads):
    df, _ = clean_data(raw_ads)

    assert (df["ad_age_days"] >= 0).all()
    assert (df["car_age"].dropna() >= 0).all()
    assert df["publish_weekday"].notna().all()
