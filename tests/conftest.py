import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_ads() -> pd.DataFrame:
    """Small synthetic dataset with the raw Kijiji schema and its known data issues."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.Timestamp("2022-08-27") - pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    year = rng.integers(2000, 2023, n)
    age_days = (pd.Timestamp("2022-08-27") - dates).days.to_numpy()
    df = pd.DataFrame(
        {
            "AdID": [str(i) for i in range(n)],
            "city": rng.choice(["Calgary", "Ottawa", "City of Toronto "], n),
            "latitude": np.nan,
            "longitude": np.nan,
            "mapAddress": np.nan,
            "title": [f"{y} Car" for y in year],
            "price": rng.integers(2_000, 80_000, n),
            "saleby": rng.choice(["dealer", "owner"], n),
            "make": rng.choice(["Ford", "toyota", "Alfaromeo"], n),
            "model": rng.choice(["focus", "corolla", "giulia"], n),
            "Trim": "SE",
            "year": year,
            "condition": rng.choice(["used", "new"], n),
            "Kilometers": rng.integers(0, 300_000, n),
            "body": rng.choice(["sedan", "suv_crossover"], n),
            "Drivetrain": rng.choice(["fwd", "awd", None], n),
            "fueltype": rng.choice(["gas", "hybrid"], n),
            "color": rng.choice(["black", "white"], n),
            "Date": dates,
            # visits grow with ad age and car age, plus noise
            "visit": np.round(np.expm1(1 + 0.02 * age_days + 0.08 * (2022 - year) + rng.normal(0, 0.5, n))),
            "No of door": rng.choice(["4", "2", "others"], n),
            "No of seats": rng.choice(["5", "7", "44"], n),
            "Transmission": rng.choice([1.0, 2.0, np.nan], n),
            "push button start": rng.choice([1.0, np.nan], n),
            "parking assist": rng.choice([1.0, np.nan], n),
            "sunroof": rng.choice([1.0, np.nan], n),
            "alloy wheels": rng.choice([1.0, np.nan], n),
            "VIN": np.nan,
        }
    )
    # Known issues: a duplicated ad, missing targets and placeholder values
    df.loc[1, "price"] = 1
    df.loc[2, "Kilometers"] = 9_999_999
    df.loc[3, "year"] = 1900
    df.loc[4:9, "visit"] = np.nan
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)
