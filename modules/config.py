"""Project-wide constants: paths, dataset facts and modelling settings."""

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "VISITS.csv"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUT_DIR = REPORTS_DIR / "output"
MODELS_DIR = REPORTS_DIR / "models"
DATABASE_FILE = ROOT_DIR / "result.sqlite"

DATE_FORMAT = "%m/%d/%Y, %H:%M:%S"

# The dataset is a single web-scraping snapshot. Every ad is still "live" on this
# date, so visits are cumulative up to it and ad age must be measured against it.
SCRAPE_DATE = pd.Timestamp("2022-08-27 23:59:59")

TARGET = "visit"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
