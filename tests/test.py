import unittest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

from modules.utils import dummify


class TestCarPricePrediction(unittest.TestCase):
    def setUp(self):
        # Load the test data
        custom_date_parser = lambda c: pd.to_datetime(
            c, format="%m/%d/%Y, %H:%M:%S", errors="coerce"
        )
        self.visit_data_test = pd.read_csv("../data/VISITS.csv", parse_dates=["Date"], date_parser=custom_date_parser)
        self.visit_data_test = self.visit_data_test[
            (self.visit_data_test["city"] == "UBC") & (~self.visit_data_test["make"].isin(["Bmw", "Toyota"]))]

        # Load the model
        self.model = joblib.load("../reports/job_lib/GBR_Ubc.joblib")
        self.X_test = self.visit_data_test

    def test_prediction(self):
        columns_to_capitalize = ["make", "condition", "city"]
        self.X_test[columns_to_capitalize] = self.X_test[columns_to_capitalize].apply(
            lambda x: x.str.title().str.strip().str.replace("_", " ")
        )
        self.X_test["cyear"] = self.X_test["Date"].dt.year
        self.X_test = dummify(self.X_test, "make")
        self.X_test = dummify(self.X_test, "condition")
        self.X_test.drop(
            columns=[
                "city",
                "fueltype",
                "price",
                "Drivetrain",
                "latitude",
                "model",
                "longitude",
                "mapAddress",
                "push button start",
                "parking assist",
                "sunroof",
                "alloy wheels",
                "saleby",
                "VIN",
                "No of door",
                "No of seats",
                "Trim",
                "body",
                "color",
                "title",
                "Kilometers",
                "Date",
                "AdID",
                "Transmission",
                "year",
            ],
            inplace=True,
        )
        self.X_test.drop_duplicates(inplace=True)
        self.y_test = self.X_test["visit"]
        self.X_test = self.X_test.drop(["visit"], axis=1)

        # Make predictions
        y_pred = self.model.predict(self.X_test)
        # Check the shape of predictions
        self.assertEqual(y_pred.shape[0], self.X_test.shape[0])

        # Evaluate the predictions using RMSE
        y_orig = self.y_test.to_numpy()
        rmse = np.sqrt(mean_squared_error(y_orig, y_pred))
        self.assertLessEqual(rmse / 1000, 1000)
