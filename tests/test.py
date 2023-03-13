import unittest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

from features.utils import dummify


class TestCarPricePrediction(unittest.TestCase):
    def setUp(self):
        # Load the test data
        self.car_data_test = pd.read_csv("../data/input/data.csv")
        self.car_data_test = self.car_data_test[self.car_data_test["make"] == "Audi"]
        self.car_data_test = self.car_data_test[
            (self.car_data_test["make"] == "Audi")
            & (~self.car_data_test["model"].isin(["autre", "r8", "rs6"]))
        ]

        # Load the model
        self.model = joblib.load("../data/output/job_lib/RFR_Audi.joblib")
        self.X_test = self.car_data_test

    def test_prediction(self):
        columns_to_capitalize = ["make", "model", "condition"]
        self.X_test[columns_to_capitalize] = self.X_test[columns_to_capitalize].apply(
            lambda x: x.str.title().str.strip().str.replace("_", " ")
        )
        self.X_test = dummify(self.X_test, "fueltype")
        self.X_test = dummify(self.X_test, "condition")
        self.X_test = dummify(self.X_test, "model")
        self.X_test = dummify(self.X_test, "city")
        self.X_test = dummify(self.X_test, "Drivetrain")
        self.X_test.drop(
            columns=[
                "latitude",
                "make",
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
                "visit",
                "title",
                "Kilometers",
                "Date",
                "AdID",
            ],
            inplace=True,
        )
        self.X_test.drop_duplicates(inplace=True)
        self.X_test.dropna(inplace=True)
        self.y_test = self.X_test["price"]
        self.X_test = self.X_test.drop(["price"], axis=1)

        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Check the shape of predictions
        self.assertEqual(y_pred.shape[0], self.X_test.shape[0])

        # Check the range of predictions
        self.assertTrue(all(y_pred > 0))

        # Evaluate the predictions using RMSE
        y_orig = self.y_test.to_numpy()
        rmse = np.sqrt(mean_squared_error(y_orig, y_pred))
        self.assertLessEqual(rmse / 1000, 1000)
