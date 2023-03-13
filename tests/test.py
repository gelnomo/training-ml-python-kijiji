import unittest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class TestCarPricePrediction(unittest.TestCase):
    def setUp(self):
        # Load the test data
        self.car_data_test = pd.read_csv("data/input/data.csv")
        self.car_data_test = self.car_data_test[self.car_data_test["make"] == "Audi"]
        self.X_test = self.car_data_test.drop(["price"], axis=1)
        self.y_test = self.car_data_test["price"]

        # Load the model
        self.model = joblib.load("data/output/job_lib/GBR_['Audi'].joblib")

    def test_prediction(self):
        # Encoding categorical variables
        le = LabelEncoder()
        self.X_test["model"] = le.fit_transform(self.X_test["model"])
        self.X_test["color"] = le.fit_transform(self.X_test["color"])
        ohe = OneHotEncoder(sparse=False)
        year_ohe = ohe.fit_transform(self.X_test["year"].values.reshape(-1, 1))
        self.X_test = pd.concat(
            [
                self.X_test,
                pd.DataFrame(
                    year_ohe, columns=["year_" + str(i) for i in range(2010, 2023)]
                ),
            ],
            axis=1,
        )
        self.X_test = self.X_test.drop(["year"], axis=1)

        # Normalizing the data
        scaler = StandardScaler()
        X_test_norm = scaler.fit_transform(self.X_test)

        # Make predictions
        y_pred = self.model.predict(X_test_norm)

        # Check the shape of predictions
        self.assertEqual(y_pred.shape[0], self.X_test.shape[0])

        # Check the range of predictions
        self.assertTrue(all(y_pred > 0))

        # Evaluate the predictions using RMSE
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        self.assertLessEqual(rmse, 1000)
