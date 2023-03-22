import joblib
from typing import List, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def train_random_forest_regressor(
    data: pd.DataFrame,
    target: str,
    exclude_columns: List[str],
    train_size: float,
    random_state: int,
    storage_model: bool = False,
) -> Tuple[RandomForestRegressor, float, float]:
    """
    Trains a RandomForestRegressor on a pandas DataFrame.

    Args:
        data: A pandas DataFrame containing the training data.
        target: The name of the column containing the target variable.
        exclude_columns: A list of column names to exclude from the training data.
        train_size: The proportion of the data to use for training.
        random_state: The random state to use for reproducibility.
        storage_model: Allow to save the model as a joblib.

    Returns:
        A tuple containing the trained RandomForestRegressor, the R^2 score on the training data,
        and the R^2 score on the testing data. Returns None if the input data is invalid or an exception occurs.
    """
    try:
        # Split data into training and testing sets
        x = data.drop(columns=exclude_columns)
        if len(x) == 0:
            return None, None, None

        y = data[target]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, random_state=random_state
        )

        # Train the LinearRegression
        gbr = RandomForestRegressor(random_state=0)
        gbr.fit(x_train, y_train)

        # Compute R^2 scores on the training and testing data
        train_score = gbr.score(x_train, y_train)
        test_score = gbr.score(x_test, y_test)

        if storage_model:
            city = data["city"].unique()
            joblib.dump(gbr, f"reports/job_lib/RFR_{'_'.join(city).replace('/', '-')}.joblib")

        # Return the trained model and the R^2 scores
        return gbr, train_score, test_score
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return None, None, None
