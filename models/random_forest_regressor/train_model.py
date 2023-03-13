import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Random Fores Regressor
def train_random_forest_regressor(
    data, target, exclude_columns, train_size, random_state, is_joblib=False
):
    try:
        x = data.drop(columns=exclude_columns)
        if len(x) == 0:
            return None

        y = data[target]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, random_state=random_state
        )
        rfr = RandomForestRegressor(random_state=0)
        rfr.fit(x_train, y_train)
        data["rfr score train"] = rfr.score(x_train, y_train)
        data["rfr score test"] = rfr.score(x_test, y_test)

        if is_joblib:
            make = data["make"].unique()
            joblib.dump(rfr, f"data/output/job_lib/RFR_{make}.joblib")

        return rfr
    except Exception as ex:
        print(ex)
        return None
