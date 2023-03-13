import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# Gradient Boosting Regressor
def train_gradient_boosting_regressor(
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
        gbr = GradientBoostingRegressor(random_state=0)
        gbr.fit(x_train, y_train)
        data["gbr score train"] = gbr.score(x_train, y_train)
        data["gbr score test"] = gbr.score(x_test, y_test)

        if is_joblib:
            make = data["make"].unique()
            joblib.dump(gbr, f"data/output/job_lib/GBR_{make}.joblib")

        return gbr
    except Exception as ex:
        print(ex)
        return None
