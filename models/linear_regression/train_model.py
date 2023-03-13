import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Linear Regression
def train_linear_regression(
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
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        data["lr score train"] = lr.score(x_train, y_train)
        data["lr score test"] = lr.score(x_test, y_test)

        if is_joblib:
            make = data["make"].unique()
            joblib.dump(lr, f"data/output/job_lib/LR_{make}.joblib")

        return lr
    except Exception as ex:
        print(ex)
        return None
