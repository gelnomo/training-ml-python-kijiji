from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Linear Regression
def train_linear_regression(data, target, exclude_columns, train_size, random_state):
    try:
        x = data.drop(columns=exclude_columns)
        y = data[target]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, random_state=random_state
        )
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        data["lr score train"] = lr.score(x_train, y_train)
        data["lr score test"] = lr.score(x_test, y_test)
        return lr
    except Exception as ex:
        return None
