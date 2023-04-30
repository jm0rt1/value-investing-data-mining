import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


class BaseModel:
    def __init__(self, model):
        self.model = model

    def preprocess_data(self, data):
        X = data[['Cash_Flow', 'Book_Value', 'Earnings']]
        y = data['Stock_Returns']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2


class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(LinearRegression())


class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__(RandomForestRegressor(random_state=42))


class SupportVectorMachineModel(BaseModel):
    def __init__(self):
        super().__init__(SVR(kernel='linear'))


class StockReturnPredictor:
    def __init__(self, data_file):
        self.data_file = data_file

    def run(self):
        data = pd.read_csv(self.data_file)
        models = [
            ('Linear Regression', LinearRegressionModel()),
            ('Random Forest', RandomForestModel()),
            ('Support Vector Machine', SupportVectorMachineModel())
        ]

        for name, model in models:
            X_train, X_test, y_train, y_test = model.preprocess_data(data)
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            mse, r2 = model.evaluate(y_test, y_pred)

            print(f"{name}:")
            print(f"Mean squared error: {mse:.2f}")
            print(f"R-squared: {r2:.2f}")
            print("-----------------------------")


if __name__ == "__main__":
    predictor = StockReturnPredictor('your_data.csv')
    predictor.run()
