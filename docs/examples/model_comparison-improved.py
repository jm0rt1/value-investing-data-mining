import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Feature engineering by creating the Earnings_to_Book_Value ratio.
# Data normalization using StandardScaler.
# Hyperparameter tuning using GridSearchCV for the Random Forest and Support Vector Machine models.

class BaseModel:
    def __init__(self, model):
        self.model = model

    def preprocess_data(self, data):
        data['Earnings_to_Book_Value'] = data['Earnings'] / data['Book_Value']

        X = data[['Cash_Flow', 'Book_Value',
                  'Earnings', 'Earnings_to_Book_Value']]
        y = data['Stock_Returns']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

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
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        super().__init__(grid_search)


class SupportVectorMachineModel(BaseModel):
    def __init__(self):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 1, 10]
        }
        model = SVR(kernel='linear')
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        super().__init__(grid_search)


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
