import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


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

    def feature_selection(self, X_train, y_train, n_features):
        selector = RFE(self.model, n_features_to_select=n_features)
        selector.fit(X_train, y_train)
        return selector

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def cross_val(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv)
        return np.mean(scores)


class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(LinearRegression())


class RandomForestModel(BaseModel):
    def __init__(self):
        param_dist = {
            'n_estimators': np.arange(10, 210, 10),
            'max_depth': [None] + list(np.arange(10, 110, 10)),
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)
        super().__init__(random_search)


class SupportVectorMachineModel(BaseModel):
    def __init__(self):
        param_dist = {
            'C': np.logspace(-3, 3, 7),
            'epsilon': np.logspace(-3, 3, 7)
        }
        model = SVR(kernel='linear')
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)
        super().__init__(random_search)


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
            selector = model.feature_selection(X_train, y_train, n_features=3)
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)

            model.train(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
            mse, r2 = model.evaluate(y_test, y_pred)
            cv_score = model.cross_val(X_train_selected, y_train)

            print(f"{name}:")
            print(f"Mean squared error: {mse:.2f}")
            print(f"R-squared: {r2:.2f}")
            print(f"Cross-validation score: {cv_score:.2f}")
            print("-----------------------------")


if __name__ == "__main__":
    predictor = StockReturnPredictor('your_data.csv')
    predictor.run()
