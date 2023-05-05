import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class BaseModel:
    def __init__(self, model):
        self.model = model

    def preprocess_data(self, data):
        data['Earnings_to_Book_Value'] = data['Earnings'] / data['BookValue']

        X = data[['CashFlow', 'BookValue',
                  'Earnings', 'Earnings_to_Book_Value']]
        y = data['5YrReturn%']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def feature_selection(self, X_train, y_train, n_features):
        # If the model is an instance of RandomizedSearchCV, fit it first
        if isinstance(self.model, RandomizedSearchCV):
            self.model.fit(X_train, y_train)
            best_estimator = self.model.best_estimator_
        else:
            best_estimator = self.model

        selector = RFE(best_estimator, n_features_to_select=n_features)
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

    # Add selector as an argument in predict_single_stock_return method
    def predict_single_stock_return(self, stock_data, selector):
        scaler = self.scaler
        stock_data_scaled = scaler.transform(stock_data)
        stock_data_selected = selector.transform(
            stock_data_scaled)  # Apply feature selection
        return self.model.predict(stock_data_selected)


class LinearRegressionModel(BaseModel):
    def __init__(self):
        model = Pipeline([('regressor', Ridge())])
        param_dist = {
            'regressor__alpha': np.logspace(-3, 3, 7)
        }
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=7, cv=5, n_jobs=-1, random_state=42)
        super().__init__(random_search)


class RandomForestModel(BaseModel):
    def __init__(self):
        param_dist = {
            'n_estimators': np.arange(50, 310, 10),  # Updated range
            'max_depth': [None] + list(np.arange(5, 110, 5)),  # Updated range
            'min_samples_split': [2, 3, 5, 7, 10],  # Updated range
            'min_samples_leaf': [1, 2, 3, 4, 5],  # Updated range
            'max_features': [1.0, 'sqrt', 'log2'],  # Updated 'auto' to 1.0
            'bootstrap': [True, False]  # Added bootstrap
        }
        model = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=200, cv=10, n_jobs=-1, random_state=42)  # Updated cv to 10
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
            # Save the scaler for later use in prediction
            model.scaler = StandardScaler().fit(X_train)
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

            # Predict the next 5 years return for each stock
            unique_tickers = data['Ticker'].unique()
            print(f"Predicted 5-year returns for {name}:")
            for ticker in unique_tickers:
                stock_data = data[data['Ticker'] == ticker]
                # Use the most recent data for prediction
                stock_data = stock_data.iloc[0]
                stock_features = stock_data[[
                    'CashFlow', 'BookValue', 'Earnings']].to_numpy().reshape(1, -1)
                # Calculate Earnings_to_Book_Value
                stock_features = np.column_stack(
                    (stock_features, stock_features[:, 2] / stock_features[:, 1]))
               # Pass selector to predict_single_stock_return method
                predicted_return = model.predict_single_stock_return(
                    stock_features, selector)
                # ...
                print(f"{ticker}: {predicted_return[0]:.2f}")


if __name__ == "__main__":
    predictor = StockReturnPredictor('out.csv')
    predictor.run()
