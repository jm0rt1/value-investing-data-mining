from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.pipeline import Pipeline
import warnings
from tqdm import tqdm  # Import the tqdm package
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')


class BaseModel:
    def __init__(self, model):
        self.model = model

    def preprocess_data(self, data):

        # Log-transform the skewed features
        skewed_features = ['CashFlow', 'BookValue', 'Earnings']
        pt = PowerTransformer(method='yeo-johnson')
        data[skewed_features] = pt.fit_transform(data[skewed_features])

        data['Earnings_to_Book_Value'] = data['Earnings'] / data['BookValue']

        X = data[['CashFlow', 'BookValue', 'Earnings', 'Earnings_to_Book_Value']]
        y = data['5YrReturn%']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Remove possible outliers
        q1 = y_train.quantile(0.25)
        q3 = y_train.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        inliers = (y_train >= lower_bound) & (y_train <= upper_bound)
        X_train_no_outliers = X_train.loc[inliers]
        y_train_no_outliers = y_train.loc[inliers]

        scaler = StandardScaler()
        X_train_no_outliers = scaler.fit_transform(X_train_no_outliers)
        X_test = scaler.transform(X_test)

        return X_train_no_outliers, X_test, y_train_no_outliers, y_test

    def feature_selection(self, X_train, y_train, n_features):
        # If the model is an instance of RandomizedSearchCV, fit it first
        if isinstance(self.model, RandomizedSearchCV):
            self.model.fit(X_train, y_train)
            best_estimator = self.model.best_estimator_
        else:
            best_estimator = self.model

        # Create a custom feature importance getter for pipelines
        def pipeline_feature_importances(pipeline):
            final_step = pipeline.steps[-1][1]
            if hasattr(final_step, 'coef_'):
                return np.abs(final_step.coef_)
            elif hasattr(final_step, 'feature_importances_'):
                return final_step.feature_importances_
            else:
                raise ValueError(
                    "The final step of the pipeline should have 'coef_' or 'feature_importances_' attribute.")

        # Check if the best_estimator is a pipeline, and if so, use the custom feature importance getter
        if isinstance(best_estimator, Pipeline):
            importance_getter = pipeline_feature_importances
        else:
            importance_getter = 'auto'

        selector = RFE(best_estimator, n_features_to_select=n_features,
                       importance_getter=importance_getter)
        selector.fit(X_train, y_train)
        return selector

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = []
        for i in tqdm(range(len(X_test)), desc="Predicting", unit="sample"):
            y_pred.append(self.model.predict(X_test[i].reshape(1, -1))[0])
        return np.array(y_pred)

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
    def __init__(self, data_file, output_file):
        self.data_file = data_file
        self.output_file = output_file

    def write_line_to_file(msg: str) -> None:

        mode = 'w' if not hasattr(
            write_line_to_file, "has_been_called") else 'a'
        with open(self.output_file, mode) as f:
            f.write(msg + '\n')

        # Set the function attribute so that we know it's been called
        self.write_line_to_file.has_been_called = True

    def run(self):
        data = pd.read_csv(self.data_file)
        models = [
            ('Linear Regression', LinearRegressionModel()),
            ('Random Forest', RandomForestModel()),
            ('Support Vector Machine', SupportVectorMachineModel())
        ]

        # Add a progress bar for the entire process
        for name, model in tqdm(models, desc="Models", unit="model"):
            X_train, X_test, y_train, y_test = model.preprocess_data(data)
            model.scaler = StandardScaler().fit(X_train)
            selector = model.feature_selection(X_train, y_train, n_features=3)
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)

            # Wrap the training step with tqdm to show progress
            with tqdm(total=1, desc=f"Training {name}", unit="step"):
                model.train(X_train_selected, y_train)
                tqdm.write("Training complete")

            y_pred = model.predict(X_test_selected)
            mse, r2 = model.evaluate(y_test, y_pred)
            cv_score = model.cross_val(X_train_selected, y_train)

            write_line_to_file(f"{name}:")
            write_line_to_file(f"Mean squared error: {mse:.2f}")
            write_line_to_file(f"R-squared: {r2:.2f}")
            write_line_to_file(f"Cross-validation score: {cv_score:.2f}")
            write_line_to_file("-----------------------------")
            # Predict the next 5 years return for each stock
            unique_tickers = data['Ticker'].unique()
            write_line_to_file(f"Predicted 5-year returns for {name}:")
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
                write_line_to_file(f"{ticker}: {predicted_return[0]:.2f}")


if __name__ == "__main__":
    predictor = StockReturnPredictor('out.csv', "output.txt")
    predictor.run()


for model_name, model in models:
    print(f"Training and evaluating {model_name}")
    X_train, X_test, y_train, y_test = model.preprocess_data(data)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    mse, r2, mae, medae, msle = model.evaluate(y_test, y_pred)
    print(f"{model_name} - MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, MedAE: {medae:.4f}, MSLE: {msle:.4f}")
