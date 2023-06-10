import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm


import time
from datetime import timedelta


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
        # Fit the model first
        self.model.fit(X_train, y_train)
        best_estimator = self.model.best_estimator_

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

        # If the best_estimator is an SVM with a non-linear kernel, use SelectKBest for feature selection
        if isinstance(best_estimator, SVR) and best_estimator.kernel != 'linear':
            selector = SelectKBest(f_regression, k=n_features)
        else:
            selector = RFE(best_estimator, n_features_to_select=n_features,
                           importance_getter=importance_getter)
        selector.fit(X_train, y_train)
        return selector

    def train(self, X_train, y_train):
        start_time = time.monotonic()
        self.model.fit(X_train, y_train)
        end_time = time.monotonic()
        self.training_time = timedelta(seconds=end_time - start_time)

    def predict(self, X_test):
        y_pred = []
        start_time = time.monotonic()
        for i in tqdm(range(len(X_test)), desc="Predicting", unit="sample"):
            y_pred.append(self.model.predict(X_test[i].reshape(1, -1))[0])
        end_time = time.monotonic()
        self.prediction_time = timedelta(seconds=end_time - start_time)
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
