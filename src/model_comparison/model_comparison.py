import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm  # Import the tqdm package
from sklearn.preprocessing import StandardScaler, PowerTransformer

from src.model_comparison.LinearRegressionModel import LinearRegressionModel
from src.model_comparison.RandomForestModel import RandomForestModel
from src.model_comparison.StackingModel import StackingModel
from src.model_comparison.SupportVectorMachineModel import SupportVectorMachineModel
warnings.filterwarnings('ignore')


class StockReturnPredictor:
    def __init__(self, data_file: str, output_file: str, time_file: str):
        self.data_file = data_file
        self.output_file = output_file
        self.write_line_called = False
        self.time_file = time_file  # Added time_file parameter

    def write_line_to_file(self, msg: str) -> None:
        mode = 'w' if not self.write_line_called else 'a'
        with open(self.output_file, mode) as f:
            f.write(msg + '\n')

        # Set the function attribute so that we know it's been called
        self.write_line_called = True

    def calculate_growth(self, data):
        data['CashFlow_Growth'] = data.groupby(
            'Ticker')['CashFlow'].pct_change()
        data['BookValue_Growth'] = data.groupby(
            'Ticker')['BookValue'].pct_change()
        data['Earnings_Growth'] = data.groupby(
            'Ticker')['Earnings'].pct_change()
        return data
    # New method to write model times to file

    def write_time_to_file(self, msg: str) -> None:
        mode = 'w' if not self.write_line_called else 'a'
        with open(self.time_file, mode) as f:
            f.write(msg + '\n')
        self.write_line_called = True

    def run(self):
        data = pd.read_csv(self.data_file)
        data = self.calculate_growth(data)
        models = [
            ('Linear Regression', LinearRegressionModel()),
            ('Random Forest', RandomForestModel()),
            ('Support Vector Machine', SupportVectorMachineModel())
        ]

        stacking_model = StackingModel(
            [(name, model.model) for name, model in models])
        models.append(('Stacking', stacking_model))

        for name, model in tqdm(models, desc="Models", unit="model"):
            X_train, X_test, y_train, y_test = model.preprocess_data(data)
            model.scaler = StandardScaler().fit(X_train)

            # Fit the model before calling feature_selection
            model.model.fit(X_train, y_train)

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

            self.write_line_to_file(f"{name}:")
            self.write_line_to_file(f"Mean squared error: {mse:.2f}")
            self.write_line_to_file(f"R-squared: {r2:.2f}")
            self.write_line_to_file(f"Cross-validation score: {cv_score:.2f}")
            self.write_line_to_file("-----------------------------")

            # Write model training and prediction times to file
            self.write_time_to_file(
                f"Training time for {name}: {model.training_time}")
            self.write_time_to_file(
                f"Prediction time for {name}: {model.prediction_time}")
            self.write_time_to_file("-----------------------------")

            # Predict the next 5 years return for each stock
            unique_tickers = data['Ticker'].unique()
            self.write_line_to_file(f"Predicted 5-year returns for {name}:")
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
                self.write_line_to_file(f"{ticker}: {predicted_return[0]:.2f}")


def main():
    predictor = StockReturnPredictor(
        "data/input_data.csv", "output.txt", "times.txt")
    predictor.run()


if __name__ == "__main__":
    main()
