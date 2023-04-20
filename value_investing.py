import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class StockData:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def preprocess_data(self):
        self.data.fillna(0, inplace=True)
        return self.data


class ValueInvesting:
    def __init__(self, stock_data):
        self.stock_data = stock_data

    def get_value_stocks(self):
        value_stocks = self.stock_data[(self.stock_data['P/E Ratio'] < 15) &
                                       (self.stock_data['P/B Ratio'] < 1.5) &
                                       (self.stock_data['Dividend Yield'] > 2)]
        return value_stocks


class MachineLearning:
    def __init__(self, stock_data):
        self.stock_data = stock_data

    def engineer_features(self):
        features = self.stock_data.drop('Target', axis=1)
        target = self.stock_data['Target']

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, target, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        return classifier

    def evaluate_model(self, classifier, X_test, y_test):
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report


# Gather stock data
stock_data = StockData('stock_data.csv')
preprocessed_data = stock_data.preprocess_data()

# Value investing principles
value_investing = ValueInvesting(preprocessed_data)
value_stocks = value_investing.get_value_stocks()

# Machine learning data mining
ml = MachineLearning(value_stocks)
X_train, X_test, y_train, y_test = ml.engineer_features()
classifier = ml.train_model(X_train, y_train)
accuracy, report = ml.evaluate_model(classifier, X_test, y_test)

print("Accuracy:", accuracy)
print("Report:", report)
