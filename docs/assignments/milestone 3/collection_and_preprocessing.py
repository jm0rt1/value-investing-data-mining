import yfinance as yf
import pandas as pd


class DataCollector:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def collect_data(self):
        data_frames = []
        for ticker in self.tickers:
            stock_data = yf.Ticker(ticker)
            financial_data = stock_data.history(
                start=self.start_date, end=self.end_date)
            financial_data['Ticker'] = ticker
            data_frames.append(financial_data)

        combined_data = pd.concat(data_frames, axis=0)
        return combined_data


class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        self.data = self.data.dropna()
        self.data = self.data.reset_index()

        return self.data


# Example usage
tickers = ['AAPL', 'GOOGL', 'MSFT']
start_date = '2020-01-01'
end_date = '2021-12-31'

collector = DataCollector(tickers, start_date, end_date)
raw_data = collector.collect_data()

preprocessor = DataPreprocessor(raw_data)
preprocessed_data = preprocessor.preprocess_data()
print(preprocessed_data)
