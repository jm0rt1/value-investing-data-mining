
from src.value_investing_strategy.strategy_system.analysis import Analyzer
from src.value_investing_strategy.strategy_system.stocks.stocks_in_use import StocksInUse
from src.value_investing_strategy.strategy_system.strategy_system import StrategySystem
from src.value_investing_strategy.strategy_system.stocks.stock.stock import Stock
import pandas as pd
class MLStrategy:

    def __init__(self, system: StrategySystem):
        pass

    def run_all(self, ):
        pass

    def run_analyzer(self, analyzer: Analyzer):
        pass

    def run_uid(self, uid: str):
        pass

    def add_analyzer(self, settings:):
        pass

    def generate_qf_analyzer(self, uid):
        pass


    def prepare_stock_data_for_training(self, stocks: StocksInUse):
        if stocks.stocks is None:
            raise TypeError("stocks not initialized")
        # Create individual DataFrames for each stock
        stock_dataframes = stocks.to_data_frame()
        # Concatenate individual DataFrames into a single DataFrame


        # Feature engineering and normalization
        feature_engineer = FeatureEngineer(combined_data)
        engineered_data = feature_engineer.engineer_features()

        normalizer = DataNormalizer(engineered_data)
        normalized_data = normalizer.normalize_data()

        return normalized_data
class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def engineer_features(self):
        self.data['CashFlow_to_BookValue'] = self.data['CashFlow'] / \
            self.data['BookValue']
        self.data['Earnings_to_BookValue'] = self.data['Earnings'] / \
            self.data['BookValue']
        self.data['CashFlow_to_Earnings'] = self.data['CashFlow'] / \
            self.data['Earnings']

        return self.data


class DataNormalizer:
    def __init__(self, data):
        self.data = data

    def normalize_data(self):
        columns_to_normalize = ['CashFlow', 'BookValue', 'Earnings',
                                'CashFlow_to_BookValue', 'Earnings_to_BookValue', 'CashFlow_to_Earnings']
        scaler = StandardScaler()
        self.data[columns_to_normalize] = scaler.fit_transform(
            self.data[columns_to_normalize])

        return self.data