
from value_investing_strategy.strategy_system.strategy_system import StrategySystem
from value_investing_strategy.strategy_system.stocks.stock import stock

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


    def prepare_stock_data_for_training(self, stocks: List[stock]):
        # Create individual DataFrames for each stock
        stock_dataframes = [stock.to_dataframe() for stock in stocks]

        # Concatenate individual DataFrames into a single DataFrame
        combined_data = pd.concat(stock_dataframes, ignore_index=True)

        # Feature engineering and normalization
        feature_engineer = FeatureEngineer(combined_data)
        engineered_data = feature_engineer.engineer_features()

        normalizer = DataNormalizer(engineered_data)
        normalized_data = normalizer.normalize_data()

        return normalized_data
