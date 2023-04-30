import pandas as pd
from sklearn.preprocessing import StandardScaler


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


# Example usage
data = pd.DataFrame({'CashFlow': [100, 200, 150], 'BookValue': [
                    500, 600, 450], 'Earnings': [50, 120, 80]})

feature_engineer = FeatureEngineer(data)
engineered_data = feature_engineer.engineer_features()

normalizer = DataNormalizer(engineered_data)
normalized_data = normalizer.normalize_data()
print(normalized_data)
